#pragma once

#include <fstream>
#include "config.h"
#include "cellList2D.h"
#include "rand.h"
#include "force2D.h"
#include "boundary2D.h"
#include "integrate2D.h"
#include "mpi.h"

class Domain_2 {
public:
  Domain_2(const Vec_2<double>& gl_l, const Vec_2<int>& proc_size_vec, MPI_Comm comm);
  ~Domain_2();

  const Vec_2<double>& get_gl_l() const { return gl_l_; }
  const Box_2<double>& get_box() const { return box_; }
  MPI_Comm get_comm() const { return comm_; }
  const Vec_2<int> get_proc_rank_vec() const { 
    return Vec_2<int>(my_rank_ % proc_size_vec_.x, my_rank_ / proc_size_vec_.x); }
  const Vec_2<int>& get_proc_size_vec() const { return proc_size_vec_; }
  int get_proc_rank() const { return my_rank_; }
  int get_proc_size() const { return tot_proc_; }

  void find_neighbor_domain(int dir);

  void find_neighbor();

  void set_buf(double r_cut=1., double amplification = 5., int size_of_one_par = 4);
  template <typename FuncPack, typename FuncUnpack, typename FuncDoSth>
  void communicate(int dir, FuncPack pack, FuncUnpack unpack, FuncDoSth do_sth);

  template <typename TPar, typename PairForce, typename BdyCondi>
  void cal_force(std::vector<BiNode<TPar>>& p_arr, CellList_2<TPar>& cl, 
                 const PairForce& f12, const BdyCondi& bc);
  template <typename TPar, typename TInteg, typename BdyCondi, typename TRan>
  void integrate(std::vector<BiNode<TPar>>& p_arr, CellList_2<TPar>& cl,
                 const TInteg& integrator, const BdyCondi& bc, TRan& myran);

  template <typename TPar, typename BdyCondi>
  void ini_rand(std::vector<BiNode<TPar>>& p_arr, int n_par_gl,
                const BdyCondi& bc, double sigma = 1.);

  template <typename TPar>
  void ini_from_file(const std::string& file_in, std::vector<BiNode<TPar>>& p_arr,
                     int n_par_gl, double sigma=1.);

protected:
  Vec_2<double> gl_l_;
  Box_2<double> box_;
  MPI_Comm comm_;
  Vec_2<int> proc_size_vec_;
  int my_rank_;
  int tot_proc_;

  Vec_2<bool> flag_comm_{};
  int neighbor_[4]{};
  double* buf_[4]{};
  int buf_size_[4]{};
  int max_buf_size_ = 0;
};

template<typename FuncPack, typename FuncUnpack, typename FuncDoSth>
void Domain_2::communicate(int dir, FuncPack pack, FuncUnpack unpack, FuncDoSth do_sth) {
  MPI_Request req[4];
  MPI_Status stat[4];
  int prev_idx = dir * 2;
  int next_idx = dir * 2 + 1;
  int prev_proc = neighbor_[prev_idx];
  int next_proc = neighbor_[next_idx];

  //! transfer data backward
  MPI_Irecv(buf_[0], max_buf_size_, MPI_DOUBLE, next_proc, 21, comm_, &req[0]);
  buf_size_[1] = pack(buf_[1], prev_idx);
  MPI_Isend(buf_[1], buf_size_[1], MPI_DOUBLE, prev_proc, 21, comm_, &req[1]);
  //! transfer data forward
  MPI_Irecv(buf_[2], max_buf_size_, MPI_DOUBLE, prev_proc, 12, comm_, &req[2]);
  buf_size_[3] = pack(buf_[3], next_idx);
  MPI_Isend(buf_[3], buf_size_[3], MPI_DOUBLE, next_proc, 12, comm_, &req[3]);

  //! do something while waiting
  do_sth();

  //! receive the data from next proc
  MPI_Wait(&req[0], &stat[0]);
  MPI_Get_count(&stat[0], MPI_DOUBLE, &buf_size_[0]);
  unpack(buf_[0], buf_size_[0]);
  //! receive the data from prev proc
  MPI_Wait(&req[2], &stat[2]);
  MPI_Get_count(&stat[2], MPI_DOUBLE, &buf_size_[2]);
  unpack(buf_[2], buf_size_[2]);
  MPI_Wait(&req[1], &stat[1]);
  MPI_Wait(&req[3], &stat[3]);
}

template <typename TPar, typename PairForce, typename BdyCondi>
void Domain_2::cal_force(std::vector<BiNode<TPar>>& p_arr, CellList_2<TPar>& cl,
                     const PairForce& f12, const BdyCondi& bc){
  auto pack = [&cl](double* buf, int idx)->int {
    return cl.pack_pos(buf, cl.get_inner_edge(idx));
  };
  auto unpack = [&cl, &p_arr](const double* buf, int buf_size) {
    cl.unpack_pos(buf, buf_size, p_arr);
  };

  int par_num0 = p_arr.size();
  if (flag_comm_.x) {
    communicate(0, pack, unpack, []() {});
  }
  if (flag_comm_.y) {
    communicate(1, pack, unpack, []() {});
  }

  // cal all pair force
  cl.cal_pair_force(f12, bc);

  // clear ghost particles
  if (flag_comm_.x) {
    cl.clear(cl.get_outer_edge(0));
    cl.clear(cl.get_outer_edge(1));
  }
  if (flag_comm_.y) {
    cl.clear(cl.get_outer_edge(2));
    cl.clear(cl.get_outer_edge(3));
  }
  while (p_arr.size() > par_num0) {
    p_arr.pop_back();
  }
}

template <typename TPar, typename TInteg, typename BdyCondi, typename TRan>
void Domain_2::integrate(std::vector<BiNode<TPar>>& p_arr, CellList_2<TPar>& cl,
  const TInteg& integrator, const BdyCondi& bc, TRan& myran) {
  const auto end = p_arr.end();
  for (auto it = p_arr.begin(); it != end; ++it) {
    integrator.update(*it, bc, myran);
  }
  cl.recreate(p_arr);

  std::deque<int> vacancy;
  auto pack = [&p_arr, &cl, &vacancy](double* buf, int idx)->int {
    return cl.pack_leaving_par(buf, cl.get_outer_edge(idx), p_arr, vacancy);
  };

  auto unpack = [&p_arr, &cl, &vacancy](const double* buf, int buf_size) {
    cl.unpack_arrived_par(buf, buf_size, p_arr, vacancy);
  };

  auto sort_ascending = [&vacancy]() {
    std::sort(vacancy.begin(), vacancy.end(), std::less<int>());
  };

  if (flag_comm_.x) {
    communicate(0, pack, unpack, sort_ascending);
  }

  if (flag_comm_.y) {
    communicate(1, pack, unpack, sort_ascending);
  }

  sort_ascending();
  cl.compact(p_arr, vacancy);

  for (auto it = p_arr.begin(); it != p_arr.end(); ++it) {
    if (box_.out_of((*it).pos)) {
      std::cout << (*it).pos << " is out of box " << box_ << "; ix=" << cl.get_idx_x(*it)
        << "; iy=" << cl.get_idx_y(*it) << "; with lc=" << cl.get_lc() << std::endl;
      std::cout << "outer_edge: " << cl.get_outer_edge(0);
      exit(1);
    }
  }
}

template<typename TPar, typename BdyCondi>
void Domain_2::ini_rand(std::vector<BiNode<TPar>>& p_arr, int n_par_gl,
                    const BdyCondi& bc, double sigma){
  int n_max = int(box_.l.x * box_.l.y / (sigma * sigma) * 5);
  p_arr.reserve(n_max);
  int n_par;
  if (my_rank_ < tot_proc_ - 1) {
    n_par = n_par_gl / tot_proc_;
  } else {
    n_par = n_par_gl - n_par_gl / tot_proc_ * (tot_proc_ - 1);
  }
  if (my_rank_ == 0) {
    std::cout << "nPar = " << n_par << "for rank =" << my_rank_ << std::endl;
    //n_par -= 1;
  }
  MPI_Barrier(comm_);
  if (my_rank_ == 1) {
    std::cout << "nPar = " << n_par << "for rank =" << my_rank_ << std::endl;
    //n_par += 1;
  }
  MPI_Barrier(comm_);
  
  Vec_2<double> origin = box_.o;
  Vec_2<double> l = box_.l;
  if (proc_size_vec_.x > 1) {
    l.x -= sigma * 0.5;
    origin.x += sigma * 0.5;
  }
  if (proc_size_vec_.y > 1) {
    l.y -= sigma * 0.5;
    origin.y += sigma * 0.5;
  }
  Ranq2 myran(1 + my_rank_);
  if (n_par_gl < box_.l.x * box_.l.y / (sigma * sigma) / 2) {
    create_rand_par_2(p_arr, n_par, origin, l, bc, myran, sigma);
  } else {
    double r_cut = sigma;
    set_buf(r_cut, 10);
    EM_BD_iso integrator(1e-4);
    SpringForce_2 pair_force(500, sigma);
    CellList_2<TPar> cl(box_, r_cut, gl_l_, proc_size_vec_);
    double sigma_new = 0.5 * sigma;
    create_rand_par_2(p_arr, n_par, origin, l, bc, myran, sigma_new);

    cl.create(p_arr);
    do {
      //std::cout << "sigma=" << sigma_new << std::endl;
      sigma_new += 0.01;
      pair_force.set_sigma(sigma_new);
      MPI_Barrier(comm_);
      for (int i = 0; i < 500; i++) {
        //std::cout << "i = " << i << std::endl;
        cal_force(p_arr, cl, pair_force, bc);
        integrate(p_arr, cl, integrator, bc, myran);
      }
    } while (sigma_new < sigma);
  }
  if (my_rank_ == 0) {
    std::cout << "initialized randomly!\n";
    std::cout << "************************************\n" << std::endl;
  }
  MPI_Barrier(comm_);
}


template<typename TPar>
void Domain_2::ini_from_file(const std::string& file_in, std::vector<BiNode<TPar>>& p_arr, int n_par_gl, double sigma) {
  int n_max = int(box_.l.x * box_.l.y / (sigma * sigma) * 5);
  p_arr.reserve(n_max);

  std::ifstream fin(file_in, std::ios::binary);
  fin.seekg(0, std::ios::end);
  int file_size = fin.tellg();
  int buf_size = file_size / sizeof(float);
  if (file_size % sizeof(float) != 0 || buf_size % n_par_gl != 0) {
    std::cout << "Error when load " << file_in << ", wrong file size" << std::endl;
    exit(5);
  } 
  float* buf = new float[buf_size];
  fin.read((char*)buf, buf_size * sizeof(float));
  fin.close();
  int buf_pos = 0;
  while (buf_pos < buf_size) {
    TPar p;
    p.load_from_file(buf, buf_pos);
    if (p.pos.x <= 0) {
      p.pos.x += gl_l_.x;
    } else if (p.pos.x > gl_l_.x) {
      p.pos.x -= gl_l_.x;
    }
    if (p.pos.y <= 0) {
      p.pos.y += gl_l_.y;
    } else if (p.pos.y > gl_l_.y) {
      p.pos.y -= gl_l_.y;
    }
    if (box_.with_in(p.pos)) {
      p_arr.push_back(p);
    }
  }

  int my_n = p_arr.size();
  int tot_n;
  MPI_Reduce(&my_n, &tot_n, 1, MPI_INT, MPI_SUM, 0, comm_);
  if (my_rank_ == 0) {
    if (tot_n == n_par_gl) {
      std::cout << "Successed to load " << tot_n << " particles from " << file_in << std::endl;
    } else {
      std::cout << tot_n << "particles are loaded, while " << n_par_gl << " are required" << std::endl;
      exit(5);
    }
  }



}
