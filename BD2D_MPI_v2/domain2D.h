#pragma once

#include <fstream>
#include "config.h"
#include "cellList2D.h"
#include "rand.h"
#include "force2D.h"
#include "boundary2D.h"
#include "integrate2D.h"
#include "particle2D.h"
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
                 const PairForce& f12, const BdyCondi& bc, bool need_ori);

  template <typename TPar, typename TInteg, typename BdyCondi, typename TRan>
  void integrate(std::vector<BiNode<TPar>>& p_arr, CellList_2<TPar>& cl,
                 const TInteg& integrator, const BdyCondi& bc, TRan& myran);

  template <typename TPar>
  void shell_sorting(std::vector<BiNode<TPar>>& p_arr, CellList_2<TPar>& cl, int interval_steps) const;

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
                     const PairForce& f12, const BdyCondi& bc, bool need_ori){
  auto pack = [&cl, need_ori](double* buf, int idx)->int {
    if (need_ori) {
      return cl.pack_pos_ori(buf, cl.get_inner_edge(idx));
    } else {
      return cl.pack_pos(buf, cl.get_inner_edge(idx));
    }
  };
  auto unpack = [&cl, &p_arr, need_ori](const double* buf, int buf_size) {
    if (need_ori) {
      cl.unpack_pos_ori(buf, buf_size, p_arr);

    } else {
      cl.unpack_pos(buf, buf_size, p_arr);
    }
  };

  int par_num0 = static_cast<int>(p_arr.size());
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

  //sort_ascending();
  cl.compact(p_arr, vacancy);

  //for (auto it = p_arr.begin(); it != p_arr.end(); ++it) {
  //  if (box_.out((*it).pos)) {
  //    std::cout << (*it).pos << " is out of box " << box_ << "; ix=" << cl.get_idx_x(*it)
  //      << "; iy=" << cl.get_idx_y(*it) << "; with lc=" << cl.get_lc() << std::endl;
  //    std::cout << "outer_edge: " << cl.get_outer_edge(0);
  //    exit(1);
  //  }
  //}
}

template<typename TPar>
void Domain_2::shell_sorting(std::vector<BiNode<TPar>>& p_arr, CellList_2<TPar>& cl,
                             int interval_steps) const{
  static int step_count = 0;
  if (step_count == interval_steps) {
    step_count = 0;
    
    int n_layer = cl.get_real_n().x;
    double inverse_l = n_layer / box_.l.x;
    int* np_layer = new int[n_layer] {};
    int n_par = p_arr.size();
    int* i_layer_par = new int[n_par] {};

    int copy_size = 0;
    if (typeid(TPar) == typeid(BP_2)) {
      copy_size = 2;
    } else if (typeid(TPar) == typeid(BP_theta_2) ||
      typeid(TPar) == typeid(BP_theta_tau_2)) {
      copy_size = 3;
    } else if (typeid(TPar) == typeid(BP_u_2) ||
      typeid(TPar) == typeid(BP_u_tau_2)) {
      copy_size = 4;
    } else {
      std::cout << "Wrong particle type when reading from file\n";
      exit(2);
    }
    double* buf = new double[n_par * copy_size];
    int buf_pos = 0;
    for (int i = 0; i < n_par; i++) {
      int i_layer = int((p_arr[i].pos.x - box_.o.x) * inverse_l);
      np_layer[i_layer] += 1;
      i_layer_par[i] = i_layer;
      p_arr[i].copy_to(buf, buf_pos);
    }
    
    int* first_par_layer = new int[n_layer];
    first_par_layer[0] = 0;
    for (int i = 1; i < n_layer; i++) {
      first_par_layer[i] = first_par_layer[i - 1] + np_layer[i - 1];
    }

    buf_pos = 0;
    for (int i = 0; i < n_par; i++) {
      int k = i_layer_par[i];
      int ip_new = first_par_layer[k] + np_layer[k] - 1;
      p_arr[ip_new].copy_from(buf, buf_pos);
      np_layer[k] -= 1;
    }
    delete[] i_layer_par;
    delete[] np_layer;
    delete[] buf;
    delete[] first_par_layer;

    cl.recreate(p_arr);
  } else {
    step_count++;
  }
}
