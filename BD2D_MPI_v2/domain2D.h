#pragma once

#include "mpi.h"
#include "node.h"
#include "rand.h"
#include "force2D.h"
#include "boundary2D.h"
#include "integrate2D.h"
#include <iostream>
#include <vector>
#include <deque>
#include <functional>
#include <algorithm>

enum {x_neg, x_pos, y_neg, y_pos};
class Box_2 {
public:
  Box_2() : l(), o() {}
  Box_2(const Vec_2<double>& ll, const Vec_2<double>& oo) : l(ll), o(oo) {}
  Box_2(const Vec_2<double>& gl_l, const Vec_2<int>& proc_rank, const Vec_2<int>& proc_size);
  void set_box(const Vec_2<double>& gl_l, const Vec_2<int>& proc_rank, const Vec_2<int>& proc_size);

  Vec_2<double> l;
  Vec_2<double> o;
};

struct Block_2 {
  void set_value(const Vec_2<int>& o, const Vec_2<int>& n) { beg = o; end = o + n; }
  Vec_2<int> beg{}; // the lower-left corner of the block
  Vec_2<int> end{}; // the upper-right corner of the block, not include
};

class Mesh_2 {
public:
  Mesh_2(const Box_2& box, double r_cut,
    const Vec_2<double>& gl_l, const Vec_2<int>& proc_size);

  template <typename TPar>
  int get_idx_x(const TPar& p) const {
    return int(inv_cell_l_.x * (p.pos.x - o_.x));
  }
  template <typename TPar>
  int get_idx_y(const TPar& p) const {
    return int(inv_cell_l_.y * (p.pos.y - o_.y));
  }
  template <typename TPar>
  int get_idx(const TPar& p) const { return get_idx_x(p) + n_.x * get_idx_y(p); }

  int get_tot() const { return n_.x * n_.y; }

  void cal_pos_offset(Vec_2<double>& offset, const Vec_2<double>& pos) const;

  void set_comm_shell();

  const Vec_2<double>& get_o() const { return o_; }
  const Vec_2<double>& get_l() const { return l_; }

  const Block_2& get_inner_edge(int idx) const { return inner_edge_[idx]; }
  const Block_2& get_outer_edge(int idx) const { return outer_edge_[idx]; }
protected:
  int n_tot_;
  Vec_2<int> n_;
  Vec_2<double> cell_l_;
  Vec_2<double> inv_cell_l_;
  Vec_2<double> o_;
  Vec_2<double> l_;
  Vec_2<double> gl_l_;
  Vec_2<bool> flag_pad_;

  Block_2 inner_edge_[4]{};
  Block_2 outer_edge_[4]{};
};

template <typename TPar>
class CList_2 : public Mesh_2 {
public:
  typedef BiNode<TPar> node_t;

  CList_2(const Box_2& box, double r_cut, const Vec_2<double>& gl_l, const Vec_2<int>& proc_size);

  void create(std::vector<node_t>& p_arr);
  void recreate(std::vector<node_t>& p_arr);
  //bool recreate_debug(std::vector<node_t>& p_arr);
  void add_node(node_t& p);

  void compact(std::vector<node_t>& p_arr, std::deque<int>& vacancy);

  int pack_pos(double* buf, const Block_2& block) const;

  void unpack_pos(const double* buf, int buf_size, std::vector<node_t>& p_arr);

  int pack_leaving_par(double* buf, const Block_2& block,
    const std::vector<node_t>& p_arr, std::deque<int>& vacancy);

  void unpack_arrived_par(const double* buf, int buf_size,
    std::vector<node_t>& p_arr, std::deque<int>& vacancy);

  template <typename BiFunc1, typename BiFunc2>
  void for_each_pair(BiFunc1 f1, BiFunc2 f2) const;

  template <typename TPairForce, typename BdyCondi>
  void cal_pair_force(const TPairForce& f12, const BdyCondi& bc) const;

  void clear(const Block_2& b);

protected:
  std::vector<node_t*> head_;
};

template<typename TPar>
CList_2<TPar>::CList_2(const Box_2& box, double r_cut, const Vec_2<double>& gl_l,
  const Vec_2<int>& proc_size) : Mesh_2(box, r_cut, gl_l, proc_size), head_(n_tot_) {
  
  //std::cout << flag_pad_ << std::endl;
  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  if (myrank == 0) {
    std::cout << outer_edge_[0].beg << "; " << outer_edge_[0].end << std::endl;
    std::cout << outer_edge_[1].beg << "; " << outer_edge_[1].end << std::endl;

  }
}

template<typename TPar>
void CList_2<TPar>::create(std::vector<node_t>& p_arr) {
  auto end = p_arr.end();
  for (auto it = p_arr.begin(); it != end; ++it) {
    add_node(*it);
  }
}

template<typename TPar>
void CList_2<TPar>::recreate(std::vector<node_t>& p_arr) {
  for (int ic = 0; ic < n_tot_; ic++) {
    head_[ic] = nullptr;
  }
  create(p_arr);
}

template<typename TPar>
void CList_2<TPar>::add_node(node_t& p) {
  auto ic = get_idx(p);
  p.append_at_front(&head_[ic]);
}

template<typename TPar>
template<typename BiFunc1, typename BiFunc2>
void CList_2<TPar>::for_each_pair(BiFunc1 f1, BiFunc2 f2) const {
  //TODO optimize by avoiding unnecessary cases
  int y_end = n_.y;
  if (flag_pad_.y) {
    y_end -= 1;
  }
  for (int yc = 0; yc < y_end; yc++) {
    int nx_yc = yc * n_.x;
    for (int xc = 0; xc < n_.x; xc++) {
      node_t* h0 = head_[xc + nx_yc];
      if (h0) {
        for_each_node_pair(h0, f1);
        int xl = xc - 1;
        int xr = xc + 1;
        int yu = yc + 1;
        if (xl == -1) {
          xl += n_.x;
        }
        if (xr == n_.x) {
          xr = 0;
        }
        if (!flag_pad_.y && yu >= n_.y) {
          yu = 0;
        }
        int nx_yu = yu * n_.x;
        node_t* h1 = head_[xr + nx_yc];
        if (h1) {
          for_each_node_pair(h0, h1, f2);
        }
        node_t* h2 = head_[xl + nx_yu];
        if (h2) {
          for_each_node_pair(h0, h2, f2);
        }
        node_t* h3 = head_[xc + nx_yu];
        if (h3) {
          for_each_node_pair(h0, h3, f2);
        }
        node_t* h4 = head_[xr + nx_yu];
        if (h4) {
          for_each_node_pair(h0, h4, f2);
        }
      }
    }
  }
}

template<typename TPar>
template<typename TPairForce, typename BdyCondi>
void CList_2<TPar>::cal_pair_force(const TPairForce& f12, const BdyCondi& bc) const {
  auto f1 = [&f12](node_t* p1, node_t* p2) {
    f12(*p1, *p2);
  };

  auto f2 = [&f12, &bc](node_t* p1, node_t* p2) {
    f12(*p1, *p2, bc);
  };
  for_each_pair(f1, f2);
}

template <typename TPar>
void CList_2<TPar>::compact(std::vector<node_t>& p_arr, std::deque<int>& vacancy) {
  //! vacancy should be sorted in ascending order before calling this function
  //! std::sort(vacancy.begin(), vacancy.end(), std::<int>());
  while (!vacancy.empty()) {
    if (vacancy.back() == p_arr.size() - 1) {
      p_arr.pop_back();
      vacancy.pop_back();
    } else {
      node_t* p = &p_arr[vacancy.front()];
      vacancy.pop_front();
      *p = p_arr.back();
      p_arr.pop_back();
      if (p->next) {
        p->next->prev = p;
      }
      if (p->prev) {
        p->prev->next = p;
      } else {
        head_[get_idx(*p)] = p;
      }
    }
  }
}

template<typename TPar>
int CList_2<TPar>::pack_pos(double* buf, const Block_2& block) const {
  int buf_pos = 0;
  for (int iy = block.beg.y; iy < block.end.y; iy++) {
    int nx_iy = n_.x * iy;
    for (int ix = block.beg.x; ix < block.end.x; ix++) {
      node_t* head_node = head_[ix + nx_iy];
      if (head_node) {
        node_t* cur_node = head_node;
        do {
          cur_node->copy_pos_to(buf, buf_pos);
          cur_node = cur_node->next;
        } while (cur_node);
      }
    }
  }
  return buf_pos;
}

template<typename TPar>
void CList_2<TPar>::unpack_pos(const double* buf, int buf_size, std::vector<node_t>& p_arr) {
  Vec_2<double> offset{};
  cal_pos_offset(offset, Vec_2<double>(buf[0], buf[1]));
  int buf_pos = 0;
  while (buf_pos < buf_size) {
    p_arr.emplace_back();
    auto& p = p_arr.back();
    p.copy_pos_from(buf, buf_pos);
    p.pos += offset;
    add_node(p);
  }
}

template<typename TPar>
int CList_2<TPar>::pack_leaving_par(double* buf, const Block_2& block,
  const std::vector<node_t>& p_arr, std::deque<int>& vacancy) {
  int buf_pos = 0;
  const node_t* p0 = &p_arr[0];
  for (int iy = block.beg.y; iy < block.end.y; iy++) {
    const int nx_iy = n_.x * iy;
    for (int ix = block.beg.x; ix < block.end.x; ix++) {
      const int idx = ix + nx_iy;
      node_t* head_node = head_[idx];
      if (head_node) {
        node_t* cur_node = head_node;
        do {
          cur_node->copy_to(buf, buf_pos);
          vacancy.push_back(cur_node - p0);
          cur_node = cur_node->next;
        } while (cur_node);
        head_[idx] = nullptr;
      }
    }
  }
  return buf_pos;
}

template<typename TPar>
void CList_2<TPar>::unpack_arrived_par(const double* buf, int buf_size,
  std::vector<node_t>& p_arr, std::deque<int>& vacancy) {
  //! vacancy should be sorted in ascending order before calling this function
  //! std::sort(vacancy.begin(), vacancy.end(), std::<int>());
  Vec_2<double> offset{};
  cal_pos_offset(offset, Vec_2<double>(buf[0], buf[1]));
  int buf_pos = 0;
  while (buf_pos < buf_size) {
    node_t* p = nullptr;
    if (vacancy.empty()) {
      p_arr.emplace_back();
      p = &p_arr.back();
    } else {
      p = &p_arr[vacancy.front()];
      vacancy.pop_front();
    }
    p->copy_from(buf, buf_pos);
    p->pos += offset;
    if (p->pos.x < o_.x || p->pos.x >= o_.x + l_.x) {
      std::cout << "error when unpack" << std::endl;
    }
    add_node(*p);
  }
}

template<typename TPar>
void CList_2<TPar>::clear(const Block_2& b) {
  for (int iy = b.beg.y; iy < b.end.y; iy++) {
    const int nx_iy = n_.x * iy;
    for (int ix = b.beg.x; ix < b.end.x; ix++) {
      head_[ix + nx_iy] = nullptr;
    }
  }
}


class Domain_2 {
public:
  Domain_2(const Vec_2<double>& gl_l, const Vec_2<int>& proc_size_vec, MPI_Comm comm);
  ~Domain_2();

  const Vec_2<double>& get_gl_l() const { return gl_l_; }
  const Box_2& get_box() const { return box_; }
  MPI_Comm get_comm() const { return comm_; }
  const Vec_2<int> get_proc_rank_vec() const { 
    return Vec_2<int>(my_rank_ % proc_size_vec_.x, my_rank_ / proc_size_vec_.x); }
  const Vec_2<int>& get_proc_size_vec() const { return proc_size_vec_; }
  int get_proc_rank() const { return my_rank_; }
  int get_proc_size() const { return tot_proc_; }

  void find_neighbor_domain(int dir);
  void set_buf(double r_cut=1., double amplification = 5., int size_of_one_par = 4);
  template <typename FuncPack, typename FuncUnpack, typename FuncDoSth>
  void communicate(int dir, FuncPack pack, FuncUnpack unpack, FuncDoSth do_sth);

  template <typename TPar, typename PairForce, typename BdyCondi>
  void cal_force(std::vector<BiNode<TPar>>& p_arr, CList_2<TPar>& cl, 
                 const PairForce& f12, const BdyCondi& bc);
  template <typename TPar, typename TInteg, typename BdyCondi, typename TRan>
  void integrate(std::vector<BiNode<TPar>>& p_arr, CList_2<TPar>& cl,
                 const TInteg& integrator, const BdyCondi& bc, TRan& myran);

  template <typename TPar, typename BdyCondi>
  void ini_rand(std::vector<BiNode<TPar>>& p_arr, int n_par_gl,
                const BdyCondi& bc, double sigma = 1.);
protected:
  Vec_2<double> gl_l_;
  Box_2 box_;
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
void Domain_2::cal_force(std::vector<BiNode<TPar>>& p_arr, CList_2<TPar>& cl,
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
void Domain_2::integrate(std::vector<BiNode<TPar>>& p_arr, CList_2<TPar>& cl,
  const TInteg& integrator, const BdyCondi& bc, TRan& myran) {
  const auto end = p_arr.end();
  for (auto it = p_arr.begin(); it != end; ++it) {
    integrator.update(*it, bc, myran);
  }
  cl.recreate(p_arr);
  //if (cl.recreate_debug(p_arr)) {
  //  exit(4);
  //}

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
  Ranq2 myran(1);
  if (n_par_gl < box_.l.x * box_.l.y / (sigma * sigma) / 2) {
    create_rand_par_2(p_arr, n_par, origin, l, bc, myran, sigma);
  } else {
    double r_cut = sigma;
    set_buf(r_cut, 10);
    EM_BD_iso integrator(1e-4);
    SpringForce_2 pair_force(500, sigma);
    CList_2<TPar> cl(box_, r_cut, gl_l_, proc_size_vec_);
    double sigma_new = 0.5 * sigma;
    create_rand_par_2(p_arr, n_par, origin, l, bc, myran, sigma_new);
    cl.create(p_arr);
    std::cout << "create cell list" << std::endl;
    do {
      //std::cout << "sigma=" << sigma_new << std::endl;
      sigma_new += 0.01;
      pair_force.set_sigma(sigma_new);
      for (int i = 0; i < 500; i++) {
        cal_force(p_arr, cl, pair_force, bc);
        integrate(p_arr, cl, integrator, bc, myran);
      }
    } while (sigma_new < sigma);
  }
  if (my_rank_ == 0) {
    std::cout << "initialized randomly!" << std::endl;
  }
}
