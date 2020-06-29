#pragma once
#include "config.h"
#include "vect.h"
#include "node.h"
#include "mpi.h"
#include <vector>
#include <deque>
#include <iostream>
#include <functional>
#include <algorithm>

enum { x_neg, x_pos, y_neg, y_pos };

template <typename T>
class Box_2 {
public:
  Box_2() : l(), o() {}
  Box_2(const Vec_2<T>& ll, const Vec_2<T>& oo) : l(ll), o(oo) {}
  Vec_2<T> get_end() const { return l + o; }

  bool within(const Vec_2<T>& p) const {
    Vec_2<T> r = l + o;
    return p.x >= o.x && p.x < r.x && p.y >= o.y && p.y < r.y;
  }

  bool out(const Vec_2<T>& p) const {
    Vec_2<T> r = l + o;
    return p.x < o.x || p.x >= r.x || p.y < o.y || p.y >= r.y;
  }

  friend std::ostream& operator << (std::ostream& out, const Box_2<T>& box) {
    out << "(" << box.o.x << ", " << box.o.y << ", " << box.l.x << ", " << box.l.y << ")";
    return out;
  }
  Vec_2<T> l;
  Vec_2<T> o;
};

typedef Box_2<int> Block_2;

template <typename UniFunc>
void for_each(const Block_2& b, const Vec_2<int>& dims, UniFunc f) {
  const Vec_2<int> end = b.get_end();
  for (int iy = b.o.y; iy < end.y; iy++) {
    const int nx_iy = dims.x * iy;
    for (int ix = b.o.x; ix < end.x; ix++) {
      f(ix + nx_iy);
    }
  }
}

class Mesh_2 {
public:
  Mesh_2(const Box_2<double>& box, double r_cut,
    const Vec_2<double>& gl_l, const Vec_2<int>& proc_size);

  template <typename TPar>
  int get_idx_x(const TPar& p) const {
    return int(inv_cell_l_.x * (p.pos.x - box_.o.x));
  }
  template <typename TPar>
  int get_idx_y(const TPar& p) const {
    return int(inv_cell_l_.y * (p.pos.y - box_.o.y));
  }
  template <typename TPar>
  int get_idx(const TPar& p) const {
    return get_idx_x(p) + n_.x * get_idx_y(p); }

  int get_tot() const { return n_.x * n_.y; }

  void cal_pos_offset(Vec_2<double>& offset, const Vec_2<double>& pos) const;

  void set_comm_shell();

  void show_info() const;

  const Vec_2<double>& get_o() const { return box_.o; }
  const Vec_2<double>& get_l() const { return box_.l; }
  const Vec_2<double>& get_lc() const {return cell_l_;}

  const Block_2& get_inner_edge(int idx) const { return inner_edge_[idx]; }
  const Block_2& get_outer_edge(int idx) const { return outer_edge_[idx]; }

  const Vec_2<int>& get_n() const { return n_; }
  Vec_2<int> get_real_n() const;
protected:
  int n_tot_;
  Vec_2<int> n_;
  Vec_2<double> cell_l_;
  Vec_2<double> inv_cell_l_;
  Box_2<double> box_;
  Box_2<double> real_box_;
  Vec_2<double> gl_l_;
  Vec_2<bool> flag_pad_;

  Block_2 inner_edge_[4]{};
  Block_2 outer_edge_[4]{};
};

template <typename TPar>
class CellList_2 : public Mesh_2 {
public:
  typedef BiNode<TPar> node_t;

  CellList_2(const Box_2<double>& box, double r_cut,
    const Vec_2<double>& gl_l, const Vec_2<int>& proc_size)
    : Mesh_2(box, r_cut, gl_l, proc_size), head_(n_tot_) {};

  void create(std::vector<node_t>& p_arr);
  void recreate(std::vector<node_t>& p_arr);
  void add_node(node_t& p);

  void compact(std::vector<node_t>& p_arr, std::deque<int>& vacancy);

  int pack_pos(double* buf, const Block_2& block) const;
  void unpack_pos(const double* buf, int buf_size, std::vector<node_t>& p_arr);

  int pack_pos_ori(double* buf, const Block_2& block) const;
  void unpack_pos_ori(const double* buf, int buf_size, std::vector<node_t>& p_arr);

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
void CellList_2<TPar>::create(std::vector<node_t>& p_arr) {
  auto end = p_arr.end();
  for (auto it = p_arr.begin(); it != end; ++it) {
    //if (out_range(*it)) {
    //  std::cout << "Error when create particles with pos=" << (*it).pos << std::endl;
    //  exit(4);
    //}
    add_node(*it);
  }
}

template<typename TPar>
void CellList_2<TPar>::recreate(std::vector<node_t>& p_arr) {
  for (int ic = 0; ic < n_tot_; ic++) {
    head_[ic] = nullptr;
  }
  create(p_arr);
}

template<typename TPar>
void CellList_2<TPar>::add_node(node_t& p) {
  //if (!box_.within(p.pos)) {
  //  std::cout << p.pos << std::endl;
  //  exit(5);
  //}
  auto ic = get_idx(p);
  p.append_at_front(&head_[ic]);
}

template<typename TPar>
template<typename BiFunc1, typename BiFunc2>
void CellList_2<TPar>::for_each_pair(BiFunc1 f1, BiFunc2 f2) const {
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
void CellList_2<TPar>::cal_pair_force(const TPairForce& f12, const BdyCondi& bc) const {
  auto f1 = [&f12](node_t* p1, node_t* p2) {
    f12(*p1, *p2);
  };

  auto f2 = [&f12, &bc](node_t* p1, node_t* p2) {
    f12(*p1, *p2, bc);
  };
  for_each_pair(f1, f2);
}

template <typename TPar>
void CellList_2<TPar>::compact(std::vector<node_t>& p_arr, std::deque<int>& vacancy) {
  //! vacancy should be sorted in ascending order before calling this function
  //! std::sort(vacancy.begin(), vacancy.end(), std::less<int>());
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
int CellList_2<TPar>::pack_pos(double* buf, const Block_2& block) const {
  int buf_pos = 0;
#ifdef USE_LAMBDA
  auto f = [this, &buf_pos, buf](int i) {
    node_t* head_node = head_[i];
    if (head_node) {
      node_t* cur_node = head_node;
      do {
        cur_node->copy_pos_to(buf, buf_pos);
        cur_node = cur_node->next;
      } while (cur_node);
    }
  };
  for_each(block, n_, f);
#else
  const Vec_2<int>& beg = block.o;
  const Vec_2<int> end = block.get_end();
  for (int iy = beg.y; iy < end.y; iy++) {
    int nx_iy = n_.x * iy;
    for (int ix = beg.x; ix < end.x; ix++) {
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
#endif
  return buf_pos;
}

template<typename TPar>
void CellList_2<TPar>::unpack_pos(const double* buf, int buf_size, std::vector<node_t>& p_arr) {
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
int CellList_2<TPar>::pack_pos_ori(double* buf, const Block_2& block) const {
  int buf_pos = 0;
#ifdef USE_LAMBDA
  auto f = [this, &buf_pos, buf](int i) {
    node_t* head_node = head_[i];
    if (head_node) {
      node_t* cur_node = head_node;
      do {
        cur_node->copy_to(buf, buf_pos);
        cur_node = cur_node->next;
      } while (cur_node);
    }
  };
  for_each(block, n_, f);
#else
  const Vec_2<int>& beg = block.o;
  const Vec_2<int> end = block.get_end();
  for (int iy = beg.y; iy < end.y; iy++) {
    int nx_iy = n_.x * iy;
    for (int ix = beg.x; ix < end.x; ix++) {
      node_t* head_node = head_[ix + nx_iy];
      if (head_node) {
        node_t* cur_node = head_node;
        do {
          cur_node->copy_to(buf, buf_pos);
          cur_node = cur_node->next;
        } while (cur_node);
      }
    }
  }
#endif
  return buf_pos;
}

template<typename TPar>
void CellList_2<TPar>::unpack_pos_ori(const double* buf, int buf_size, std::vector<node_t>& p_arr) {
                                  Vec_2<double> offset{};
  cal_pos_offset(offset, Vec_2<double>(buf[0], buf[1]));
  int buf_pos = 0;
  while (buf_pos < buf_size) {
    p_arr.emplace_back();
    auto& p = p_arr.back();
    p.copy_from(buf, buf_pos);
    p.pos += offset;
    add_node(p);
  }
}

template<typename TPar>
int CellList_2<TPar>::pack_leaving_par(double* buf, const Block_2& block,
  const std::vector<node_t>& p_arr, std::deque<int>& vacancy) {
  int buf_pos = 0;
  const node_t* p0 = &p_arr[0];
#ifdef USE_LAMBDA
  auto f = [this, p0, &buf_pos, buf, &vacancy](int i) {
    node_t* head_node = head_[i];
    if (head_node) {
      node_t* cur_node = head_node;
      do {
        cur_node->copy_to(buf, buf_pos);
        vacancy.push_back(static_cast<int>(cur_node - p0));
        cur_node = cur_node->next;
      } while (cur_node);
      head_[i] = nullptr;
    }
  };
  for_each(block, n_, f);
#else
  const Vec_2<int>& beg = block.o;
  const Vec_2<int> end = block.get_end();
  for (int iy = beg.y; iy < end.y; iy++) {
    const int nx_iy = n_.x * iy;
    for (int ix = beg.x; ix < end.x; ix++) {
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
#endif
  return buf_pos;
}

template<typename TPar>
void CellList_2<TPar>::unpack_arrived_par(const double* buf, int buf_size,
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
    //! when communicate along both direction, the following case is possible to happen
    //if (real_box_.out(p->pos)) {
    //  std::cout << "after unpack, pos=" << p->pos << " is out of " << real_box_ << std::endl;
    //  exit(10);
    //}
    add_node(*p);
  }
}

template<typename TPar>
void CellList_2<TPar>::clear(const Block_2& b) {
#ifdef USE_LAMBDA
  auto f = [this](int i) { head_[i] = nullptr; };
  for_each(b, n_, f);
#else
  const Vec_2<int>& beg = b.o;
  const Vec_2<int> end = b.get_end();
  for (int iy = beg.y; iy < end.y; iy++) {
    const int nx_iy = n_.x * iy;
    for (int ix = beg.x; ix < end.x; ix++) {
      head_[ix + nx_iy] = nullptr;
    }
  }
#endif
}
