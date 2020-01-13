#pragma once
#include <vector>
#include <algorithm>
#include "vect.h"
#include "node.h"

/**
 * @brief Rectangular blocks in 2D.
 *
 * Represent a rectangular block by its lower-left and upper-right corners.
 *
 * @tparam T integer type
 */
template <typename IntType>
struct RectBlock_2 {
  RectBlock_2() = default;
  RectBlock_2(const Vec_2<IntType>& n) : beg(0, 0), end(n) {}
  Vec_2<IntType> beg;   // the lower-left corner of the block
  Vec_2<IntType> end;   // the upper-right corner of the block, not include
};

/**
 * @brief Base class of 2D celllist
 *
 * The simulation domain of length l_.x * l_.y is divided into n_.x * n_.y cells.
 *
 * If the whole domain is divided into several subdomains, then use flag_padded to
 * indicate whether the subdomain is supposed to communicate with its adjacent
 * subdomains. For example, flag_padded.x = true if the domain need communicate with
 * its left and right neighbors.
 */
class CellListBase_2 {
public:
  typedef Vec_2<double> Vec2d;

  template <class TDomain, class TGrid>
  CellListBase_2(const TDomain& dm, const TGrid& grid, int pad_w = 1);

  int get_nx(double x) const { return int((x - origin_.x) * lc_recip_.x); }
  int get_ny(double y) const { return int((y - origin_.y) * lc_recip_.y); }
  template <typename TPar>
  int get_ic(const TPar& p) const {
    return get_nx(p.pos.x) + get_ny(p.pos.y) * n_.x;
  }

  int n_cells() const { return n_cells_; }
  const Vec2d& origin() const { return origin_; }
  const Vec2d& l() const { return l_; }
  const Vec2d& gl_l() const { return gl_l_; }
  const Vec_2<bool>& flag_ext() const { return flag_padded; }
  const Vec_2<int>& cells_size() const { return n_; }

  template <typename T>
  int get_par_num(const std::vector<T>& n_arr) const;

  template <typename T>
  Vec_2<double> get_pos_offset(const Vec_2<T>& pos) const;
protected:
  int n_cells_;             //total number of cells of the padded domain..
  Vec_2<int> n_;            //the size of cells in x and y direction of the padded domain
  Vec_2<double> origin_;    //the upper-left point of the padded domain 
  Vec_2<double> lc_recip_;  //one over lc
  Vec_2<double> l_;         //the length in x and y direction of the padded domain
  Vec_2<double> gl_l_;      //the length in x and y direction of the global domain
  Vec_2<bool> flag_padded;  //whether the original domain is extended in x and y direction

  //the lower-left and upper-right corners of real cells w.r.t. the origin of the padded domain
  RectBlock_2<int> real_cells_;

  //the offset (dx, dy) betwwen one cell and its neighbors
  const int cell_offset[4][2] = { { 0,  1 },        // right
                                  { 1, -1 },        // upper-left
                                  { 1,  0 },        // upper
                                  { 1,  1 } };      // upper-right

  RectBlock_2<int> inner_shell_[4];
  RectBlock_2<int> outer_shell_[4];
};

template <class TDomain, class TGrid>
CellListBase_2::CellListBase_2(const TDomain& dm, const TGrid& grid, int pad_w)
                              :n_(grid.n()), origin_(dm.origin()),
                               lc_recip_(grid.inverse_lc()),
                               l_(dm.l()), gl_l_(dm.gl_l()),
                               flag_padded(dm.proc_size().x > 1, dm.proc_size().y > 1),
                               real_cells_(n_) {
  for (int dim = 0; dim < 2; dim++) {
    if (flag_padded[dim]) {
      origin_[dim] -= pad_w * grid.lc()[dim];
      n_[dim] += 2 * pad_w;
      l_[dim] += 2 * pad_w * grid.lc()[dim];
      real_cells_.beg[dim] = pad_w;
      real_cells_.end[dim] = n_[dim] - pad_w;
    }
  }
  n_cells_ = n_.x * n_.y;

  //TODO set inner and outter shells for communications

  //std::cout << "Success to create cell list with size " << n_ << std::endl;
}

template <typename T>
int CellListBase_2::get_par_num(const std::vector<T>& n_arr) const {
  int n_sum = 0;
  for (int row = real_cells_.beg.y; row < real_cells_.end.y; row++) {
    for (int col = real_cells_.beg.x; col < real_cells_.end.x; col++) {
      int ic = col + row * n_.x;
      n_sum += n_arr[ic];
    }
  }
  return n_sum;
}

template <typename T>
Vec_2<double> CellListBase_2::get_pos_offset(const Vec_2<T>& pos) const {
  Vec_2<double> offset{};
  Vec_2<double> dR = pos - origin_;
  for (int dim = 0; dim < 2; dim++) {
    if (dR[dim] < 0) {
      offset[dim] = gl_l_[dim];
    } else if (dR[dim] >= l_[dim]) {
      offset[dim] = -gl_l_[dim];
    }
  }
  return offset;
}

/**
 * @brief cell list constituted by nodes with head and tail pointers.
 *
 * The particla class is wrapped by the BiNode class so that one can access the
 * the paricles by the linked list they constitute. For each cell, one head
 * pointer for a linked list is assigned.
 *
 * Provide three ways to visit each pair of particles. Call one of them to calculate
 * pair force among particles.
 *
 * When particles move, the celllist need be upgraded. If the displacement of a
 * particle during one time step is comparable to the length of one cell, it's highly
 * possible that the particle would move into a new cell, thus every step we recreate
 * all cell lists. Otherwise most particles remain to stay in the previous cell, thus
 * only a low fracton of lists where particle had leave or join in are upgraded.
 *
 * @tparam TNode template class for node
 */
template <typename TNode>
class CellListNode_2 : public CellListBase_2 {
public:
  typedef typename std::vector<TNode*>::iterator IT;
  typedef typename std::vector<TNode*>::const_iterator CIT;
  typedef Vec_2<int> Vec2i;

  template <typename TDomain, typename TGrid>
  CellListNode_2(const TDomain& dm, const TGrid& grid) : CellListBase_2(dm, grid), head_(n_cells_) {}

  template <typename BiFunc1, typename BiFunc2>
  void for_each_pair(BiFunc1 f1, BiFunc2 f2, const Vec2i& ic_beg, const Vec2i& ic_end) const;
  template <typename BiFunc1, typename BiFunc2>
  void for_each_pair(BiFunc1 f1, BiFunc2 f2) const { for_each_pair(f1, f2, Vec_2<int>(), real_cells_.end); }

  template <typename BiFunc, typename TriFunc>
  void for_each_pair_fast(BiFunc f1, TriFunc f2, const Vec2i& ic_beg, const Vec2i& ic_end) const;
  template <typename BiFunc, typename TriFunc>
  void for_each_pair_fast(BiFunc f1, TriFunc f2) const { for_each_pair_fast(f1, f2, Vec_2<int>(), real_cells_.end); }

  template <typename BiFunc1, typename BiFunc2>
  void for_each_pair_slow(BiFunc1 f1, BiFunc2 f2, const Vec2i& ic_beg, const Vec2i& ic_end) const;
  template <typename BiFunc1, typename BiFunc2>
  void for_each_pair_slow(BiFunc1 f1, BiFunc2 f2) const;

  template <typename UniFunc>
  void for_each_cell(UniFunc f, const Vec2i& beg, const Vec2i& end);

  void create(std::vector<TNode>& p_arr);
  void recreate(std::vector<TNode>& p_arr);
  void update(TNode& p, int ic_old, int ic_new);

  void add_node(TNode& p) { p.append_at_front(&head_[get_ic(p)]); }

  void clear(const Vec2i& first, const Vec2i& last);

  void replace(TNode& p1, const TNode& p2);

  void make_compact(std::vector<TNode>& p_arr, std::vector<int>& vacancy);

  int get_par_num(const Vec2i& beg, const Vec2i& end) const;
  int get_par_num() const { return get_par_num(Vec2i(), n_); }

  void reserve_particles(std::vector<TNode>& p_arr, int new_size, double magnification = 1.1);

protected:
  std::vector<TNode*> head_;
};

template <typename TNode>
template <typename BiFunc1, typename BiFunc2>
void CellListNode_2<TNode>::for_each_pair_slow(BiFunc1 f1, BiFunc2 f2,
  const Vec2i& ic_beg,
  const Vec2i& ic_end) const {
  for (int y0 = ic_beg.y; y0 < ic_end.y; y0++) {
    for (int x0 = ic_beg.x; x0 < ic_end.x; x0++) {
      int nx_y0 = y0 * n_.x;
      int i0 = x0 + nx_y0;
      if (head_[i0]) {
        for_each_node_pair(head_[i0], f1);
        int x_left = x0 - 1;
        if (x_left < 0) {
          x_left += n_.x;
        }
        int x_right = x0 + 1;
        if (x_right >= n_.x) {
          x_right = 0;
        }
        int y_up = y0 + 1;
        if (y_up >= n_.y) {
          y_up = 0;
        }
        int nx_y_up = y_up * n_.x;
        TNode* h1 = head_[x_right + nx_y0];
        if (h1) {
          for_each_node_pair(head_[i0], h1, f2);
        }
        TNode* h2 = head_[x_left + nx_y_up];
        if (h2) {
          for_each_node_pair(head_[i0], h2, f2);
        }
        TNode* h3 = head_[x0 + nx_y_up];
        if (h3) {
          for_each_node_pair(head_[i0], h3, f2);
        }
        TNode* h4 = head_[x_right + nx_y_up];
        if (h4) {
          for_each_node_pair(head_[i0], h4, f2);
        }
      }
    }
  }
}

template <typename TNode>
template <typename BiFunc1, typename BiFunc2>
void CellListNode_2<TNode>::for_each_pair_slow(BiFunc1 f1, BiFunc2 f2) const {
  Vec_2<int> beg = real_cells_.beg;
  Vec_2<int> end = n_;
  if (flag_padded.x) {
    beg.x = 0;
    end.x = n_.x;
  }
  if (flag_padded.y) {
    beg.y -= 1;
    end.y -= 1;
  }
  for_each_pair_slow(f1, f2, beg, end);
}

template <typename TNode>
template <typename BiFunc1, typename BiFunc2>
void CellListNode_2<TNode>::for_each_pair(BiFunc1 f1, BiFunc2 f2,
  const Vec2i& ic_beg,
  const Vec2i& ic_end) const {
  int y[2];
  int x[2];
  int i[4];

  for (y[0] = ic_beg.y; y[0] < ic_end.y; y[0]++) {
    y[1] = y[0] + 1;
    if (y[1] >= n_.y)
      y[1] -= n_.y;
    const int y_nx[2] = { y[0] * n_.x, y[1] * n_.x };
    for (x[0] = ic_beg.x; x[0] < ic_end.x; x[0]++) {
      x[1] = x[0] + 1;
      if (x[1] >= n_.x)
        x[1] -= n_.x;
      i[0] = x[0] + y_nx[0];
      i[1] = x[1] + y_nx[0];
      i[2] = x[0] + y_nx[1];
      i[3] = x[1] + y_nx[1];

      if (head_[i[0]]) {
        for_each_node_pair(head_[i[0]], f1);
        if (head_[i[1]]) {
          for_each_node_pair(head_[i[0]], head_[i[1]], f2);
        }
        if (head_[i[2]]) {
          for_each_node_pair(head_[i[0]], head_[i[2]], f2);
        }
        if (head_[i[3]]) {
          for_each_node_pair(head_[i[0]], head_[i[3]], f2);
        }
      }
      if (head_[i[1]] && head_[i[2]]) {
        for_each_node_pair(head_[i[1]], head_[i[2]], f2);
      }
    }
  }
}

template <typename TNode>
template <typename BiFunc, typename TriFunc>
void CellListNode_2<TNode>::for_each_pair_fast(BiFunc f1, TriFunc f2,
  const Vec2i& ic_beg,
  const Vec2i& ic_end) const {
  int y[2];
  int x[2];
  int i[4];

  for (y[0] = ic_beg.y; y[0] < ic_end.y; y[0]++) {
    double ly = 0;
    y[1] = y[0] + 1;
    if (y[1] >= n_.y) {
      y[1] -= n_.y;
      ly = gl_l_.y;
    }
    const int y_nx[2] = { y[0] * n_.x, y[1] * n_.x };
    for (x[0] = ic_beg.x; x[0] < ic_end.x; x[0]++) {
      double lx = 0;
      x[1] = x[0] + 1;
      if (x[1] >= n_.x) {
        x[1] -= n_.x;
        lx = gl_l_.x;
      }
      i[0] = x[0] + y_nx[0];
      i[1] = x[1] + y_nx[0];
      i[2] = x[0] + y_nx[1];
      i[3] = x[1] + y_nx[1];

      if (head_[i[0]]) {
        for_each_node_pair(head_[i[0]], f1);
        if (head_[i[1]]) {
          for_each_node_pair(head_[i[0]], head_[i[1]], Vec2d(lx, 0.), f2);
        }
        if (head_[i[2]]) {
          for_each_node_pair(head_[i[0]], head_[i[2]], Vec2d(0., ly), f2);
        }
        if (head_[i[3]]) {
          for_each_node_pair(head_[i[0]], head_[i[3]], Vec2d(lx, ly), f2);
        }
      }
      if (head_[i[1]] && head_[i[2]]) {
        for_each_node_pair(head_[i[1]], head_[i[2]], Vec2d(-lx, ly), f2);
      }
    }
  }
}

template<typename TNode>
void CellListNode_2<TNode>::create(std::vector<TNode>& p_arr) {
  auto end = p_arr.end();
  for (auto it = p_arr.begin(); it != end; ++it) {
    add_node(*it);
  }
}

template <typename TNode>
void CellListNode_2<TNode>::recreate(std::vector<TNode>& p_arr) {
  for (int ic = 0; ic < n_cells_; ic++) {
    head_[ic] = nullptr;
  }
  create(p_arr);
}

template<typename TNode>
void CellListNode_2<TNode>::update(TNode& p, int ic_old, int ic_new) {
  if (&p == head_[ic_old]) {
    p.break_away(&head_[ic_old]);
  } else {
    p.break_away();
  }
  p.append_at_front(&head_[ic_new]);
}


template <typename TNode>
template <typename UniFunc>
void CellListNode_2<TNode>::for_each_cell(UniFunc f, const Vec2i& beg, const Vec2i& end) {
  for (int y = beg.y; y < end.y; y++) {
    const auto y_nx = y * n_.x;
    for (int x = beg.x; x < end.x; x++) {
      int ic = x + y_nx;
      f(&head_[ic]);
    }
  }
}

template <typename TNode>
void CellListNode_2<TNode>::clear(const Vec2i& first, const Vec2i& last) {
  for (int y0 = first.y; y0 < last.y; y0++) {
    const auto y0_nx = y0 * n_.x;
    for (int x0 = first.x; x0 < last.x; x0++) {
      int ic = x0 + y0_nx;
      head_[ic] = nullptr;
    }
  }
}

template <typename TNode>
void CellListNode_2<TNode>::replace(TNode& p1, const TNode& p2) {
  p1 = p2;
  if (p1.next) {
    p1.next->prev = &p1;
  }
  if (p1.prev) {
    p1.prev->next = &p1;
  } else {
    head_[get_ic(p1)] = &p1;
  }
}


template<typename TNode>
void CellListNode_2<TNode>::make_compact(std::vector<TNode>& p_arr,
  std::vector<int>& vacancy) {
  //! vacancy should be sorted in descending order
  std::sort(vacancy.begin(), vacancy.end(), std::greater<int>());
  int k = 0;
  while (k < vacancy.size()) {
    if (p_arr.size() - 1 == vacancy[k]) {
      p_arr.pop_back();
      k++;
    } else {
      auto i = vacancy.back();
      replace(p_arr[i], p_arr.back());
      p_arr.pop_back();
      vacancy.pop_back();
    }
  }
  vacancy.clear();
}

template <typename TNode>
int CellListNode_2<TNode>::get_par_num(const Vec2i& beg, const Vec2i& end) const {
  int count = 0;
  for (int y0 = beg.y; y0 < end.y; y0++) {
    const auto y0_nx = y0 * n_.x;
    for (int x0 = beg.x; x0 < end.x; x0++) {
      int ic = x0 + y0_nx;
      TNode* cur_node = head_[ic];
      while (cur_node) {
        count++;
        cur_node = cur_node->next;
      }
    }
  }
  return count;
}

template <typename TNode>
void CellListNode_2<TNode>::reserve_particles(std::vector<TNode>& p_arr,
                                              int new_size,
                                              double magnification) {
  size_t new_cap = new_size * magnification;
  std::cout << "old capacity = " << p_arr.capacity() << "\t";
  p_arr.reserve(new_cap);
  recreate(p_arr);
  std::cout << "new capacity = " << new_cap << std::endl;
}
