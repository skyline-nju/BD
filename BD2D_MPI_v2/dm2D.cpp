#include "domain2D.h"

Box_2::Box_2(const Vec_2<double>& gl_l, const Vec_2<int>& proc_rank, const Vec_2<int>& proc_size) {
  set_box(gl_l, proc_rank, proc_size);
}

void Box_2::set_box(const Vec_2<double>& gl_l, const Vec_2<int>& proc_rank, const Vec_2<int>& proc_size) {
  l.x = gl_l.x / proc_size.x;
  l.y = gl_l.y / proc_size.y;
  o.x = l.x * proc_rank.x;
  o.y = l.y * proc_rank.y;
}

Mesh_2::Mesh_2(const Box_2& box, double r_cut, const Vec_2<double>& gl_l, const Vec_2<int>& proc_size)
  : o_(box.o), l_(box.l), gl_l_(gl_l), flag_pad_(proc_size.x > 1, proc_size.y > 1) {
  n_.x = int(l_.x / r_cut);
  n_.y = int(l_.y / r_cut);
  cell_l_.x = l_.x / n_.x;
  cell_l_.y = l_.y / n_.y;
  inv_cell_l_.x = 1. / cell_l_.x;
  inv_cell_l_.y = 1. / cell_l_.y;
  if (flag_pad_.x) {
    n_.x += 2;
    o_.x -= cell_l_.x;
    l_.x += 2 * cell_l_.x;
  }
  if (flag_pad_.y) {
    n_.y += 2;
    o_.y -= cell_l_.y;
    l_.y += 2 * cell_l_.y;
  }
  n_tot_ = n_.x * n_.y;
  set_comm_shell();
}

void Mesh_2::cal_pos_offset(Vec_2<double>& offset, const Vec_2<double>& pos) const {
  Vec_2<double> dR = pos - o_;
  offset.x = offset.y = 0.;
  if (flag_pad_.x) {
    if (dR.x < 0) {
      offset.x = gl_l_.x;
    } else if (dR.x >= gl_l_.x) {
      offset.x = -gl_l_.x;
    }
  }
  if (flag_pad_.y) {
    if (dR.y < 0) {
      offset.y = gl_l_.y;
    } else if (dR.y >= gl_l_.y) {
      offset.y = -gl_l_.y;
    }
  }
}

void Mesh_2::set_comm_shell() {
  Vec_2<int> shell_l{};
  if (flag_pad_.x) {
    if (flag_pad_.y) {
      shell_l = Vec_2<int>(1, n_.y - 2);
      inner_edge_[x_neg].set_value(Vec_2<int>(1, 1), shell_l);
      inner_edge_[x_pos].set_value(Vec_2<int>(n_.x - 2, 1), shell_l);
    } else {
      shell_l = Vec_2<int>(1, n_.y);
      inner_edge_[x_neg].set_value(Vec_2<int>(1, 0), shell_l);
      inner_edge_[x_pos].set_value(Vec_2<int>(n_.x - 2, 0), shell_l);
      
    }
    shell_l = Vec_2<int>(1, n_.y);
    outer_edge_[x_neg].set_value(Vec_2<int>(0, 0), shell_l);
    outer_edge_[x_pos].set_value(Vec_2<int>(n_.x - 1, 0), shell_l);
  }
  if (flag_pad_.y) {
    shell_l = Vec_2<int>(n_.x, 1);
    inner_edge_[y_neg].set_value(Vec_2<int>(0, 1), shell_l);
    inner_edge_[y_pos].set_value(Vec_2<int>(0, n_.y - 2), shell_l);
    if (flag_pad_.x) {
      shell_l = Vec_2<int>(n_.x - 2, 1);
      outer_edge_[y_neg].set_value(Vec_2<int>(1, 0), shell_l);
      outer_edge_[y_pos].set_value(Vec_2<int>(1, n_.y - 1), shell_l);
    } else {
      shell_l = Vec_2<int>(n_.x, 1);
      outer_edge_[y_neg].set_value(Vec_2<int>(0, 0), shell_l);
      outer_edge_[y_pos].set_value(Vec_2<int>(0, n_.y - 1), shell_l);
    }
  }
}

Domain_2::Domain_2(const Vec_2<double>& gl_l, const Vec_2<int>& proc_size_vec, MPI_Comm comm) 
  : gl_l_(gl_l), comm_(comm), proc_size_vec_(proc_size_vec),
    flag_comm_(proc_size_vec.x > 1, proc_size_vec.y > 1) {
  MPI_Comm_size(comm, &tot_proc_);
  if (tot_proc_ != proc_size_vec.x * proc_size_vec.y) {
    std::cout << "Error! tot proc = " << tot_proc_ << " != " 
      << proc_size_vec.x << " x " << proc_size_vec.y << std::endl;
    exit(2);
  }
  MPI_Comm_rank(comm, &my_rank_);
  box_.set_box(gl_l, get_proc_rank_vec(), proc_size_vec_);
  
  find_neighbor_domain(0);
  find_neighbor_domain(1);
}

Domain_2::~Domain_2() {
  for (int i = 0; i < 4; i++) {
    delete[] buf_[i];
  }
}

void Domain_2::find_neighbor_domain(int dir){
  int idx_prev = dir * 2;
  int idx_next = dir * 2 + 1;
  if (flag_comm_[dir]) {
    Vec_2<int> rank = get_proc_rank_vec();
    Vec_2<int> prev = rank;
    Vec_2<int> next = rank;
    prev[dir] = rank[dir] - 1;
    next[dir] = rank[dir] + 1;
    if (prev[dir] < 0) {
      prev[dir] = proc_size_vec_[dir] - 1;
    }
    if (next[dir] >= proc_size_vec_[dir]) {
      next[dir] = 0;
    }
    neighbor_[idx_prev] = prev.x + prev.y * proc_size_vec_.x;
    neighbor_[idx_next] = next.x + next.y * proc_size_vec_.x;
  } else {
    neighbor_[idx_prev] = neighbor_[idx_next] = MPI_PROC_NULL;
  }
}

void Domain_2::set_buf(double r_cut, double amplification, int size_of_one_par) {
  double l_max = 0;
  if (flag_comm_.x) {
    l_max = box_.l.x;
  } 
  if (flag_comm_.y && box_.l.y > l_max) {
    l_max = box_.l.y;
  }
  max_buf_size_ = int((l_max + 2) * r_cut * amplification) * size_of_one_par;
  for (int i = 0; i < 4; i++) {
    if (buf_[i]) {
      delete[] buf_[i];
    }
    buf_[i] = new double[max_buf_size_];
  }
}


