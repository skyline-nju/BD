#include "domain2D.h"

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
  box_.l = Vec_2<double>(gl_l.x / proc_size_vec_.x, gl_l.y / proc_size_vec_.y);
  box_.o = box_.l * get_proc_rank_vec();
  
  find_neighbor();
  //! find_neighbor_domain(0);  //! do not use this function, which would give
  //! find_neighbor_domain(1);  //! wrongresults for some compiler.

  for (int i = 0; i < tot_proc_; i++) {
    if (i == my_rank_) {
      if (i == 0) {
        std::cout << "The global doamin with size (" << gl_l.x << ", "
          << gl_l.y << ") is partitioned into (" << proc_size_vec.x << ", "
          << proc_size_vec.y << ") subdomains.\n";
      }
      std::cout << "rank = " << my_rank_ << "\n";
      std::cout << "  domain: " << box_ << "\n";
      std::cout << "  neighbors: " << "(" << neighbor_[0] << ","
        << neighbor_[1] << "," << neighbor_[2] << "," << neighbor_[3] << ")" << std::endl;
      if (i == tot_proc_ - 1) {
        std::cout << "Succeed to construct domain!\n";
        std::cout << "-------------------------------------------------------------------" 
          << "\n" << std::endl;
      }
    }
    MPI_Barrier(comm_);  
  }
}

Domain_2::~Domain_2() {
  for (int i = 0; i < 4; i++) {
    delete[] buf_[i];
  }
}


//! Caution: this function may give wrong results for some compiler with -O2 option.
//! Information of the complier that run uncorrectely:
//! mpicxx for MPICH version 3.1.3
//! icpc version 15.0.1 (gcc version 4.8.5 compatibility)
//! It runs correctly with -O0 option.
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

void Domain_2::find_neighbor() {
  Vec_2<int> rank = get_proc_rank_vec();
  if (flag_comm_.x) {
    int x_prev = rank.x - 1;
    int x_next = rank.x + 1;
    if (x_prev < 0) {
      x_prev += proc_size_vec_.x;
    }
    if (x_next == proc_size_vec_.x) {
      x_next = 0;
    }
    neighbor_[0] = x_prev + rank.y * proc_size_vec_.x;
    neighbor_[1] = x_next + rank.y * proc_size_vec_.x;
  } else {
    neighbor_[0] = neighbor_[1] = MPI_PROC_NULL;
  }

  if (flag_comm_.y) {
    int y_prev = rank.y - 1;
    int y_next = rank.y + 1;
    if (y_prev < 0) {
      y_prev += proc_size_vec_.y;
    } 
    if (y_next == proc_size_vec_.y) {
      y_next = 0;
    }
    neighbor_[2] = rank.x + y_prev * proc_size_vec_.x;
    neighbor_[3] = rank.x + y_next * proc_size_vec_.x;
#ifdef WALL_Y
    if (rank.y == 0) {
      neighbor_[2] = MPI_PROC_NULL;
    } else if (rank.y == proc_size_vec_.y - 1) {
      neighbor_[3] = MPI_PROC_NULL;
    }
#endif
  } else {
    neighbor_[2] = neighbor_[3] = MPI_PROC_NULL;
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
      buf_[i] = nullptr;
    }
    buf_[i] = new double[max_buf_size_];
  }
}
