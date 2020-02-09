#include "domain2D.h"

#ifdef USE_MPI
void get_proc_rank_vec(const Vec_2<int>& proc_size, Vec_2<int>& proc_rank, MPI_Comm group_comm) {
  int my_rank;
  MPI_Comm_rank(group_comm, &my_rank);
  proc_rank.x = my_rank % proc_size.x;
  proc_rank.y = my_rank / proc_size.x;
}
#endif

#ifndef USE_MPI
Grid_2::Grid_2(const Vec_2<double>& gl_l, double r_cut) {
  gl_n_.x = int(gl_l.x / r_cut);
  gl_n_.y = int(gl_l.y / r_cut);
  lc_.x = gl_l.x / gl_n_.x;
  lc_.y = gl_l.y / gl_n_.y;
  n_ = gl_n_;
  origin_.x = origin_.y = 0;
  inverse_lc_.x = 1. / lc_.x;
  inverse_lc_.y = 1. / lc_.y;
}
#else
Grid_2::Grid_2(const Vec_2<double>& gl_l, double r_cut,
               const Vec_2<int>& proc_size, MPI_Comm group_comm) {
  //! The program would break down if the number of cells in each subdomain is unequal
  //! for unkonw reason.
  //gl_n_.x = int(gl_l.x / r_cut);
  //gl_n_.y = int(gl_l.y / r_cut);
  //lc_.x = gl_l.x / gl_n_.x;
  //lc_.y = gl_l.y / gl_n_.y;
  //inverse_lc_.x = 1. / lc_.x;
  //inverse_lc_.y = 1. / lc_.y;
  //Vec_2<int> n_per_proc(gl_n_.x / proc_size.x, gl_n_.y / proc_size.y);
  //Vec_2<int> proc_rank;
  //get_proc_rank_vec(proc_size, proc_rank, group_comm);
  //origin_ = n_per_proc * proc_rank;
  ////MPI_Barrier(group_comm);

  //if (proc_rank.x == proc_size.x - 1) {
  //  n_.x = gl_n_.x - proc_rank.x * n_per_proc.x;
  //} else {
  //  n_.x = n_per_proc.x;
  //}
  //if (proc_rank.y == proc_size.y - 1) {
  //  n_.y = gl_n_.y - proc_rank.y * n_per_proc.y;
  //} else {
  //  n_.y = n_per_proc.y;
  //}
  
  double lx = gl_l.x / proc_size.x;
  double ly = gl_l.y / proc_size.y;
  n_.x = int(lx / r_cut);
  n_.y = int(ly / r_cut);
  gl_n_.x = n_.x * proc_size.x;
  gl_n_.y = n_.y * proc_size.y;
  lc_.x = lx / n_.x;
  lc_.y = ly / n_.y;
  inverse_lc_.x = 1. / lc_.x;
  inverse_lc_.y = 1. / lc_.y;
  Vec_2<int> proc_rank;
  get_proc_rank_vec(proc_size, proc_rank, group_comm);
  origin_.x = proc_rank.x * n_.x;
  origin_.y = proc_rank.y * n_.y;
  std::cout << "(" << n_.x << ", " << n_.y << ") grids for proc " << proc_rank << " with lc = (" << lc_.x << ", " << lc_.y << ")" << std::endl;
}
#endif

#ifdef USE_MPI
Domain_2::Domain_2(const Vec_2<double>& gl_l, const Grid_2& grid,
                   const Vec_2<int>& proc_size, MPI_Comm group_comm)
  : comm_(group_comm), proc_size_(proc_size) {
  get_proc_rank_vec(proc_size, proc_rank_, group_comm);
  gl_l_ = gl_l;
  l_.x = grid.n().x * grid.lc().x;
  l_.y = grid.n().y * grid.lc().y;
  o_.x = grid.origin().x * grid.lc().x;
  o_.y = grid.origin().y * grid.lc().y;
}

void Domain_2::find_neighbor(int (*neighbor)[2]) const {
  const int nx = proc_size_.x;
  for (int dim = 0; dim < 2; dim++) {
    if (proc_size_[dim] > 1) {
      Vec_2<int> prev(proc_rank_);
      Vec_2<int> next(proc_rank_);
      prev[dim] -= 1;
      next[dim] += 1;
      if (prev[dim] < 0)
        prev[dim] += proc_size_[dim];
      if (next[dim] >= proc_size_[dim])
        next[dim] -= proc_size_[dim];
      neighbor[dim][0] = prev.x + prev.y * nx;
      neighbor[dim][1] = next.x + next.y * nx;
    } else {
      neighbor[dim][0] = neighbor[dim][1] = MPI_PROC_NULL;
    }
  }
}
#endif

#ifndef USE_MPI
PeriodicDomain_2::PeriodicDomain_2(const Vec_2<double>& gl_l,
                                   const Vec_2<bool> &flag_PBC)
  : Domain_2(gl_l), flag_PBC_(flag_PBC) {
  gl_half_l_.x = gl_l.x * 0.5;
  gl_half_l_.y = gl_l.y * 0.5;
}
#else
PeriodicDomain_2::PeriodicDomain_2(const Vec_2<double>& gl_l,
  const Grid_2& grid,
  const Vec_2<int>& proc_size,
  MPI_Comm group_comm)
  : Domain_2(gl_l, grid, proc_size, group_comm) {
  gl_half_l_.x = gl_l.x * 0.5;
  gl_half_l_.y = gl_l.y * 0.5;
  flag_PBC_.x = proc_size_.x == 1;
  flag_PBC_.y = proc_size_.y == 1;
}
#endif