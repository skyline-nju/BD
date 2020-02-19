#include "cellList2D.h"

Mesh_2::Mesh_2(const Box_2<double>& box, double r_cut,
              const Vec_2<double>& gl_l, const Vec_2<int>& proc_size)
  : real_box_(box), box_(box), gl_l_(gl_l), flag_pad_(proc_size.x > 1, proc_size.y > 1) {
  n_.x = int(box_.l.x / r_cut);
  n_.y = int(box_.l.y / r_cut);
  cell_l_.x = box_.l.x / n_.x;
  cell_l_.y = box_.l.y / n_.y;
  inv_cell_l_.x = 1. / cell_l_.x;
  inv_cell_l_.y = 1. / cell_l_.y;
  if (flag_pad_.x) {
    n_.x += 2;
    box_.o.x -= cell_l_.x;
    box_.l.x += 2 * cell_l_.x;
  }
  if (flag_pad_.y) {
    n_.y += 2;
    box_.o.y -= cell_l_.y;
    box_.l.y += 2 * cell_l_.y;
  }
  n_tot_ = n_.x * n_.y;
  set_comm_shell();

  {
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if (my_rank == 0) {
      show_info();
      std::cout << "\n" << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank == 1) {
      show_info();
      std::cout << "\n" << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

  }

}

void Mesh_2::show_info()const {
  std::cout << "The domain " << real_box_ << " is extended to " << box_ << ", \n"
    << "partitoned into (" << n_.x << ", " << n_.y << ") cells, "
    << "with linear size (" << cell_l_.x << ", " << cell_l_.y << ")\n";
  if (flag_pad_.x) {
    std::cout << "inner edges in x direction: " << inner_edge_[x_neg] << ", " <<
      inner_edge_[x_pos] << "\n";
    std::cout << "outer edges in x direction: " << outer_edge_[x_neg] << ", " <<
      outer_edge_[x_pos] << "\n";
  }
  if (flag_pad_.y) {
    std::cout << "inner edges in y direction: " << inner_edge_[y_neg] << ", " <<
      inner_edge_[y_pos] << "\n";
    std::cout << "outer edges in y direction: " << outer_edge_[y_neg] << ", " <<
      outer_edge_[y_pos] << "\n";
  }
}

Vec_2<int> Mesh_2::get_real_n() const {
  Vec_2<int> m = n_;
  if (flag_pad_.x) {
    m.x -= 2;
  } 
  if (flag_pad_.y) {
    m.y -= 2;
  }
  return m;
}

void Mesh_2::cal_pos_offset(Vec_2<double>& offset, const Vec_2<double>& pos) const {
  Vec_2<double> dR = pos - box_.o;
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
  if (flag_pad_.x) {
    if (flag_pad_.y) {
      inner_edge_[x_neg].o = Vec_2<int>(1, 1);
      inner_edge_[x_pos].o = Vec_2<int>(n_.x - 2, 1);
      inner_edge_[x_neg].l = inner_edge_[x_pos].l = Vec_2<int>(1, n_.y - 2);
    } else {
      inner_edge_[x_neg].o = Vec_2<int>(1, 0);
      inner_edge_[x_pos].o = Vec_2<int>(n_.x - 2, 0);
      inner_edge_[x_neg].l = inner_edge_[x_pos].l = Vec_2<int>(1, n_.y);
    }
    outer_edge_[x_neg].o = Vec_2<int>(0, 0);
    outer_edge_[x_pos].o = Vec_2<int>(n_.x - 1, 0);
    outer_edge_[x_neg].l = outer_edge_[x_pos].l = Vec_2<int>(1, n_.y);
  }
  if (flag_pad_.y) {
    inner_edge_[y_neg].o = Vec_2<int>(0, 1);
    inner_edge_[y_pos].o = Vec_2<int>(0, n_.y - 2);
    inner_edge_[y_neg].l = inner_edge_[y_pos].l = Vec_2<int>(n_.x, 1);
    if (flag_pad_.x) {
      outer_edge_[y_neg].o = Vec_2<int>(1, 0);
      outer_edge_[y_pos].o = Vec_2<int>(1, n_.y - 1);
      outer_edge_[y_neg].l = outer_edge_[y_pos].l = Vec_2<int>(n_.x - 2, 1);
    } else {
      outer_edge_[y_neg].o = Vec_2<int>(0, 0);
      outer_edge_[y_pos].o = Vec_2<int>(0, n_.y - 1);
      outer_edge_[y_neg].l = outer_edge_[y_pos].l = Vec_2<int>(n_.x, 1);
    }
  }
}
