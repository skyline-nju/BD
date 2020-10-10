#pragma once
#include "vect.h"

inline void tangle_1d(double &x, double L) {
  if (x < 0.) {
    x += L;
  } else if (x >= L) {
    x -= L;
  }
}

inline void untangle_1d(double& dx, double half_L, double L) {
  if (dx < -half_L) {
    dx += L;
  } else if (dx > half_L) {
    dx -= L;
  }
}

class PeriodicBdyCondi_2 {
public:
  PeriodicBdyCondi_2(const Vec_2<double>& gl_l,
                     const Vec_2<int>& proc_size = Vec_2<int>(1, 1))
    : l_(gl_l), half_l_(gl_l * 0.5),
      flag_PBC_(proc_size.x == 1, proc_size.y == 1) {}

  void tangle(Vec_2<double>& pos) const {
    if (flag_PBC_.x)
      tangle_1d(pos.x, l_.x);
    if (flag_PBC_.y)
      tangle_1d(pos.y, l_.y);
  }

  void untangle(Vec_2<double>& r12_vec) const {
    if (flag_PBC_.x)
      untangle_1d(r12_vec.x, half_l_.x, l_.x);
    if (flag_PBC_.y)
      untangle_1d(r12_vec.y, half_l_.y, l_.y);
  }

  template <typename TPar>
  void wall_force(TPar& p) const {}

private:
  Vec_2<double> l_;
  Vec_2<double> half_l_;
  Vec_2<bool> flag_PBC_;
};

class PBCxWally {
public:
  PBCxWally(const Vec_2<double>& gl_l, double eps,
    const Vec_2<int>& proc_size = Vec_2<int>(1, 1), double sigma = 1.)
    : l_(gl_l), half_l_(gl_l * 0.5), flag_PBC_(proc_size.x == 1, false),
    eps24_(eps * 24.), sigma_square_(sigma* sigma),
    y_min_(std::pow(2.0, 1. / 6)* sigma * 0.5), y_max_(gl_l.y - y_min_) {}

  void tangle(Vec_2<double>& pos) const {
    if (flag_PBC_.x)
      tangle_1d(pos.x, l_.x);
  }

  void untangle(Vec_2<double>& r12_vec) const {
    if (flag_PBC_.x)
      untangle_1d(r12_vec.x, half_l_.x, l_.x);
  }

  template <typename TPar>
  void wall_force(TPar& p) const;

private:
  Vec_2<double> l_;
  Vec_2<double> half_l_;
  Vec_2<bool> flag_PBC_;
  double eps24_;
  double sigma_square_;
  double y_min_;
  double y_max_;
};

template<typename TPar>
void PBCxWally::wall_force(TPar& p) const {
  if (p.pos.y < y_min_) {
    double dy = p.pos.y + y_min_;
    double r_2 = sigma_square_ / (dy * dy);
    double r_6 = r_2 * r_2 * r_2;
    p.f.y += dy * eps24_ * (2 * r_6 * r_6 - r_6) * r_2;
  } else if (p.pos.y > y_max_) {
    double dy = l_.y - p.pos.y + y_min_;
    double r_2 = sigma_square_ / (dy * dy);
    double r_6 = r_2 * r_2 * r_2;
    p.f.y -= dy * eps24_ * (2 * r_6 * r_6 - r_6) * r_2;
  }
}