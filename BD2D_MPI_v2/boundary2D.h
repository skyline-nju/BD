#pragma once
#include "vect.h"

class PeriodicBdyCondi_2 {
public:
  PeriodicBdyCondi_2(const Vec_2<double>& gl_l,
                     const Vec_2<int>& proc_size = Vec_2<int>(1, 1))
    : l_(gl_l), half_l_(gl_l * 0.5),
      flag_PBC_(proc_size.x == 1, proc_size.y == 1) {}

  void tangle(Vec_2<double>& pos) const;

  void untangle(Vec_2<double>& v) const;
private:
  Vec_2<double> l_;
  Vec_2<double> half_l_;
  Vec_2<bool> flag_PBC_;
};

inline void PeriodicBdyCondi_2::tangle(Vec_2<double>& pos) const {
  if (flag_PBC_.x) {
    if (pos.x < 0.) {
      pos.x += l_.x;
    } else if (pos.x >= l_.x) {
      pos.x -= l_.x;
    }
  }
  if (flag_PBC_.y) {
    if (pos.y < 0.) {
      pos.y += l_.y;
    } else if (pos.y >= l_.y) {
      pos.y -= l_.y;
    }
  }
}

inline void PeriodicBdyCondi_2::untangle(Vec_2<double>& r12_vec) const {
  if (flag_PBC_.x) {
    if (r12_vec.x < -half_l_.x) {
      r12_vec.x += l_.x;
    } else if (r12_vec.x > half_l_.x) {
      r12_vec.x -= l_.x;
    }
  }
  if (flag_PBC_.y) {
    if (r12_vec.y < -half_l_.y) {
      r12_vec.y += l_.y;
    } else if (r12_vec.y > half_l_.y) {
      r12_vec.y -= l_.y;
    }
  }
}