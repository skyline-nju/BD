#pragma once
#include "vect.h"
#include <cmath>

class SpringForce_2 {
public:
  SpringForce_2() = default;
  explicit SpringForce_2(double k, double sigma) 
    : spring_const_(k), sigma_(sigma), r_cut_square_(sigma* sigma) {}

  void set_spring_const(double spring_const) { spring_const_ = spring_const; }
  void set_sigma(double sigma) { sigma_ = sigma; r_cut_square_ = sigma * sigma; }

  void cal_force(double r12_square, const Vec_2<double>& r12_vec, Vec_2<double>& f12_vec) const;
  template <typename TPar>
  void accum_force(double r12_square, const Vec_2<double>& r12_vec, TPar& p1, TPar& p2) const;
  template <typename TPar>
  void operator ()(TPar& p1, TPar& p2) const;
  template <typename TPar, typename BdyCondi>
  void operator ()(TPar& p1, TPar& p2, const BdyCondi& bc) const;

  std::string get_info() const;
private:
  double spring_const_;
  double sigma_;
  double r_cut_square_;
};


class WCAForce_2 {
public:
  WCAForce_2() = default;
  WCAForce_2(double eps, double sigma = 1) : eps24_(eps * 24.) { set_sigma(sigma); }

  void set_sigma(double sigma) {
    double r_cut = std::pow(2.0, 1. / 6) * sigma;
    r_cut_square_ = r_cut * r_cut;
    sigma_square_ = sigma * sigma;
  }
  double get_r_cut() { return std::sqrt(r_cut_square_); }

  void cal_force(double r12_square, const Vec_2<double>& r12_vec, Vec_2<double>& f12_vec) const;
  template <typename TPar>
  void accum_force(double r12_square, const Vec_2<double>& r12_vec, TPar& p1, TPar& p2) const;
  template <typename TPar>
  void operator ()(TPar& p1, TPar& p2) const;
  template <typename TPar, typename BdyCondi>
  void operator ()(TPar& p1, TPar& p2, const BdyCondi& bc) const;

  std::string get_info() const;

private:
  double eps24_;
  double r_cut_square_;
  double sigma_square_;
};

inline void SpringForce_2::cal_force(double r12_square, const Vec_2<double>& r12_vec, Vec_2<double>& f12_vec) const {
  const double r12 = std::sqrt(r12_square);
  const double f = (sigma_ - r12) * spring_const_;
  const double tmp = f / r12;
  f12_vec = r12_vec * tmp;
}

template <typename TPar>
void SpringForce_2::accum_force(double r12_square, const Vec_2<double>& r12_vec, TPar& p1, TPar& p2) const {
  Vec_2<double> f12_vec{};
  cal_force(r12_square, r12_vec, f12_vec);
  p1.f -= f12_vec;
  p2.f += f12_vec;
}

template <typename TPar>
void SpringForce_2::operator ()(TPar& p1, TPar& p2) const {
  Vec_2<double> r12 = p2.pos - p1.pos;
  const double r12_square = r12.square();
  if (r12_square < r_cut_square_) {
    accum_force(r12_square, r12, p1, p2);
  }
}

template <typename TPar, typename BdyCondi>
void SpringForce_2::operator ()(TPar& p1, TPar& p2, const BdyCondi& bc) const {
  Vec_2<double> r12 = p2.pos - p1.pos;
  bc.untangle(r12);
  const double r12_square = r12.square();
  if (r12_square < r_cut_square_) {
    accum_force(r12_square, r12, p1, p2);
  }
}

inline std::string SpringForce_2::get_info() const {
  char info[200];
  snprintf(info, 200, "spring force--k:%g", spring_const_);
  return info;
}


inline void WCAForce_2::cal_force(double r12_square, const Vec_2<double>& r12_vec, Vec_2<double>& f12_vec) const {
  double r_2 = sigma_square_ / r12_square;
  double r_6 = r_2 * r_2 * r_2;
  double tmp = eps24_ * (2 * r_6 * r_6 - r_6) * r_2;
  f12_vec = r12_vec * tmp;
}

template <typename TPar>
void WCAForce_2::accum_force(double r12_square, const Vec_2<double>& r12_vec, TPar& p1, TPar& p2) const {
  Vec_2<double> f12_vec{};
  cal_force(r12_square, r12_vec, f12_vec);
  p1.f -= f12_vec;
  p2.f += f12_vec;
}

template <typename TPar>
void WCAForce_2::operator ()(TPar& p1, TPar& p2) const {
  Vec_2<double> r12 = p2.pos - p1.pos;
  const double r12_square = r12.square();
  if (r12_square < r_cut_square_) {
    accum_force(r12_square, r12, p1, p2);
  }
}

template <typename TPar, typename BdyCondi>
void WCAForce_2::operator ()(TPar& p1, TPar& p2, const BdyCondi& bc) const {
  Vec_2<double> r12 = p2.pos - p1.pos;
  bc.untangle(r12);
  const double r12_square = r12.square();
  if (r12_square < r_cut_square_) {
    accum_force(r12_square, r12, p1, p2);
  }
}

inline std::string WCAForce_2::get_info() const {
  char info[200];
  snprintf(info, 200, "WCA--eps:%g", eps24_ / 24);
  return info;
}

