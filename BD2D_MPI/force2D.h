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

  void eval(const Vec_2<double> &r12_vec, Vec_2<double>& f12_vec) const {
    const double r12_square = r12_vec.square();
    if (r12_square < r_cut_square_) {
      const double r12 = std::sqrt(r12_square);
      const double f = (sigma_ - r12) * spring_const_;
      const double tmp = f / r12;
      f12_vec.x = tmp * r12_vec.x;
      f12_vec.y = tmp * r12_vec.y;
    }
  }

  std::string get_info() const {
    char info[200];
    snprintf(info, 200, "spring force--k:%g", spring_const_);
    return info;
  }
private:
  double spring_const_;
  double sigma_;
  double r_cut_square_;
};

class WCAForce_2 {
public:
  WCAForce_2() = default;
  WCAForce_2(double eps, double sigma = 1) {
    eps24_ = eps * 24;
    double r_cut = std::pow(2.0, 1. / 6) * sigma;
    r_cut_square_ = r_cut * r_cut;
    sigma_square_ = sigma * sigma;
  }

  void set_sigma(double sigma) {
    double r_cut = std::pow(2.0, 1. / 6) * sigma;
    r_cut_square_ = r_cut * r_cut;
    sigma_square_ = sigma * sigma;
  }

  double get_r_cut() { return std::sqrt(r_cut_square_); }

  void eval(const Vec_2<double>& r12_vec, Vec_2<double>& f12_vec) const {
    const double r12_square = r12_vec.square();
    if (r12_square < r_cut_square_) {
      double r_2 = sigma_square_ / r12_square;
      double r_6 = r_2 * r_2 * r_2;
      double tmp = eps24_ * (2 * r_6 * r_6 - r_6) * r_2;
      f12_vec.x = tmp * r12_vec.x;
      f12_vec.y = tmp * r12_vec.y;
    }
  }

  std::string get_info() const {
    char info[200];
    snprintf(info, 200, "WCA--eps:%g", eps24_ / 24);
    return info;
  }

private:
  double eps24_;
  double r_cut_square_;
  double sigma_square_;
};

// U(r12_vec, q1, q2) = C * exp(-lambda(r-1.))/r12**2 (q1 - q2) * r21_vec
class AmphiphilicWCA_2 {
public:
  AmphiphilicWCA_2() = default;
  AmphiphilicWCA_2(double eps, double lambda, double c, double r_cut)
    : C_(c), lambda_(lambda) {
    eps24_ = eps * 24;
    double r_cut_WCA = std::pow(2.0, 1. / 6);
    rcut_square_WCA_ = r_cut_WCA * r_cut_WCA;
    rcut_square_AN_ = r_cut * r_cut;
  }

  void eval(const Vec_2<double>& r12_vec, const Vec_2<double>& q1, const Vec_2<double>& q2,
    Vec_2<double>& f12_vec, double& tau1, double& tau2) {
    const double r12_square = r12_vec.square();
    if (r12_square < rcut_square_AN_) {
      double r = sqrt(r12_square);
      double r_2 = 1. / r12_square;
      double grad_V_pre = (lambda_ * r + 2.) * r_2;
      double V = C_ * exp(-lambda_ * (r - 1.)) * r_2;
      Vec_2<double> Vq1 = V * q1;
      Vec_2<double> Vq2 = V * q2;
      Vec_2<double> Vq1_m_q2 = Vq1 - Vq2;
      f12_vec = -(grad_V_pre * (Vq1_m_q2.dot(r12_vec))) * r12_vec + Vq1_m_q2;
      tau1 = Vq1.cross(r12_vec);
      tau2 = -Vq2.cross(r12_vec);
      if (r12_square < rcut_square_WCA_) {
        double r_6 = r_2 * r_2 * r_2;
        double tmp = eps24_ * (2 * r_6 * r_6 - r_6) * r_2;
        f12_vec.x += tmp * r12_vec.x;
        f12_vec.y += tmp * r12_vec.y;
      }
    }
  }

  std::string get_info() const {
    char info[100];
    snprintf(info, 100, "Amphiphilic--C:%g,lambda:%g,r_cut:%g|WCA--eps:%g",
             C_, lambda_, sqrt(rcut_square_AN_), eps24_ / 24);
    return info;
  }
private:
  double C_;
  double lambda_;
  double eps24_;
  double rcut_square_WCA_;
  double rcut_square_AN_;
};

