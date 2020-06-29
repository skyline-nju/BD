/**
 * @file AmABP.h
 * @author Yu Duan (duanyu.nju@qq.com)
 * @brief Active Brownian particles with Amphiphilic force.
 * @version 0.1
 * @date 2020-06-29
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#include "config.h"
#include "rand.h"
#include "domain2D.h"
#include "particle2D.h"
#include "boundary2D.h"
#include "iodata2D.h"
#include "string.h"
#include "mpi.h"
#include "ABP2D.h"
#include <typeinfo>

 // U(r12_vec, q1, q2) = C * exp(-lambda(r-1.))/r12**2 (q1 - q2) * r21_vec
class AmphiphilicWCA_2 {
public:
  AmphiphilicWCA_2() = default;
  AmphiphilicWCA_2(double eps, double lambda, double c, double r_cut)
    : C_(c), lambda_(lambda), eps24_(eps * 24.), rcut_square_AN_(r_cut* r_cut) {
    double r_cut_WCA = std::pow(2.0, 1. / 6);
    rcut_square_WCA_ = r_cut_WCA * r_cut_WCA;
  }

  void cal_force_torque(double r12_square, const Vec_2<double>& r12_vec, Vec_2<double>& f12_vec,
    const Vec_2<double>& q1, const Vec_2<double>& q2, double& tau1, double& tau2) const;

  template <typename TPar>
  void accum_force_torque(double r12_square, const Vec_2<double>& r12_vec, TPar& p1, TPar& p2) const;

  template<typename TPar>
  void operator ()(TPar& p1, TPar& p2) const;

  template<typename TPar, typename BdyCondi>
  void operator ()(TPar& p1, TPar& p2, const BdyCondi& bc) const;

  std::string get_info() const;
private:
  double C_;
  double lambda_;
  double eps24_;
  double rcut_square_WCA_;
  double rcut_square_AN_;
};



template <typename TPar>
void AmphiphilicWCA_2::accum_force_torque(double r12_square, const Vec_2<double>& r12_vec, TPar& p1, TPar& p2) const {
  Vec_2<double> f12_vec{};
  double tau1 = 0.;
  double tau2 = 0.;
  cal_force_torque(r12_square, r12_vec, f12_vec, p1.u, p2.u, tau1, tau2);
  p1.f -= f12_vec;
  p2.f += f12_vec;
  p1.tau += tau1;
  p2.tau += tau2;
}

template<typename TPar>
void AmphiphilicWCA_2::operator ()(TPar& p1, TPar& p2) const {
  Vec_2<double> r12_vec = p2.pos - p1.pos;
  const double r12_square = r12_vec.square();
  if (r12_square < rcut_square_AN_) {
    accum_force_torque(r12_vec.square(), r12_vec, p1, p2);
  }
}

template<typename TPar, typename BdyCondi>
void AmphiphilicWCA_2::operator ()(TPar& p1, TPar& p2, const BdyCondi& bc) const {
  Vec_2<double> r12_vec = p2.pos - p1.pos;
  bc.untangle(r12_vec);
  const double r12_square = r12_vec.square();
  if (r12_square < rcut_square_AN_) {
    accum_force_torque(r12_vec.square(), r12_vec, p1, p2);
  }
}
