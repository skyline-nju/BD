/**
 * @file biExclusion.h
 * @author Yu Duan (duanyu.nju@qq.com)
 * @brief SPP with two types of exclustion interactions.
 * @version 0.1
 * @date 2020-06-29
 * 
 * Two types of exclusion interactions for SPP: 1) mechanical exclusion,
 * wherein two particles mechanically repel each other when overlapping;
 * 2) scattering exclusion, wherein the directions along which each object
 * tries to move are modulated to avoid overlapping.
 * 
 * Ref: PHYSICAL REVIEW E 99, 012614 (2019)
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#pragma once
#include "rand.h"
#include "domain2D.h"
#include "particle2D.h"
#include "boundary2D.h"
#include "iodata2D.h"
#include "string.h"
#include "mpi.h"
#include "ABP2D.h"
#include <typeinfo>

class InverseRForce_2 {
public:
  InverseRForce_2() = default;
  explicit InverseRForce_2(double alpha, double beta, double sigma)
    : alpha_(alpha), beta_(beta), sigma_(sigma), r_cut_square_(sigma* sigma) {}

  void set_sigma(double sigma) { sigma_ = sigma; r_cut_square_ = sigma * sigma; }

  void cal_force(double r12_square, const Vec_2<double>& r12_vec, Vec_2<double>& f12_vec) const {
    f12_vec = r12_vec / r12_square;
  }

  template<typename TPar>
  void operator ()(TPar& p1, TPar& p2) const;

  template<typename TPar, typename BdyCondi>
  void operator ()(TPar& p1, TPar& p2, const BdyCondi& bc) const;

  template <typename TPar>
  void accum_force_torque(double r12_square, const Vec_2<double>& r12_vec, TPar& p1, TPar& p2) const;
private:
  double alpha_;
  double beta_;
  double sigma_;
  double r_cut_square_;
};

template<typename TPar>
void InverseRForce_2::operator()(TPar& p1, TPar& p2) const {
  Vec_2<double> r12_vec = p2.pos - p1.pos;
  const double r12_square = r12_vec.square();
  if (r12_square < r_cut_square_) {
    accum_force_torque(r12_vec.square(), r12_vec, p1, p2);
  }
}

template<typename TPar, typename BdyCondi>
void InverseRForce_2::operator()(TPar& p1, TPar& p2, const BdyCondi& bc) const {
  Vec_2<double> r12_vec = p2.pos - p1.pos;
  bc.untangle(r12_vec);
  const double r12_square = r12_vec.square();
  if (r12_square < r_cut_square_) {
    accum_force_torque(r12_vec.square(), r12_vec, p1, p2);
  }
}

template <typename TPar>
void InverseRForce_2::accum_force_torque(double r12_square, const Vec_2<double>& r12_vec, TPar& p1, TPar& p2) const {
  Vec_2<double> f12_vec{};
  cal_force(r12_square, r12_vec, f12_vec);
  Vec_2<double> force = f12_vec * beta_;
  p1.f -= force;
  p2.f += force;
  p1.tau += alpha_ * (f12_vec.cross(p1.u));
  p2.tau -= alpha_ * (f12_vec.cross(p2.u));
}
