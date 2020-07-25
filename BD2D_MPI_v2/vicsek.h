/**
 * @file vicsek.h
 * @author Yu Duan (duanyu.nju@qq.com)
 * @brief Vicsek model with density-dependent motility and alignment
 * @version 0.1
 * @date 2020-06-29
 * 
 * Simulating self-propelled particles whose motility and alignment are
 * density-dependent.
 * 
 * Ref: PRL 108, 248101 (2012)
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#pragma once
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


class VicsekPar {
public:
  VicsekPar() : pos(), u(), dtheta(0.), n_neighbor(0) {}
  VicsekPar(const Vec_2<double>& pos0) : pos(pos0), u(), dtheta(0.), n_neighbor(0) {}
  VicsekPar(const Vec_2<double>& pos0, const Vec_2<double>& u0) 
    : pos(pos0), u(u0), dtheta(0.), n_neighbor(0) {}
  template <typename TRan>
  VicsekPar(TRan& myran, const Vec_2<double>& l, const Vec_2<double>& o);

  template <typename TInt>
  void copy_pos_to(double* dest, TInt& idx) const;

  template <typename TInt>
  void copy_to(double* dest, TInt& idx) const;

  template <typename TInt>
  void copy_pos_from(const double* source, TInt& idx);

  template <typename TInt>
  void copy_from(const double* source, TInt& idx);

  template <typename TInt>
  void load_from_file(const float* buf, TInt& idx);

  double get_ori() const { return atan2(u.y, u.x); }

  Vec_2<double> pos;
  Vec_2<double> u;
  double dtheta;
  int n_neighbor;
};

template<typename TRan>
VicsekPar::VicsekPar(TRan& myran, const Vec_2<double>& l, const Vec_2<double>& o) {
  pos.x = o.x + myran.doub() * l.x;
  pos.y = o.y + myran.doub() * l.y;
  double theta = myran.doub() * M_PI * 2;
  u.x = cos(theta);
  u.y = sin(theta);
  dtheta = 0.;
  n_neighbor = 0;
}

template <typename TInt>
void VicsekPar::copy_pos_to(double* dest, TInt& idx) const {
  dest[idx] = pos.x;
  dest[idx + 1] = pos.y;
  idx += 2;
}

template <typename TInt>
void VicsekPar::copy_to(double* dest, TInt& idx) const {
  dest[idx] = pos.x;
  dest[idx + 1] = pos.y;
  dest[idx + 2] = u.x;
  dest[idx + 3] = u.y;
  idx += 4;
}

template <typename TInt>
void VicsekPar::copy_pos_from(const double* source, TInt& idx) {
  pos.x = source[idx];
  pos.y = source[idx + 1];
  idx += 2;
}

template <typename TInt>
void VicsekPar::copy_from(const double* source, TInt& idx) {
  pos.x = source[idx];
  pos.y = source[idx + 1];
  u.x = source[idx + 2];
  u.y = source[idx + 3];
  idx += 4;
}

template <typename TInt>
void VicsekPar::load_from_file(const float* buf, TInt& idx) {
  pos.x = buf[idx];
  pos.y = buf[idx + 1];
  double theta = buf[idx + 2];
  u.x = std::cos(theta);
  u.y = std::sin(theta);
  idx += 3;
}

class AligningForce_2 {
public:
  AligningForce_2() = default;
  explicit AligningForce_2(double sigma): r_cut_square_(sigma* sigma) {}

  template <typename TPar>
  void cal_torque(TPar& p1, TPar& p2) const;

  template <typename TPar>
  void operator()(TPar& p1, TPar& p2) const;

  template <typename TPar, typename BdyCondi>
  void operator()(TPar& p1, TPar& p2, const BdyCondi& bc) const;

private:
  double r_cut_square_;
};

template<typename TPar>
void AligningForce_2::cal_torque(TPar& p1, TPar& p2) const {
  double sin_dtheta = p2.u.y * p1.u.x - p2.u.x * p1.u.y;
  p1.dtheta += sin_dtheta;
  p2.dtheta -= sin_dtheta;
}

template <typename TPar>
void AligningForce_2::operator()(TPar& p1, TPar& p2) const {
  Vec_2<double> r12 = p2.pos - p1.pos;
  const double r12_square = r12.square();
  if (r12_square < r_cut_square_) {
    cal_torque(p1, p2);
    p1.n_neighbor++;
    p2.n_neighbor++;
  }
}

template <typename TPar, typename BdyCondi>
void AligningForce_2::operator()(TPar& p1, TPar& p2, const BdyCondi& bc) const {
  Vec_2<double> r12 = p2.pos - p1.pos;
  bc.untangle(r12);
  const double r12_square = r12.square();
  if (r12_square < r_cut_square_) {
    cal_torque(p1, p2);
    p1.n_neighbor++;
    p2.n_neighbor++;
  }
}

class EM_VM {
public:
  EM_VM(double h, double gamma, double eps, double v0)
    : h_(h), gamma_h_(gamma / M_PI * h), Dr_(std::sqrt(24. * eps * h)), v0_(v0) {}

  template <typename TPar, class BdyCondi, class TRan>
  void update(TPar& p, const BdyCondi& bc, TRan& myran, double v) const;

  template <typename TPar, class BdyCondi, class TRan>
  void update(TPar& p, const BdyCondi& bc, TRan& myran) const {
    return update(p, bc, myran, v0_);
  }
protected:
  double h_;
  double gamma_h_;
  double Dr_;  // sqrt(24 * epsilon * h)
  double v0_;
};

template <class TPar, class BdyCondi, class TRan>
void EM_VM::update(TPar& p, const BdyCondi& bc, TRan& myran, double v) const {
  const double dtheta = p.dtheta * gamma_h_ + Dr_ * (myran.doub() - 0.5);
  p.u.rotate(dtheta);
  const double dr = v * h_;
  p.pos.x += dr * p.u.x;
  p.pos.y += dr * p.u.y;
  bc.tangle(p.pos);
  p.dtheta = 0.;
}

class EM_VM_Scheme1: public EM_VM {
public:
  EM_VM_Scheme1(double h, double gamma, double eps, double v0, double v1, double lambda)
    : EM_VM(h, gamma, eps, v0), v1_(v1), lambda_(lambda) {}

  template <typename TPar, class BdyCondi, class TRan>
  void update(TPar& p, const BdyCondi& bc, TRan& myran) const;

protected:
  double v1_;
  double lambda_;
};

template <typename TPar, class BdyCondi, class TRan>
void EM_VM_Scheme1::update(TPar& p, const BdyCondi& bc, TRan& myran) const {
  double v = v0_ * exp(-lambda_ * p.n_neighbor) + v1_;
  //update(p, bc, myran, v);
  const double dtheta = p.dtheta * gamma_h_ + Dr_ * (myran.doub() - 0.5);
  p.u.rotate(dtheta);
  const double dr = v * h_;
  p.pos.x += dr * p.u.x;
  p.pos.y += dr * p.u.y;
  bc.tangle(p.pos);
  p.dtheta = 0.;
}

class EM_VM_Scheme2 : public EM_VM {
public:
  EM_VM_Scheme2(double h, double gamma, double eps, double v0, double xi, double rho_c)
    : EM_VM(h, gamma, eps, v0), inv_area_xi_(1. / (M_PI * xi)), rho_c_over_xi_(rho_c / xi) {}

  template <typename TPar, class BdyCondi, class TRan>
  void update(TPar& p, const BdyCondi& bc, TRan& myran) const;

protected:
  double inv_area_xi_;    // 1 / (PI * sigma^2 * xi)
  double rho_c_over_xi_;  // rho_c / xi
};

template <typename TPar, class BdyCondi, class TRan>
void EM_VM_Scheme2::update(TPar& p, const BdyCondi& bc, TRan& myran) const {
  //double v = v0_ * exp(-lambda_ * p.n_neighbor) + v1_;
  //update(p, bc, myran, v);
  const double v = 0.5 * v0_ * (1. - tanh((p.n_neighbor + 1) * inv_area_xi_ - rho_c_over_xi_));
  const double dtheta = p.dtheta * gamma_h_ + Dr_ * (myran.doub() - 0.5);
  p.u.rotate(dtheta);
  const double dr = v * h_;
  p.pos.x += dr * p.u.x;
  p.pos.y += dr * p.u.y;
  bc.tangle(p.pos);
  p.dtheta = 0.;
}

class EM_VM_Scheme3 : public EM_VM {
public:
  EM_VM_Scheme3(double h, double gamma, double eps, double v0, double xi, double rho_c1, double rho_c2)
    : EM_VM(h, gamma, eps, v0), inv_area_xi_(1. / (M_PI * xi)),
      rho_c1_over_xi_(rho_c1 / xi), rho_c2_over_xi_(rho_c2 / xi) {}

  template <typename TPar, class BdyCondi, class TRan>
  void update(TPar& p, const BdyCondi& bc, TRan& myran) const;

protected:
  double inv_area_xi_;     // 1 / (PI * sigma^2 * xi)
  double rho_c1_over_xi_;  // rho_c1 / xi
  double rho_c2_over_xi_;  // rho_c2 / xi
};

template <typename TPar, class BdyCondi, class TRan>
void EM_VM_Scheme3::update(TPar& p, const BdyCondi& bc, TRan& myran) const {
  //double v = v0_ * exp(-lambda_ * p.n_neighbor) + v1_;
  //update(p, bc, myran, v);
  const double rho_over_xi = (p.n_neighbor + 1) * inv_area_xi_;
  double v1 = 0.1;
  const double v = 0.5 * (v0_ - v1) * (1. - tanh(rho_over_xi - rho_c1_over_xi_)) + v1;
  const double aligning = - tanh(rho_over_xi - rho_c2_over_xi_);
  const double dtheta = p.dtheta * gamma_h_ * aligning + Dr_ * (myran.doub() - 0.5);
  p.u.rotate(dtheta);
  const double dr = v * h_;
  p.pos.x += dr * p.u.x;
  p.pos.y += dr * p.u.y;
  bc.tangle(p.pos);
  p.dtheta = 0.;
}


template <typename TDomain, typename TPar, typename BdyCondi>
void ini_rand_VM(std::vector<BiNode<TPar>>& p_arr, int n_par_gl, TDomain& dm,
                 const BdyCondi& bc, double sigma = 1.) {
  const Box_2<double>& box = dm.get_box();
  int my_rank = dm.get_proc_rank();
  int tot_proc = dm.get_proc_size();
  Vec_2<int> proc_size_vec = dm.get_proc_size_vec();
  MPI_Comm comm = dm.get_comm();
  int n_max = n_par_gl / tot_proc * 4;
  p_arr.reserve(n_max);

  int n_par;
  if (my_rank < tot_proc - 1) {
    n_par = n_par_gl / tot_proc;
  } else {
    n_par = n_par_gl - n_par_gl / tot_proc * (tot_proc - 1);
  }

  int* n_par_arr = new int[tot_proc] {};
  MPI_Gather(&n_par, 1, MPI_INT, n_par_arr, 1, MPI_INT, 0, comm);
  if (my_rank == 0) {
    std::cout << "ini particle num: " << n_par_gl << " = " << n_par_arr[0];
    for (int i = 1; i < tot_proc; i++) {
      std::cout << " + " << n_par_arr[i];
    }
    std::cout << std::endl;
  }
  delete[] n_par_arr;

  Ranq2 myran(1 + my_rank);
  for (int i = 0; i < n_par; i++) {
    p_arr.emplace_back(myran, box.l, box.o);
  }
#ifdef INI_ORDERED
  for (auto& p : p_arr) {
    p.u.x = -1;
    p.u.y = 0;
  }
#endif
  if (my_rank == 0) {
    std::cout << "initialized randomly!\n";
    std::cout << "************************************\n" << std::endl;
  }
  MPI_Barrier(comm);
}