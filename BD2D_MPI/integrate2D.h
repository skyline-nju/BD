#pragma once
#include <cmath>
#include "particle2D.h"

class EM_BD_iso {
public:
  EM_BD_iso(double h) : h_(h), Dt_(std::sqrt(24. * h)) {}

  template <class TPar, class TDomain, class TRan>
  void update(TPar& p, const TDomain& dm, TRan& myran) const;

  void set_h(double h) { h_ = h; }

  void set_Dt(double Dt) { Dt_ = Dt; }
protected:
  double h_;
  double Dt_; // sqrt(24 * h) by default
};

template<class TPar, class TDomain, class TRan>
void EM_BD_iso::update(TPar& p, const TDomain& dm, TRan& myran) const {
  p.pos.x += p.f.x * h_ + (myran.doub() - 0.5) * Dt_;
  p.pos.y += p.f.y * h_ + (myran.doub() - 0.5) * Dt_;
  dm.tangle(p.pos);
  p.f.x = 0.;
  p.f.y = 0.;
}

class EM_ABD_iso: public EM_BD_iso {
public:
  EM_ABD_iso(double h, double Pe) : EM_BD_iso(h), Dr_(std::sqrt(72. * h)), Pe_(Pe) {}

  template <class TPar, class TDomain, class TRan>
  void update(TPar& p, const TDomain& dm, TRan& myran) const;

  void set_Pe(double Pe) { Pe_ = Pe; }
  void set_Dr(double Dr) { Dr_ = Dr; }
protected:
  double Dr_; // sqrt(72 * h) by default
  double Pe_;
};

template<class TPar, class TDomain, class TRan>
void EM_ABD_iso::update(TPar& p, const TDomain& dm, TRan& myran) const {
  const double d_theta = (myran.doub() - 0.5) * Dr_;
  Vec_2<double> u = p.update_ori(d_theta);
  p.pos.x += (p.f.x + u.x * Pe_) * h_ + (myran.doub() - 0.5) * Dt_;
  p.pos.y += (p.f.y + u.y * Pe_) * h_ + (myran.doub() - 0.5) * Dt_;
  dm.tangle(p.pos);
  p.f.x = 0.;
  p.f.y = 0.;
}

class EM_ABD_aniso : public EM_ABD_iso {
public:
  EM_ABD_aniso(double h, double Pe) : EM_ABD_iso(h, Pe) {}

  template <typename TPar, class TDomain, class TRan>
  void update(TPar& p, const TDomain& dm, TRan& myran) const;
};

template<typename TPar, class TDomain, class TRan>
void EM_ABD_aniso::update(TPar& p, const TDomain& dm, TRan& myran) const {
  const double d_theta = p.tau * 3. * h_ + (myran.doub() - 0.5) * Dr_;
  Vec_2<double> u = p.update_ori(d_theta);
  p.pos.x += (p.f.x + u.x * Pe_) * h_ + (myran.doub() - 0.5) * Dt_;
  p.pos.y += (p.f.y + u.y * Pe_) * h_ + (myran.doub() - 0.5) * Dt_;
  dm.tangle(p.pos);
  p.f.x = 0.;
  p.f.y = 0.;
  p.tau = 0.;
}

/*
class BD_EM {
public:
  BD_EM(double h)
    : h_(h), Dt_(std::sqrt(24. * h)), Dr_(std::sqrt(72. * h)) {}

  template <class TPar, class TDomain, class TRan>
  void update(TPar& p, const TDomain& dm, TRan& myran) const;

  template <class TPar, class TDomain, class TRan>
  void update(TPar& p, double Pe, const TDomain& dm, TRan& myran) const;

  template <class TDomain, class TRan>
  void update(BiNode<BP_u_tau_2>& p, double Pe, const TDomain& dm, TRan& myran) const;

protected:
  double h_;
  double Dt_; // sqrt(24 * h) by default
  double Dr_; // sqrt(72 * h) by default
};

template <class TPar, class TDomain, class TRan>
void BD_EM::update(TPar& p, const TDomain& dm, TRan& myran) const{
  p.pos.x += p.f.x * h_ + (myran.doub() - 0.5) * Dt_;
  p.pos.y += p.f.y * h_ + (myran.doub() - 0.5) * Dt_;
  dm.tangle(p.pos);
  p.f.x = 0.;
  p.f.y = 0.;
}

template<class TPar, class TDomain, class TRan>
void BD_EM::update(TPar& p, double Pe, const TDomain& dm, TRan& myran) const{
  const double d_theta = (myran.doub() - 0.5) * Dr_;
  Vec_2<double> u = p.update_ori(d_theta);
  p.pos.x += (p.f.x + u.x * Pe) * h_ + (myran.doub() - 0.5) * Dt_;
  p.pos.y += (p.f.y + u.y * Pe) * h_ + (myran.doub() - 0.5) * Dt_;
  dm.tangle(p.pos);
  p.f.x = 0.;
  p.f.y = 0.;
}

template<class TDomain, class TRan>
void BD_EM::update(BiNode<BP_u_tau_2>& p, double Pe, const TDomain& dm, TRan& myran) const {
  const double d_theta = p.tau * 3. * h_ + (myran.doub() - 0.5) * Dr_;
  Vec_2<double> u = p.update_ori(d_theta);
  p.pos.x += (p.f.x + u.x * Pe) * h_ + (myran.doub() - 0.5) * Dt_;
  p.pos.y += (p.f.y + u.y * Pe) * h_ + (myran.doub() - 0.5) * Dt_;
  dm.tangle(p.pos);
  p.f.x = 0.;
  p.f.y = 0.;
  p.tau = 0.;
}

*/
