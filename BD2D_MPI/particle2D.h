#pragma once
#include "config.h"
#include "vect.h"
#define M_PI 3.14159265358979323846

template <typename TPar, typename TDomain>
bool check_overlap_2(const TPar& p_new, const std::vector<TPar>& p_arr,
                     double sigma_square, const TDomain& dm) {
  bool flag_overlap = false;
  auto end = p_arr.cend();
  for (auto it = p_arr.cbegin(); it != end; ++it) {
    Vec_2<double> r12(p_new.pos - (*it).pos);
    dm.untangle(r12);
    if (r12.square() < sigma_square) {
      flag_overlap = true;
      break;
    }
  }
  return flag_overlap;
}

template <typename TPar, typename TDomain, typename TRan>
void create_rand_2(std::vector<TPar>& p_arr, int n_gl, TRan& myran,
                   const TDomain& dm, double sigma = 1.) {
  const double sigma_square = sigma * sigma;
  int n, n_max;
  Vec_2<double> l = dm.l();
  Vec_2<double> origin = dm.origin();
#ifndef USE_MPI
  n = n_gl;
  n_max = n_gl;
#else
  n_max = static_cast<int>(1.4 * dm.l().x * dm.l().y);
  int my_rank = dm.proc_rank().x + dm.proc_rank().y * dm.proc_size().x;
  int tot_proc = dm.proc_size().x * dm.proc_size().y;
  if (my_rank < tot_proc - 1) {
    n = n_gl / tot_proc;
  } else {
    n = n_gl - n_gl / tot_proc * (tot_proc - 1);
  }
  if (dm.proc_size().x > 1) {
    l.x -= sigma * 0.5;
    origin.x += sigma * 0.5;
  }
  if (dm.proc_size().y > 1) {
    l.y -= sigma * 0.5;
    origin.y += sigma * 0.5;
  }
#endif
  p_arr.reserve(n_max);
  for (int i = 0; i < n; i++) {
    int count = 0;
    int max_count = 1000;
    while (count < max_count) {
      TPar p_new(myran, l, origin);
      bool flag_overlap = check_overlap_2(p_new, p_arr, sigma_square, dm);
      if (!flag_overlap) {
        p_arr.push_back(p_new);
        break;
      }
      count++;
    }
    if (count >= max_count) {
      std::cout << "count = " << count << std::endl;
      exit(1);
    }
  }
  
  if (my_rank == 0) {
    std::cout << "create " << n_gl << " partiles" << std::endl;
  }
  //for (int i = 0; i < n - 1; i++) {
  //  for (int j = i + 1; j < n; j++) {
  //    Vec_2<double> r12_vec = p_arr[j].pos - p_arr[i].pos;
  //    dm.untangle(r12_vec);
  //    if (r12_vec.square() < sigma_square) {
  //      std::cout << r12_vec << std::endl;
  //      exit(1);
  //    }
  //  }
  //}
}

class BP_2 {
public:
  BP_2() : pos(), f() {}
  BP_2(const Vec_2<double>& pos0) : pos(pos0), f() {}
  BP_2(const Vec_2<double>& pos0, const Vec_2<double>& f0)
    : pos(pos0), f(f0) {}
  template <typename TRan>
  BP_2(TRan& myran, const Vec_2<double>& l, const Vec_2<double>& o)
    : pos(o.x + myran.doub() * l.x, o.y + myran.doub() * l.y), f() {}

  template <typename TInt>
  void copy_pos_to(double* dest, TInt& idx) const;

  template <typename TInt>
  void copy_to(double* dest, TInt& idx) const { copy_pos_to(dest, idx); }

  template <typename TInt>
  void copy_pos_from(const double* source, TInt& idx);

  template <typename TInt>
  void copy_from(const double* source, TInt& idx) { copy_pos_from(source, idx); }

  Vec_2<double> pos;
  Vec_2<double> f;
};

template <typename TInt>
void BP_2::copy_pos_to(double* dest, TInt& idx) const {
  dest[idx] = pos.x;
  dest[idx + 1] = pos.y;
  idx += 2;
}

template <typename TInt>
void BP_2::copy_pos_from(const double* source, TInt& idx) {
  pos.x = source[idx];
  pos.y = source[idx + 1];
  idx += 2;
}

class BP_u_2: public BP_2 {
public:
  BP_u_2() :BP_2(), u() {}
  BP_u_2(const Vec_2<double>& pos0, const Vec_2<double>& u0) : BP_2(pos0), u(u0) {}
  BP_u_2(const Vec_2<double>& pos0, const Vec_2<double>& u0, const Vec_2<double>& f0) 
    : BP_2(pos0, f0), u(u0) {}
  template <typename TRan>
  BP_u_2(TRan& myran, const Vec_2<double>& l, const Vec_2<double>& o) : BP_2(myran, l, o) {
    double theta = myran.doub() * M_PI * 2.;
    u.x = cos(theta);
    u.y = sin(theta);
  }

  template <typename TInt>
  void copy_to(double* dest, TInt& idx) const;

  template <typename TInt>
  void copy_from(const double* source, TInt& idx);

  const Vec_2<double>& update_ori(double d_theta) { 
    u.rotate(d_theta); return u; }

  double get_ori() const { return atan2(u.y, u.x); }

  Vec_2<double> u;
#ifdef HAS_CELL_INDEX
  int ic = 0;
#endif
};

template <typename TInt>
void BP_u_2::copy_to(double* dest, TInt& idx) const {
  dest[idx] = pos.x;
  dest[idx + 1] = pos.y;
  dest[idx + 2] = u.x;
  dest[idx + 3] = u.y;
  idx += 4;
}

template <typename TInt>
void BP_u_2::copy_from(const double* source, TInt& idx) {
  pos.x = source[idx];
  pos.y = source[idx + 1];
  u.x = source[idx + 2];
  u.y = source[idx + 3];
  idx += 4;
}

class BP_u_tau_2 : public BP_u_2 {
public:
  BP_u_tau_2() :BP_u_2(), tau(0.) {}
  BP_u_tau_2(const Vec_2<double>& pos0, const Vec_2<double>& u0) 
    : BP_u_2(pos0, u0), tau(0.)  {}
  BP_u_tau_2(const Vec_2<double>& pos0, const Vec_2<double>& u0, const Vec_2<double>& f0)
    : BP_u_2(pos0, u0, f0), tau(0.) {}
  template <typename TRan>
  BP_u_tau_2(TRan& myran, const Vec_2<double>& l, const Vec_2<double>& o) 
    : BP_u_2(myran, l, o), tau(0.) {}
  double tau;
};

class BP_theta_2 : public BP_2 {
public:
  BP_theta_2() : BP_2(), theta(0.) {}
  BP_theta_2(const Vec_2<double>& pos0, double theta0) : BP_2(pos0), theta(theta0) {}
  BP_theta_2(const Vec_2<double>& pos0, double theta0, const Vec_2<double>& f0)
    : BP_2(pos0, f0), theta(theta0) {}
  template <typename TRan>
  BP_theta_2(TRan& myran, const Vec_2<double>& l, const Vec_2<double>& o)
    : BP_2(myran, l, o), theta(myran.doub()* M_PI * 2.) {}

  const Vec_2<double> update_ori(double d_theta) {
    theta += d_theta;
    const double c = std::cos(theta);
    const double s = std::sin(theta);
    return Vec_2<double>(c, s);
  }

  template <typename TInt>
  void copy_to(double* dest, TInt& idx) const;

  template <typename TInt>
  void copy_from(const double* source, TInt& idx);

  double get_ori() const { return theta; }

  double theta;
};

template <typename TInt>
void BP_theta_2::copy_to(double* dest, TInt& idx) const {
  dest[idx] = pos.x;
  dest[idx + 1] = pos.y;
  dest[idx + 2] = theta;
  idx += 3;
}

template <typename TInt>
void BP_theta_2::copy_from(const double* source, TInt& idx) {
  pos.x = source[idx];
  pos.y = source[idx + 1];
  theta = source[idx + 2];
  idx += 3;
}

class BP_theta_tau_2 : public BP_theta_2 {
public:
  BP_theta_tau_2() : BP_theta_2(), tau(0.) {}
  BP_theta_tau_2(const Vec_2<double>& pos0, double theta0) : BP_theta_2(pos0, theta0), tau(0.) {}
  BP_theta_tau_2(const Vec_2<double>& pos0, double theta0, const Vec_2<double>& f0)
    : BP_theta_2(pos0, theta0, f0), tau(0.) {}

  template <typename TRan>
  BP_theta_tau_2(TRan& myran, const Vec_2<double>& l, const Vec_2<double>& o)
    : BP_theta_2(myran, l, o), tau(0.) {}

  double tau;
};

