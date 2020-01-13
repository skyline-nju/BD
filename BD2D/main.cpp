#include <iostream>
#include "domain2D.h"
#include "cellList2D.h"
#include "particle2D.h"
#include "force2D.h"
#include "integrate2D.h"
#include "exporter2D.h"

void test_spring_force() {
  typedef BiNode<BP_u_2> node_t;
  Ranq2 myran(1);
  Vec_2<double> gl_l(100, 100);
  int n_par = 8000;
  double r_cut = 1.;
  Grid_2 grid(gl_l, r_cut);
  Domain_2 dm(gl_l);
  PeriodicDomain_2 pdm(gl_l);
  CellListNode_2<node_t> cl(pdm, grid);
  std::vector<node_t> p_arr;
  double sigma = 1.;
  double h0 = 1e-3;
  BD_EM BD_integrator(h0);
  SpringForce_2 spring(1., 500.);
  // cal force
  auto f1 = [&spring](node_t* p1, node_t* p2) {
    Vec_2<double> r12 = p2->pos - p1->pos;
    Vec_2<double> f12{};
    spring.eval(r12, f12);
    p1->f -= f12;
    p2->f += f12;
  };

  auto f2 = [&spring, &pdm](node_t* p1, node_t* p2) {
    Vec_2<double> r12 = p2->pos - p1->pos;
    pdm.untangle(r12);
    Vec_2<double> f12{};
    spring.eval(r12, f12);
    p1->f -= f12;
    p2->f += f12;
  };

  auto f3 = [&spring](node_t* p1, node_t* p2, const Vec_2<double>& offset) {
    Vec_2<double> r12 = p2->pos - p1->pos + offset;
    Vec_2<double> f12{};
    spring.eval(r12, f12);
    p1->f -= f12;
    p2->f += f12;
  };
  if (n_par <= 5000) {
    create_rand_2(p_arr, n_par, myran, pdm);
    cl.create(p_arr);
  } else {
    double sigma_new = 0.5;
    create_rand_2(p_arr, n_par, myran, pdm, sigma_new);
    cl.create(p_arr);
    do {
      spring.set_sigma(sigma_new);
      for (int t = 0; t < 1000; t++) {
        cl.for_each_pair_fast(f1, f3);
        for (int i = 0; i < n_par; i++) {
          BD_integrator.update(p_arr[i], pdm, myran);
        }
        cl.recreate(p_arr);
      }
      sigma_new += 0.1;
    } while (sigma_new < sigma);
    spring.set_sigma(sigma);
  }

  double Pe = 100;
  char fname[100];
  int n_step = 100000;
  snprintf(fname, 100, "test.extxyz");
  exporter::XyzExporter_2 xyz(fname, 0, n_step, 1000, gl_l);
  xyz.dump_pos(0, p_arr);

  for (int t = 1; t <= n_step; t++) {
    cl.for_each_pair_fast(f1, f3);
    for (int i = 0; i < n_par; i++) {
      BD_integrator.update(p_arr[i], Pe, pdm, myran);
    }
    cl.recreate(p_arr);
    xyz.dump_pos(t, p_arr);
    if (t % 1000 == 0) {
      std::cout << t << "\t" << std::endl;
    }
  }
}

void run_ABP(int argc, char** argv) {
#ifdef _MSC_VER
  double Lx = 500;
  double Ly = 500;
  int n_par = 3000;
  double Pe = 0;
  int n_step = 100000;
#else
  double Lx = atoi(argv[1]);
  double Ly = Lx;
  int n_par = atoi(argv[2]);
  double Pe = atof(argv[3]);
  int n_step = atoi(argv[4]);
#endif
  double sigma = 1.;
  double h0 = 5e-5;
  BD_EM BD_integrator(h0);
  WCAForce_2 f_wca(1., sigma);
  typedef BiNode<BP_u_2> node_t;
  Ranq2 myran(1);
  Vec_2<double> gl_l(Lx, Ly);
  double r_cut = f_wca.get_r_cut();
  Grid_2 grid(gl_l, r_cut);
  Domain_2 dm(gl_l);
  PeriodicDomain_2 pdm(gl_l);
  CellListNode_2<node_t> cl(pdm, grid);
  std::vector<node_t> p_arr;
  // cal force
  auto f1 = [&f_wca](node_t* p1, node_t* p2) {
    Vec_2<double> r12 = p2->pos - p1->pos;
    Vec_2<double> f12{};
    f_wca.eval(r12, f12);
    p1->f -= f12;
    p2->f += f12;
  };

  auto f2 = [&f_wca, &pdm](node_t* p1, node_t* p2) {
    Vec_2<double> r12 = p2->pos - p1->pos;
    pdm.untangle(r12);
    Vec_2<double> f12{};
    f_wca.eval(r12, f12);
    p1->f -= f12;
    p2->f += f12;
  };

  auto f3 = [&f_wca](node_t* p1, node_t* p2, const Vec_2<double>& offset) {
    Vec_2<double> r12 = p2->pos - p1->pos + offset;
    Vec_2<double> f12{};
    f_wca.eval(r12, f12);
    p1->f -= f12;
    p2->f += f12;
  };
  if (n_par <= Lx * Ly / 2) {
    create_rand_2(p_arr, n_par, myran, pdm);
    cl.create(p_arr);
  } else {
    double sigma_new = 0.5;
    create_rand_2(p_arr, n_par, myran, pdm, sigma_new);
    cl.create(p_arr);
    do {
      f_wca.set_sigma(sigma_new);
      for (int t = 0; t < 1000; t++) {
        cl.for_each_pair_fast(f1, f3);
        for (int i = 0; i < n_par; i++) {
          BD_integrator.update(p_arr[i], pdm, myran);
        }
        cl.recreate(p_arr);
      }
      sigma_new += 0.1;
    } while (sigma_new < sigma);
    f_wca.set_sigma(sigma);
  }

  char xyz_file[100];
  char log_file[100];
  snprintf(xyz_file, 100, "Lx%g_Ly%g_N%d_v%g.extxyz", Lx, Ly, n_par, Pe);
  snprintf(log_file, 100, "Lx%g_Ly%g_N%d_v%g.log", Lx, Ly, n_par, Pe);

  exporter::XyzExporter_2 xyz(xyz_file, 0, n_step, 5000, gl_l);
  exporter::LogExporter log(log_file, 0, n_step, 10000, n_par);
  xyz.dump_pos(0, p_arr);

  //for (int i = 0; i < n_par; i++) {
  //  p_arr[i].ic = cl.get_ic(p_arr[i]);
  //}
  for (int t = 1; t <= n_step; t++) {
    cl.for_each_pair_slow(f1, f2);
    for (int i = 0; i < n_par; i++) {
      BD_integrator.update(p_arr[i], Pe, pdm, myran);
    }
    cl.recreate(p_arr);

    //for (int i = 0; i < n_par; i++) {
    //  BD_integrator.update(p_arr[i], Pe, pdm, myran);
    //  int ic_new = cl.get_ic(p_arr[i]);
    //  if (ic_new != p_arr[i].ic) {
    //    cl.update(p_arr[i], p_arr[i].ic, ic_new);
    //    p_arr[i].ic = ic_new;
    //  }
    //}
    xyz.dump_pos(t, p_arr);
    if (t % 1000 == 0) {
      std::cout << t << "\t" << std::endl;
    }
    log.record(t);
  }
}

template<typename node_t, typename TRan>
void ini_rand(std::vector<node_t>& p_arr, int n_par, 
              const Vec_2<double>& l, double sigma, TRan& myran) {
  PeriodicDomain_2 pdm(l);
  if (n_par <=l.x * l.y / 2) {
    create_rand_2(p_arr, n_par, myran, pdm);
  } else {
    double r_cut = sigma;
    Grid_2 grid(l, r_cut);
    CellListNode_2<node_t> cl(pdm, grid);
    WCAForce_2 f_pair(1., sigma);

    BD_EM BD_integrator(1e-4);
    auto f1 = [&f_pair](node_t* p1, node_t* p2) {
      Vec_2<double> r12 = p2->pos - p1->pos;
      Vec_2<double> f12{};
      f_pair.eval(r12, f12);
      p1->f -= f12;
      p2->f += f12;
    };

    auto f2 = [&f_pair, &pdm](node_t* p1, node_t* p2) {
      Vec_2<double> r12 = p2->pos - p1->pos;
      pdm.untangle(r12);
      Vec_2<double> f12{};
      f_pair.eval(r12, f12);
      p1->f -= f12;
      p2->f += f12;
    };
    double sigma_new = 0.5;
    create_rand_2(p_arr, n_par, myran, pdm, sigma_new);
    cl.create(p_arr);
    do {
      sigma_new += 0.01;
      f_pair.set_sigma(sigma_new);
      for (int t = 0; t < 100; t++) {
        cl.for_each_pair(f1, f2);
        for (int i = 0; i < n_par; i++) {
          BD_integrator.update(p_arr[i], pdm, myran);
        }
        cl.recreate(p_arr);
      }
    } while (sigma_new < sigma);
  }
  std::cout << "initialized!" << std::endl;
}

int main(int argc, char* argv[]) {
#ifdef _MSC_VER
  double Lx = 10;
  double Ly = 10;
  double phi = 0.5;
  double Pe = 0;
  int n_step = 400000;
  double lambda = 3.;
  double C = 12;
  double epsilon = 10.;
#else
  double Lx = atof(argv[1]);
  double Ly = atof(argv[2]);
  double phi = atof(argv[3]);
  double Pe = atof(argv[4]);
  int n_step = atoi(argv[5]);
  double lambda = 3.;
  double C = atof(argv[6]);
  double epsilon = 1.;
#endif
  int n_par = int(phi * 4. * Lx * Ly / M_PI);
  //int n_par = 2;
  double sigma = 1.;
  double h0 = 1e-4;
  BD_EM BD_integrator(h0);
  typedef BiNode<BP_u_tau_2> node_t;
  Ranq2 myran(1);
  Vec_2<double> gl_l(Lx, Ly);
  double r_cut = 5;
  Grid_2 grid(gl_l, r_cut);
  Domain_2 dm(gl_l);
  PeriodicDomain_2 pdm(gl_l);
  CellListNode_2<node_t> cl(pdm, grid);
  std::vector<node_t> p_arr;
  

  ini_rand(p_arr, n_par, gl_l, sigma, myran);
  cl.create(p_arr);

  AmphiphilicWCA_2 f_Am(epsilon, lambda, C, r_cut);

  // cal pair force
  auto f1 = [&f_Am](node_t* p1, node_t* p2) {
    Vec_2<double> r12 = p2->pos - p1->pos;
    Vec_2<double> f12{};
    double tau1 = 0.;
    double tau2 = 0.;
    f_Am.eval(r12, p1->u, p2->u, f12, tau1, tau2);
    p1->f -= f12;
    p2->f += f12;
    p1->tau += tau1;
    p2->tau += tau2;
  };

  auto f2 = [&f_Am, &pdm](node_t* p1, node_t* p2) {
    Vec_2<double> r12 = p2->pos - p1->pos;
    pdm.untangle(r12);
    Vec_2<double> f12{};
    double tau1 = 0.;
    double tau2 = 0.;
    f_Am.eval(r12, p1->u, p2->u, f12, tau1, tau2);
    p1->f -= f12;
    p2->f += f12;
    p1->tau += tau1;
    p2->tau += tau2;
  };

  auto f3 = [&f_Am](node_t* p1, node_t* p2, const Vec_2<double>& offset) {
    Vec_2<double> r12 = p2->pos - p1->pos + offset;
    Vec_2<double> f12{};
    double tau1 = 0.;
    double tau2 = 0.;
    f_Am.eval(r12, p1->u, p2->u, f12, tau1, tau2);
    p1->f -= f12;
    p2->f += f12;
    p1->tau += tau1;
    p2->tau += tau2;
  };
  
  char xyz_file[100];
  char log_file[100];
  snprintf(xyz_file, 100, "Lx%g_Ly%g_p%g_v%g_C%g.extxyz", Lx, Ly, phi, Pe, C);
  snprintf(log_file, 100, "Lx%g_Ly%g_p%g_v%g_C%g.log", Lx, Ly, phi, Pe, C);

  exporter::XyzExporter_2 xyz(xyz_file, 0, n_step, 1000, gl_l);
  exporter::LogExporter log(log_file, 0, n_step, 5000, n_par);
  xyz.dump_doub_pos(0, p_arr); 

#ifdef HAS_CELL_INDEX
  for (int i = 0; i < n_par; i++) {
    p_arr[i].ic = cl.get_ic(p_arr[i]);
  }
#endif
  for (int t = 1; t <= n_step; t++) {
    cl.for_each_pair_slow(f1, f2);
#ifndef HAS_CELL_INDEX
    for (int i = 0; i < n_par; i++) {
      BD_integrator.update(p_arr[i], Pe, pdm, myran);
    }
    cl.recreate(p_arr);
#else
    for (int i = 0; i < n_par; i++) {
      BD_integrator.update(p_arr[i], Pe, pdm, myran);
      int ic_new = cl.get_ic(p_arr[i]);
      if (ic_new != p_arr[i].ic) {
        cl.update(p_arr[i], p_arr[i].ic, ic_new);
        p_arr[i].ic = ic_new;
      }
    }
#endif
    xyz.dump_doub_pos(t, p_arr);
    if (t % 1000 == 0) {
      std::cout << t << "\t" << std::endl;
    }
    log.record(t);
  }

}