#include "run2D.h"
#include "exporter2D.h"

#ifndef USE_MPI
void run_ABP(Vec_2<double>& gl_l, double phi, double Pe, double eps,
             double h0, int n_step, unsigned long long seed) {
#else
void run_ABP(Vec_2<double>& gl_l, double phi, double Pe, double eps,
             double h0, int n_step, unsigned long long seed, MPI_Comm comm) {
#endif
  typedef BiNode<BP_u_2> node_t;
  std::vector<node_t> p_arr;
  int n_par_gl = int(phi * 4. * gl_l.x * gl_l.y / M_PI);
  double sigma = 1.;

#ifndef USE_MPI
  Ranq2 myran(seed);
  ini_rand(p_arr, n_par_gl, gl_l, sigma, myran);
#else
  int tot_proc, my_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &tot_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  Ranq2 myran(seed + my_rank);
  Vec_2<int> proc_size(tot_proc, 1);
  ini_rand(p_arr, n_par_gl, gl_l, sigma, myran, proc_size, comm);
#endif

  EM_ABD_iso integrator(h0, Pe);
  WCAForce_2 f_wca(1., sigma);
  double r_cut = f_wca.get_r_cut();

#ifndef USE_MPI
  Grid_2 grid(gl_l, r_cut);
  PeriodicDomain_2 pdm(gl_l);
#else
  Grid_2 grid(gl_l, r_cut, proc_size, comm);
  PeriodicDomain_2 pdm(gl_l, grid, proc_size, comm);
  Communicator_2 communicator(pdm, grid);
#endif
  CellListNode_2<node_t> cl(pdm, grid);

  // cal pair force
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

  auto pair_force = [f1, f2, &cl]() {
    cl.for_each_pair_slow(f1, f2);
  };

  auto integrate = [&integrator, &myran, &pdm](node_t& p) {
    integrator.update(p, pdm, myran);
  };

  char xy_file[100];
#ifndef USE_MPI
  snprintf(xy_file, 100, "ABP_Lx%g_Ly%g_p%g_v%g.extxzy", gl_l.x, gl_l.y, phi, Pe);
#else
  snprintf(xy_file, 100, "ABP_Lx%g_Ly%g_p%g_v%g_n%d.extxzy", gl_l.x, gl_l.y, phi, Pe, my_rank);
#endif
  XyzExporter_2 xy_outer(xy_file, 0, n_step, 1000, gl_l);
  auto exporter = [&xy_outer](int i_step, const std::vector<node_t>& par_arr) {
    xy_outer.dump_pos(i_step, par_arr);
  };

  exporter(0, p_arr);
  for (int i = 1; i <= n_step; i++) {
#ifdef USE_MPI
    one_step(i, p_arr, cl, integrate, pair_force, exporter, communicator);
#else
    one_step(i, p_arr, cl, integrate, pair_force, exporter);
#endif
  }
}

#ifndef USE_MPI
void run_ABP_Amphiphilic(Vec_2<double>& gl_l, double phi, double Pe,
                         double eps, double lambda, double C,
                         double h0, int n_step, unsigned long long seed) {
#else
void run_ABP_Amphiphilic(Vec_2<double> & gl_l, double phi, double Pe,
                         double eps, double lambda, double C,
                         double h0, int n_step, unsigned long long seed,
                         MPI_Comm comm) {
#endif
  typedef BiNode<BP_u_tau_2> node_t;
  std::vector<node_t> p_arr;
  int n_par_gl = int(phi * 4. * gl_l.x * gl_l.y / M_PI);
  double sigma = 1.;

#ifndef USE_MPI
  Ranq2 myran(seed);
  ini_rand(p_arr, n_par_gl, gl_l, sigma, myran);
#else
  int tot_proc, my_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &tot_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  Ranq2 myran(seed + my_rank);
  Vec_2<int> proc_size(tot_proc, 1);
  ini_rand(p_arr, n_par_gl, gl_l, sigma, myran, proc_size, comm);
#endif

  EM_ABD_aniso integrator(h0, Pe);
  double r_cut = 2.5;
  AmphiphilicWCA_2 f_Am(eps, lambda, C, r_cut);

#ifndef USE_MPI
  Grid_2 grid(gl_l, r_cut);
  PeriodicDomain_2 pdm(gl_l);
#else
  Grid_2 grid(gl_l, r_cut, proc_size, comm);
  PeriodicDomain_2 pdm(gl_l, grid, proc_size, comm);
  Communicator_2 communicator(pdm, grid, 1, 10);
#endif
  CellListNode_2<node_t> cl(pdm, grid);

  // cal force
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
  auto pair_force = [f1, f2, &cl]() {
    cl.for_each_pair_slow(f1, f2);
  };

  auto integrate = [&integrator, &myran, &pdm](node_t& p) {
    integrator.update(p, pdm, myran);
  };

  char xy_file[100];
  char log_file[100];
  snprintf(log_file, 100, "AmABP_Lx%g_Ly%g_p%g_v%g_C%g.log", gl_l.x, gl_l.y, phi, Pe, C);
#ifndef USE_MPI
  snprintf(xy_file, 100, "AmABP_Lx%g_Ly%g_p%g_v%g_C%g.extxzy", gl_l.x, gl_l.y, phi, Pe, C);
  LogExporter log_outer(log_file, 0, n_step, 10000, n_par_gl);
#else
  snprintf(xy_file, 100, "AmABP_Lx%g_Ly%g_p%g_v%g_C%g_n%d.extxzy", gl_l.x, gl_l.y, phi, Pe, C, my_rank);
  LogExporter log_outer(log_file, 0, n_step, 10000, n_par_gl, comm);
#endif

  XyzExporter_2 xy_outer(xy_file, 0, n_step, 1000, gl_l);
  auto exporter = [&xy_outer, &log_outer](int i_step, const std::vector<node_t>& par_arr) {
    log_outer.record(i_step);
    xy_outer.dump_doub_pos(i_step, par_arr);
  };
  exporter(0, p_arr);
  for (int i = 1; i <= n_step; i++) {
#ifdef USE_MPI
    one_step(i, p_arr, cl, integrate, pair_force, exporter, communicator);
#else
    one_step(i, p_arr, cl, integrate, pair_force, exporter);
#endif
  }
}
