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
#include "config.h"
#ifdef VICSEK
#include "vicsek.h"
int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  double Lx = 25;
  double Ly = 25;
  double phi = 16;
  double gamma = 1.;
  //double v0 = 2.;
  double v0 = 0.2;
  double xi = 0.5;
  double rho_c1 = 1000000000;
  double rho_c2 = 10;
  double eps = 0.5;

  double r_cut = 1.;
  double h0 = 0.05;
  int n_step = 100000;
  int snap_dt = 20;

  int tot_proc;
  MPI_Comm_size(MPI_COMM_WORLD, &tot_proc);
  Vec_2<int> proc_size(tot_proc, 1);

  typedef VicsekPar par_t;
  typedef BiNode<par_t> node_t;
  Vec_2<double> gl_l(Lx, Ly);
  Domain_2 dm(gl_l, proc_size, MPI_COMM_WORLD);
  std::vector<node_t> p_arr;
  int n_par_gl = int(phi * gl_l.x * gl_l.y);
  PeriodicBdyCondi_2 bc(gl_l, proc_size);
  AligningForce_2 f_a(r_cut);
  EM_VM_Scheme3 integrator(h0, gamma, eps, v0, xi, rho_c1, rho_c2);

  {
    // set output
    char prefix[100];
    snprintf(prefix, 100, "VM_Lx%g_Ly%g_p%g_g%.2f_e%.2f_x%.2f_rc%.2f_%.2f", gl_l.x, gl_l.y, phi, gamma, eps, xi, rho_c1, rho_c2);
    char file_info[200];
    snprintf(file_info, 200, "VM with density-dependent motility;Lx=%g;Ly=%g;phi=%g;N=%d;h=%g;gamma=%g;eps=%g;data=x,y,theta;format=fff",
      gl_l.x, gl_l.y, phi, n_par_gl, h0, gamma, eps);

    int t_first = 0;
    ini_rand_VM(p_arr, n_par_gl, dm, bc, r_cut);

    //if (ini_mode == "rand") {
    //  ini_rand(p_arr, n_par_gl, dm, bc);
    //} else if (ini_mode == "file") {
    //  ini_from_file(prefix, p_arr, n_par_gl, t_first, dm);
    //} else if (isdigit(ini_mode.c_str()[0])) {
    //  char prefix2[100];
    //  snprintf(prefix2, 100, "%s_t%d", prefix, atoi(ini_mode.c_str()));
    //  //snprintf(prefix, 100, "%s", prefix2);
    //  ini_from_file(prefix2, p_arr, n_par_gl, t_first, dm);
    //} else {
    //  std::cout << "Wrong ini mode, which should be one of 'rand', 'file'." << std::endl;
    //  exit(1);
    //}

    XyzExporter_2 xy_outer(prefix, t_first, n_step,snap_dt, gl_l, MPI_COMM_WORLD);
    //SnapExporter_2 snap_outer(prefix, t_first, n_step, 200, file_info, MPI_COMM_WORLD);
    LogExporter log_outer(prefix, t_first, n_step, 50000, n_par_gl, MPI_COMM_WORLD);
    auto exporter = [&log_outer, &xy_outer](int i_step, const std::vector<node_t>& par_arr) {
      log_outer.record(i_step);
      xy_outer.dump_pos_ori(i_step, par_arr);
      //snap_outer.dump_pos_ori(i_step, par_arr);
    };
    if (t_first == 0) {
      exporter(0, p_arr);
    }

    Ranq2 myran(1 + dm.get_proc_rank());
    CellList_2<par_t> cl(dm.get_box(), r_cut, gl_l, proc_size);
    cl.create(p_arr);
    dm.set_buf(r_cut, 50);
    for (int i = 1; i <= n_step; i++) {
      dm.cal_force(p_arr, cl, f_a, bc, true);
      dm.integrate(p_arr, cl, integrator, bc, myran);
      exporter(i, p_arr);
      for (auto& p : p_arr) {
        p.n_neighbor = 0;
      }
    }

  }
  MPI_Finalize();
}
#endif