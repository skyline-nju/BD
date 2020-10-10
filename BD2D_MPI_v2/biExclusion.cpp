/**
 * @file biExclusion.cpp
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

#include "config.h"
#include "biExclusion.h"
#ifdef TWO_EXCLUSION
int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
#ifdef _MSC_VER
  double Lx = 400;
  double Ly = 100;
  double phi = 0.5;
  double Dr = 0.1;
  double alpha = 5;
  double beta = 5;
  double r_cut = 1.;
  double h0 = 1e-2;
  double v0 = 1;
  int n_step = 1600000;
  int tot_proc;
  std::string ini_mode = "new";
  int snap_dt = 1000;
  MPI_Comm_size(MPI_COMM_WORLD, &tot_proc);
  Vec_2<int> proc_size(tot_proc, 1);
#else
  double Lx = atof(argv[1]);
  double Ly = atof(argv[2]);
  double phi = atof(argv[3]);
  double Dr = atof(argv[4]);
  double alpha = atof(argv[5]);
  double beta = atof(argv[6]);
  int n_step = atoi(argv[7]);
  Vec_2<int> proc_size(atoi(argv[8]), atoi(argv[9]));
  std::string ini_mode = argv[10];
  double h0 = 1e-2;
  double r_cut = 1.0;
  int snap_dt = 1000;
#endif
  typedef BP_u_tau_2 par_t;
  typedef BiNode<par_t> node_t;
  Vec_2<double> gl_l(Lx, Ly);
  Domain_2 dm(gl_l, proc_size, MPI_COMM_WORLD);
  std::vector<node_t> p_arr;
  int n_par_gl = int(phi * 4. * gl_l.x * gl_l.y / M_PI);
  PeriodicBdyCondi_2 bc(gl_l, proc_size);
  InverseRForce_2 f_rep(alpha, beta, r_cut);
  EM_ABD_aniso_Dt0 integrator(h0, v0);

  integrator.set_Dr(Dr);

  {
    // set output
    char prefix[100];
    snprintf(prefix, 100, "data/Cell_Lx%g_Ly%g_p%g_a%g_b%g_Dr%g_v%.1f",
      gl_l.x, gl_l.y, phi, alpha, beta, Dr, v0);

    Snap_GSD_2 snap(prefix, snap_dt, gl_l, ini_mode, MPI_COMM_WORLD);
    int t_first = 0;
    if (ini_mode == "new") {
      ini_rand(p_arr, n_par_gl, dm, bc);
    } else if (ini_mode == "restart") {
      ini_from_gsd(p_arr, n_par_gl, snap,  dm, false);
    } else{
      std::cout << "Wrong ini mode, which should be one of 'new', 'recreate'." << std::endl;
      exit(1);
    }

    Log log(prefix, 50000, n_par_gl, ini_mode, MPI_COMM_WORLD);
    auto exporter = [&log, &snap](int i_step, const std::vector<node_t>& par_arr) {
      log.dump(i_step);
      snap.dump(i_step, par_arr, false);
    };
    if (t_first == 0) {
      exporter(0, p_arr);
    }

    Ranq2 myran(1 + dm.get_proc_rank());
    CellList_2<par_t> cl(dm.get_box(), r_cut, gl_l, proc_size);
    cl.create(p_arr);
    dm.set_buf(r_cut, 10);
    for (int i = 1; i <= n_step; i++) {
      dm.cal_force(p_arr, cl, f_rep, bc, true);
      dm.integrate(p_arr, cl, integrator, bc, myran);
      exporter(i, p_arr);
    }
  }
  MPI_Finalize();
}
#endif