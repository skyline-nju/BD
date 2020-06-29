/**
 * @brief Two types of exclusion interactions for SPP: 1) mechanical exclusion,
 *        wherein two particles mechanically repel each other when they overlap;
 *        2) scattering exclusion, wherein the directions along which each object
 *        tries to move are modulated to avoid overlapping.  
 *        Ref: PHYSICAL REVIEW E 99, 012614 (2019)
 */
#include "config.h"
#include "twoExclusion.h"
#ifdef TWO_EXCLUSION
int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
#ifdef _MSC_VER
  double Lx = 100;
  double Ly = 100;
  double phi = 0.4;
  double Dr = 0.1;
  double alpha = 1;
  double beta = 1;
  double r_cut = 1.;
  double h0 = 1e-2;
  int n_step = 20000;
  int tot_proc;
  std::string ini_mode = "rand";
  //std::string ini_mode = "file";

  MPI_Comm_size(MPI_COMM_WORLD, &tot_proc);
  Vec_2<int> proc_size(1, 1);
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
#endif
  typedef BP_u_tau_2 par_t;
  typedef BiNode<par_t> node_t;
  Vec_2<double> gl_l(Lx, Ly);
  Domain_2 dm(gl_l, proc_size, MPI_COMM_WORLD);
  std::vector<node_t> p_arr;
  int n_par_gl = int(phi * 4. * gl_l.x * gl_l.y / M_PI);
  PeriodicBdyCondi_2 bc(gl_l, proc_size);
  InverseRForce_2 f_rep(alpha, beta, r_cut);
  EM_ABD_aniso_Dt0 integrator(h0, 1.);

  integrator.set_Dr(Dr);

  {
    // set output
    char prefix[100];
    snprintf(prefix, 100, "Cell_Lx%g_Ly%g_p%g_a%g_b%g_Dr%g", gl_l.x, gl_l.y, phi, alpha, beta, Dr);
    char file_info[200];
    snprintf(file_info, 200, "Self-propelled cells with PBC;Lx=%g;Ly=%g;phi=%g;N=%d;h=%g;Dr=%g;data=x,y,theta;format=fff",
      gl_l.x, gl_l.y, phi, n_par_gl, h0, Dr);


    int t_first = 0;
    if (ini_mode == "rand") {
      ini_rand(p_arr, n_par_gl, dm, bc);
    } else if (ini_mode == "file") {
      ini_from_file(prefix, p_arr, n_par_gl, t_first, dm);
    } else if (isdigit(ini_mode.c_str()[0])) {
      char prefix2[100];
      snprintf(prefix2, 100, "%s_t%d", prefix, atoi(ini_mode.c_str()));
      //snprintf(prefix, 100, "%s", prefix2);
      ini_from_file(prefix2, p_arr, n_par_gl, t_first, dm);
    } else {
      std::cout << "Wrong ini mode, which should be one of 'rand', 'file'." << std::endl;
      exit(1);
    }

    XyzExporter_2 xy_outer(prefix, t_first, n_step, 200, gl_l, MPI_COMM_WORLD);
    SnapExporter_2 snap_outer(prefix, t_first, n_step, 200, file_info, MPI_COMM_WORLD);
    LogExporter log_outer(prefix, t_first, n_step, 50000, n_par_gl, MPI_COMM_WORLD);
    auto exporter = [&log_outer, &snap_outer, &xy_outer](int i_step, const std::vector<node_t>& par_arr) {
      log_outer.record(i_step);
      xy_outer.dump_pos_ori(i_step, par_arr);
      snap_outer.dump_pos_ori(i_step, par_arr);
    };
    if (t_first == 0) {
      exporter(0, p_arr);
    }

    Ranq2 myran(1 + dm.get_proc_rank());
    CellList_2<par_t> cl(dm.get_box(), r_cut, gl_l, proc_size);
    cl.create(p_arr);
    dm.set_buf(r_cut, 10);
    for (int i = 1; i <= n_step; i++) {
      dm.cal_force(p_arr, cl, f_rep, bc);
      dm.integrate(p_arr, cl, integrator, bc, myran);
      exporter(i, p_arr);
    }
  }
  MPI_Finalize();
}
#endif