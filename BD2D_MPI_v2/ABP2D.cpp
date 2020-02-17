#include "ABP2D.h"

#ifdef ABP
int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
#ifdef _MSC_VER
  double Lx = 300;
  double Ly = 25;
  double phi = 0.5;
  double Pe = 0;
  int n_step = 30000;
  double epsilon = 10.;
  int tot_proc;
  MPI_Comm_size(MPI_COMM_WORLD, &tot_proc);
  Vec_2<int> proc_size(tot_proc, 1);
#else
  double Lx = atof(argv[1]);
  double Ly = atof(argv[2]);
  double phi = atof(argv[3]);
  double Pe = atof(argv[4]);
  int n_step = atoi(argv[5]);
  Vec_2<int> proc_size(atoi(argv[6]), atoi(argv[7]));
  double epsilon = 10.;
#endif

  double h0 = 1e-5;
  Vec_2<double> gl_l(Lx, Ly);
  Domain_2 dm(gl_l, proc_size, MPI_COMM_WORLD);
  std::vector< BiNode<BP_u_2>> p_arr;
  int n_par_gl = int(phi * 4. * gl_l.x * gl_l.y / M_PI);
  PeriodicBdyCondi_2 bc(gl_l, proc_size);
  ini_rand(p_arr, n_par_gl, dm, bc);

  {
    WCAForce_2 f_wca(1., 1);
    double r_cut = f_wca.get_r_cut();
    dm.set_buf(r_cut, 10);
    EM_ABD_iso integrator(h0, Pe);
    CellList_2<BP_u_2> cl(dm.get_box(), r_cut, gl_l, proc_size);

    char prefix[100];
    snprintf(prefix, 100, "ABP_Lx%g_Ly%g_p%g_v%g", gl_l.x, gl_l.y, phi, Pe);
    char file_info[200];
    snprintf(file_info, 200, "ABP2D with PBC;Lx=%g;Ly=%g;phi=%g;N=%d;Force=%s;h=%g;data=x,y;format=ff",
      gl_l.x, gl_l.y, phi, n_par_gl, f_wca.get_info().c_str(), h0);
    XyzExporter_2 xy_outer(prefix, 0, n_step, 10000, gl_l, MPI_COMM_WORLD);
    //SnapExporter_2 snap_outer(prefix, 0, n_step, 1000, file_info, MPI_COMM_WORLD);
    LogExporter log_outer(prefix, 0, n_step, 10000, n_par_gl, MPI_COMM_WORLD);

    auto exporter = [&log_outer, &xy_outer](int i_step, const std::vector<BiNode<BP_u_2>>& par_arr) {
      log_outer.record(i_step);
      xy_outer.dump_pos(i_step, par_arr);
      //snap_outer.dump_pos(i_step, par_arr);
    };
    exporter(0, p_arr);
    Ranq2 myran(1);
    for (int i = 1; i <= n_step; i++) {
      dm.cal_force(p_arr, cl, f_wca, bc);
      dm.integrate(p_arr, cl, integrator, bc, myran);
      exporter(i, p_arr);
    }
  }
  std::cout << "finished " << std::endl;
  MPI_Finalize();
}
#endif

#ifdef AmphiphilicABP
int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
#ifdef _MSC_VER
  double Lx = 100;
  double Ly = 80;
  double phi = 0.5;
  double Pe = 0;
  int n_step = 40000;
  double lambda = 3.;
  double C = 12;
  double epsilon = 10.;
  double r_cut = 1.5;
  double Dr = 3.;
  double h0 = 1e-5;
  int tot_proc;
  std::string ini_mode = "rand";
  //std::string ini_mode = "file";

  MPI_Comm_size(MPI_COMM_WORLD, &tot_proc);
  Vec_2<int> proc_size(2, 2);
#else
  double Lx = atof(argv[1]);
  double Ly = atof(argv[2]);
  double phi = atof(argv[3]);
  double Pe = atof(argv[4]);
  int n_step = atoi(argv[5]);
  double C = atof(argv[6]);
  double Dr = atof(argv[7]);
  double h0 = atof(argv[8]);
  Vec_2<int> proc_size(atoi(argv[9]), atoi(argv[10]));
  double lambda = 3.;
  double epsilon = 10.;
  double r_cut = 1.5;
#endif
  typedef BP_u_tau_2 par_t;
  typedef BiNode<par_t> node_t;
  Vec_2<double> gl_l(Lx, Ly);
  Domain_2 dm(gl_l, proc_size, MPI_COMM_WORLD);
  std::vector<node_t> p_arr;
  int n_par_gl = int(phi * 4. * gl_l.x * gl_l.y / M_PI);
  PeriodicBdyCondi_2 bc(gl_l, proc_size);
  AmphiphilicWCA_2 f_Am(epsilon, lambda, C, r_cut);
  EM_ABD_aniso integrator(h0, Pe);

  integrator.set_Dr(Dr);

  {
    // set output
    char prefix[100];
    snprintf(prefix, 100, "AmABP_Lx%g_Ly%g_p%g_v%g_C%g_Dr%g", gl_l.x, gl_l.y, phi, Pe, C, Dr);
    char file_info[200];
    snprintf(file_info, 200, "amphiphilic ABP2D with PBC;Lx=%g;Ly=%g;phi=%g;N=%d;Force=%s;h=%g;Dr=%g;data=x,y,theta;format=fff",
      gl_l.x, gl_l.y, phi, n_par_gl, f_Am.get_info().c_str(), h0, Dr);

 
    int t_first = 0;
    if (ini_mode == "rand") {
      ini_rand(p_arr, n_par_gl, dm, bc);
    } else if (ini_mode == "file") {
      ini_from_file(prefix, p_arr, n_par_gl, t_first, dm);
    } else {
      std::cout << "Wrong ini mode, which should be one of 'rand', 'file'." << std::endl;
      exit(1);
    }


    //XyzExporter_2 xy_outer(prefix, t_first, n_step, 10000, gl_l, MPI_COMM_WORLD);
    SnapExporter_2 snap_outer(prefix, t_first, n_step, 10000, file_info, MPI_COMM_WORLD);
    LogExporter log_outer(prefix, t_first, n_step, 50000, n_par_gl, MPI_COMM_WORLD);
    auto exporter = [&log_outer, &snap_outer](int i_step, const std::vector<node_t>& par_arr) {
      log_outer.record(i_step);
      //xy_outer.dump_doub_pos(i_step, par_arr);
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
      dm.cal_force(p_arr, cl, f_Am, bc);
      dm.integrate(p_arr, cl, integrator, bc, myran);
      exporter(i, p_arr);
    }
  }
  MPI_Finalize();
}
#endif