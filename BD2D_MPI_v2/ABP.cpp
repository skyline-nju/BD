#include "config.h"
#ifdef ABP_CPP
#include "domain2D.h"
#include "particle2D.h"
#include "exporter2D.h"
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
  dm.ini_rand(p_arr, n_par_gl, bc);

  {
    WCAForce_2 f_wca(1., 1);
    double r_cut = f_wca.get_r_cut();
    EM_ABD_iso integrator(h0, Pe);
    CList_2<BP_u_2> cl(dm.get_box(), r_cut, gl_l, proc_size);

    char prefix[100];
    snprintf(prefix, 100, "ABP_Lx%g_Ly%g_p%g_v%g", gl_l.x, gl_l.y, phi, Pe);
    char file_info[200];
    snprintf(file_info, 200, "ABP2D with PBC;Lx=%g;Ly=%g;phi=%g;N=%d;Force=%s;h=%g;data=x,y",
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
  MPI_Finalize();
}
#endif