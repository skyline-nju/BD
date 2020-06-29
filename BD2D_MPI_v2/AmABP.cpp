/**
 * @file AmABP.cpp
 * @author Yu Duan (duanyu.nju@qq.com)
 * @brief Active Brownian particles with Amphiphilic force.
 * @version 0.1
 * @date 2020-06-29
 *
 * @copyright Copyright (c) 2020
 *
 */
#include "AmABP.h"

void AmphiphilicWCA_2::cal_force_torque(double r12_square, const Vec_2<double>& r12_vec, Vec_2<double>& f12_vec,
  const Vec_2<double>& q1, const Vec_2<double>& q2, double& tau1, double& tau2) const {
  double r = sqrt(r12_square);
  double r_2 = 1. / r12_square;
  double grad_V_pre = (lambda_ * r + 2.) * r_2;
  double V = C_ * exp(-lambda_ * (r - 1.)) * r_2;
  Vec_2<double> Vq1 = V * q1;
  Vec_2<double> Vq2 = V * q2;
  Vec_2<double> Vq1_m_q2 = Vq1 - Vq2;
  f12_vec = -(grad_V_pre * (Vq1_m_q2.dot(r12_vec))) * r12_vec + Vq1_m_q2;
  tau1 = Vq1.cross(r12_vec);
  tau2 = -Vq2.cross(r12_vec);
  if (r12_square < rcut_square_WCA_) {
    double r_6 = r_2 * r_2 * r_2;
    double tmp = eps24_ * (2 * r_6 * r_6 - r_6) * r_2;
    f12_vec += r12_vec * tmp;
  }
}

std::string AmphiphilicWCA_2::get_info() const {
  char info[100];
  snprintf(info, 100, "Amphiphilic--C:%g,lambda:%g,r_cut:%g|WCA--eps:%g",
    C_, lambda_, sqrt(rcut_square_AN_), eps24_ / 24);
  return info;
}

#ifdef AmphiphilicABP
int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
#ifdef _MSC_VER
  double Lx = 50;
  double Ly = 150;
  double phi = 0.1;
  double Pe = -50;
  int n_step = 100000;
  double lambda = 3.;
  double C = 6;
  double epsilon = 10.;
  double r_cut = 1.5;
  double Dr = 0.6;
  double h0 = 1e-5;
  int tot_proc;
  std::string ini_mode = "rand";
  //std::string ini_mode = "file";

  MPI_Comm_size(MPI_COMM_WORLD, &tot_proc);
  Vec_2<int> proc_size(tot_proc, 1);
  int snap_dt = 2000;
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
  std::string ini_mode = argv[11];
  double lambda = 3.;
  double epsilon = 10.;
  double r_cut = 1.5;
  int snap_dt = round(0.5 / h0);
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
    } else if (isdigit(ini_mode.c_str()[0])) {
      char prefix2[100];
      snprintf(prefix2, 100, "%s_t%d", prefix, atoi(ini_mode.c_str()));
      snprintf(prefix, 100, "%s", prefix2);
      ini_from_file(prefix, p_arr, n_par_gl, t_first, dm);
    } else {
      std::cout << "Wrong ini mode, which should be one of 'rand', 'file'." << std::endl;
      exit(1);
    }

    //XyzExporter_2 xy_outer(prefix, t_first, n_step, snap_dt, gl_l, MPI_COMM_WORLD);
    SnapExporter_2 snap_outer(prefix, t_first, n_step, snap_dt, file_info, MPI_COMM_WORLD);
    LogExporter log_outer(prefix, t_first, n_step, 50000, n_par_gl, MPI_COMM_WORLD);
    auto exporter = [&log_outer, &snap_outer](int i_step, const std::vector<node_t>& par_arr) {
      log_outer.record(i_step);
      //xy_outer.dump_pos_ori(i_step, par_arr);
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
      dm.cal_force(p_arr, cl, f_Am, bc, true);
      dm.integrate(p_arr, cl, integrator, bc, myran);
      exporter(i, p_arr);
    }
  }
  MPI_Finalize();
}
#endif
