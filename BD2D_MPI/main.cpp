#include "run2D.h"

int main(int argc, char* argv[]) {
#ifdef USE_MPI
  MPI_Init(&argc, &argv);
#endif
#ifdef _MSC_VER
  double Lx = 300;
  double Ly = 25;
  double phi = 0.1;
  double Pe = 0;
  int n_step = 5000;
  double epsilon = 10.;
  double lambda = 3.;
  double r_cut = 1.5;
  double C = 12;
#ifdef USE_MPI
  int tot_proc;
  MPI_Comm_size(MPI_COMM_WORLD, &tot_proc);
  Vec_2<int> proc_size(tot_proc, 1);
#endif
#else
  double Lx = atof(argv[1]);
  double Ly = atof(argv[2]);
  double phi = atof(argv[3]);
  double Pe = atof(argv[4]);
  int n_step = atoi(argv[5]);
  double lambda = 3.;
  double C = atof(argv[6]);
  double r_cut = atof(argv[7]);
  Vec_2<int> proc_size(atoi(argv[8]), atoi(argv[9]));
  double epsilon = 10.;
#endif
  unsigned long long seed = 1;
  int n_par_gl = int(phi * 4. * Lx * Ly / M_PI);
  double sigma = 1.;
  double h0 = 5e-6;
  Vec_2<double> l_gl(Lx, Ly);
#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
  //run_ABP(l_gl, phi, Pe, epsilon, h0, n_step, seed, proc_size, MPI_COMM_WORLD);
  run_ABP_Amphiphilic(l_gl, phi, Pe, epsilon, lambda, C, r_cut, h0, n_step, seed, proc_size, MPI_COMM_WORLD);
  MPI_Finalize();
#else
  //run_ABP(l_gl, phi, Pe, epsilon, h0, n_step, seed);
  run_ABP_Amphiphilic(l_gl, phi, Pe, epsilon, lambda, C, r_cut, h0, n_step, seed);
#endif
}