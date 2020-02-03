#include "run2D.h"

int main(int argc, char* argv[]) {
#ifdef USE_MPI
  MPI_Init(&argc, &argv);
#endif
#ifdef _MSC_VER
  double Lx = 40;
  double Ly = 20;
  double phi = 0.7;
  double Pe = 0;
  int n_step = 50000;
  double epsilon = 10.;
  double lambda = 3.;
  double C = 12;
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
  unsigned long long seed = 1;
  int n_par_gl = int(phi * 4. * Lx * Ly / M_PI);
  double sigma = 1.;
  double h0 = 5e-5;
  Vec_2<double> l_gl(Lx, Ly);
#ifdef USE_MPI
  //int rank, procs;
  //MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  //MPI_Comm_size(MPI_COMM_WORLD, &procs);
  //std::cout << rank << "/" << procs << std::endl;
  //run_ABP(l_gl, phi, Pe, epsilon, h0, n_step, seed, MPI_COMM_WORLD);
  run_ABP_Amphiphilic(l_gl, phi, Pe, epsilon, lambda, C, h0, n_step, seed, MPI_COMM_WORLD);
  MPI_Finalize();
#else
  run_ABP_Amphiphilic(l_gl, phi, Pe, epsilon, lambda, C, h0, n_step, seed);
#endif
}