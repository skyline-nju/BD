#pragma once

#include "config.h"
#include "vect.h"
#include "domain2D.h"
#include "particle2D.h"
#include "cellList2D.h"
#include "force2D.h"
#include "integrate2D.h"
#include "communicator2D.h"
#include "exporter2D.h"

template <typename TNode, typename TPairForce, typename TIntegrate, typename TExporter>
#ifndef USE_MPI
void one_step(int i_step, std::vector<TNode>& p_arr, CellListNode_2<TNode>& cl,
              TIntegrate integrate, TPairForce pair_force, TExporter exporter) {
#else
void one_step(int i_step, std::vector<TNode>& p_arr, CellListNode_2<TNode>& cl,
              TIntegrate integrate, TPairForce pair_force, TExporter exporter,
              Communicator_2& communicator) {
  int n_ghost = 0;
  communicator.comm_before_cal_force(p_arr, cl, n_ghost);
#endif
  pair_force();
#ifdef USE_MPI
  communicator.clear_padded_particles(cl, p_arr, n_ghost);
#endif
  const auto end = p_arr.end();
  for (auto it = p_arr.begin(); it != end; ++it) {
    integrate(*it);
  }
  cl.recreate(p_arr);

#ifdef USE_MPI
  communicator.comm_after_integration(p_arr, cl);
#endif
  exporter(i_step, p_arr);
}

template <typename node_t, typename TRan>
#ifndef USE_MPI
void ini_rand(std::vector<node_t>& p_arr, int n_par_gl,
              const Vec_2<double>& gl_l, double sigma, TRan &myran) {
  Grid_2 grid(gl_l, sigma);
  PeriodicDomain_2 pdm(gl_l);
#else
void ini_rand(std::vector<node_t>& p_arr, int n_par_gl,
              const Vec_2<double>& gl_l, double sigma, TRan& myran,
              const Vec_2<int> &proc_size, MPI_Comm comm) {
  int my_rank;
  MPI_Comm_rank(comm, &my_rank);
  Grid_2 grid(gl_l, sigma, proc_size, comm);
  PeriodicDomain_2 pdm(gl_l, grid, proc_size, comm);
#endif
  if (n_par_gl < gl_l.x * gl_l.y / 2) {
    create_rand_2(p_arr, n_par_gl, myran, pdm);
  } else {
    Communicator_2 communicator(pdm, grid);
    double r_cut = sigma;
    CellListNode_2<node_t> cl(pdm, grid);
    //WCAForce_2 f_pair(1., sigma);
    SpringForce_2 f_pair(500, sigma);
    EM_BD_iso integrator(1e-4);
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

    auto pair_force = [&f1, &f2, &cl]() {
      cl.for_each_pair_slow(f1, f2);
    };
    double sigma_new = 0.5;
    create_rand_2(p_arr, n_par_gl, myran, pdm, sigma_new);
    cl.create(p_arr);
    auto integrate = [&pdm, &myran, &integrator](node_t& p) {
      integrator.update(p, pdm, myran);
    };
    auto exporter = [](int i, const std::vector<node_t>& p_arr) {};
    do {
      sigma_new += 0.01;
      f_pair.set_sigma(sigma_new);
      for (int t = 0; t < 500; t++) {
#ifdef USE_MPI
        one_step(t, p_arr, cl, integrate, pair_force, exporter, communicator);
#else
        one_step(t, p_arr, cl, integrate, pair_force, exporter);
#endif
      }
    } while (sigma_new < sigma);
  }
  if (pdm.proc_rank().x == 0 && pdm.proc_rank().y == 0) {
    std::cout << "initialized!" << std::endl;
  }
}

#ifndef USE_MPI
void run_ABP(Vec_2<double>& gl_l, double phi, double Pe, double eps,
  double h0, int n_step, unsigned long long seed);
#else
void run_ABP(Vec_2<double>& gl_l, double phi, double Pe, double eps,
  double h0, int n_step, unsigned long long seed,
  const Vec_2<int>& proc_size, MPI_Comm comm);
#endif


#ifndef USE_MPI
void run_ABP_Amphiphilic(Vec_2<double>& gl_l, double phi, double Pe,
                         double eps, double lambda, double C, double r_cut,
                         double h0, int n_step, unsigned long long seed);
#else
void run_ABP_Amphiphilic(Vec_2<double>& gl_l, double phi, double Pe,
                         double eps, double lambda, double C, double r_cut,
                         double h0, int n_step, unsigned long long seed,
                         const Vec_2<int>& proc_size, MPI_Comm comm);
#endif