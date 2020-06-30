#pragma once

#include "config.h"
#include "rand.h"
#include "domain2D.h"
#include "particle2D.h"
#include "boundary2D.h"
#include "iodata2D.h"
#include "string.h"
#include "mpi.h"
#include <typeinfo>

template <typename TDomain, typename TPar, typename BdyCondi>
void ini_rand(std::vector<BiNode<TPar>>& p_arr, int n_par_gl, TDomain& dm,
              const BdyCondi& bc, double sigma = 1., bool avoid_overlap=true) {
  const Box_2<double>& box = dm.get_box();
  int n_max = int(box.l.x * box.l.y / (sigma * sigma) * 5);
  int my_rank = dm.get_proc_rank();
  int tot_proc = dm.get_proc_size();
  Vec_2<int> proc_size_vec = dm.get_proc_size_vec();
  MPI_Comm comm = dm.get_comm();
  p_arr.reserve(n_max);

  int n_par;
  if (my_rank < tot_proc - 1) {
    n_par = n_par_gl / tot_proc;
  } else {
    n_par = n_par_gl - n_par_gl / tot_proc * (tot_proc - 1);
  }

  int* n_par_arr = new int[tot_proc] {};
  MPI_Gather(&n_par, 1, MPI_INT, n_par_arr, 1, MPI_INT, 0, comm);
  if (my_rank == 0) {
    std::cout << "ini particle num: " << n_par_gl << " = " << n_par_arr[0];
    for (int i = 1; i < tot_proc; i++) {
      std::cout << " + " << n_par_arr[i];
    }
    std::cout << std::endl;
  }
  delete[] n_par_arr;

  Vec_2<double> origin = box.o;
  Vec_2<double> l = box.l;
  if (proc_size_vec.x > 1) {
    l.x -= sigma * 0.5;
    origin.x += sigma * 0.5;
  }
  if (proc_size_vec.y > 1) {
    l.y -= sigma * 0.5;
    origin.y += sigma * 0.5;
  }
  Ranq2 myran(1 + my_rank);

  if (!avoid_overlap || (n_par_gl < box.l.x * box.l.y / (sigma * sigma) / 2)) {
    create_rand_par_2(p_arr, n_par, origin, l, bc, myran, sigma);
  } else {
    double r_cut = sigma;
    dm.set_buf(r_cut, 10);
    EM_BD_iso integrator(1e-4);
    SpringForce_2 pair_force(500, sigma);
    CellList_2<TPar> cl(box, r_cut, dm.get_gl_l(), proc_size_vec);
    double sigma_new = 0.5 * sigma;
    create_rand_par_2(p_arr, n_par, origin, l, bc, myran, sigma_new);

    cl.create(p_arr);
    do {
      //std::cout << "sigma=" << sigma_new << std::endl;
      sigma_new += 0.01;
      pair_force.set_sigma(sigma_new);
      for (int i = 0; i < 500; i++) {
        //std::cout << "i = " << i << std::endl;
        dm.cal_force(p_arr, cl, pair_force, bc, false);
        dm.integrate(p_arr, cl, integrator, bc, myran);
      }
    } while (sigma_new < sigma);
  }
#ifdef INI_ORDERED
  for (auto& p : p_arr) {
    p.u.x = -1;
    p.u.y = 0;
  }
#endif
  if (my_rank == 0) {
    std::cout << "initialized randomly!\n";
    std::cout << "************************************\n" << std::endl;
  }
  MPI_Barrier(comm);
}

template <typename TPar, typename TDomain>
void ini_from_file(const std::string& file_in, std::vector<BiNode<TPar>>& p_arr,
                   int n_par_gl, int &t_last, const TDomain& dm, double sigma = 1.) {
  int float_per_par = 0;
  if (typeid(TPar) == typeid(BP_2)) {
    float_per_par = 2;
  } else if (typeid(TPar) == typeid(BP_theta_2) ||
    typeid(TPar) == typeid(BP_theta_tau_2) ||
    typeid(TPar) == typeid(BP_u_2) ||
    typeid(TPar) == typeid(BP_u_tau_2)) {
    float_per_par = 3;
  } else {
    std::cout << "Wrong particle type when reading from file\n";
    exit(2);
  }
  int tot_proc = dm.get_proc_size();
  int my_rank = dm.get_proc_rank();
  MPI_Comm comm = dm.get_comm();

  int buf_size = n_par_gl * float_per_par;
  float* buf = new float[buf_size];
  if (my_rank == 0) {
    load_last_frame(file_in, buf, t_last);
  }
  MPI_Bcast(buf, buf_size, MPI_FLOAT, 0, comm);
  MPI_Bcast(&t_last, 1, MPI_INT, 0, comm);
  PeriodicBdyCondi_2 pbc(dm.get_gl_l());
  const Box_2<double> box = dm.get_box();
  int n_max = int(box.l.x * box.l.y / (sigma * sigma) * 5);
  p_arr.reserve(n_max);
  int buf_pos = 0;  
  while (buf_pos < buf_size) {
    BiNode<TPar> p{};
    p.load_from_file(buf, buf_pos);
    pbc.tangle(p.pos);
    if (box.within(p.pos)) {
      p_arr.push_back(p);
    }
  }

  //std::cout << "buf size = " << buf_size << ", buf_pos = " << buf_pos
  //  << ", box = " << box << ", gl_l = " << dm.get_gl_l() << ", particle num = " << p_arr.size() << std::endl;

  int n_par = static_cast<int>(p_arr.size());
  int* n_par_arr = new int[tot_proc];
  MPI_Gather(&n_par, 1, MPI_INT, n_par_arr, 1, MPI_INT, 0, comm);
  if (my_rank == 0) {
    std::cout << "load from " << add_suffix(file_in, ".bin") 
      << ", particle num: " << n_par_gl << " = " << n_par_arr[0];
    int n_sum = n_par_arr[0];
    for (int i = 1; i < tot_proc; i++) {
      std::cout << " + " << n_par_arr[i];
      n_sum += n_par_arr[i];
    }
    std::cout << "\n";
    if (n_sum == n_par_gl) {
      std::cout << "initial t = " << t_last << "\n";
      std::cout << "************************************\n" << std::endl;
    } else {
      std::cout << "Error, particle num = " << n_sum << ", while " << n_par_gl << " particles are needed" << std::endl;
      exit(6);
    }
    
  }
  delete[] buf;
  delete[] n_par_arr;
  MPI_Barrier(comm);
}