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
#ifdef WALL_Y
  l.y -= sigma;
  origin.y += sigma * 0.5;
#else
  if (proc_size_vec.y > 1) {
    l.y -= sigma * 0.5;
    origin.y += sigma * 0.5;
  }
#endif

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
void ini_from_gsd(std::vector<BiNode<TPar>>& p_arr, int n_par_gl, Snap_GSD_2& snap,
                  const TDomain& dm, bool oppsite_ori, double sigma = 1.) {
  int buf_size = n_par_gl * 3;
  float* buf = new float[buf_size];
  snap.load_frame(-1, buf, buf_size);
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
      if (oppsite_ori) {
        p.u = -p.u;
      }
      p_arr.push_back(p);
    }
  }
}
