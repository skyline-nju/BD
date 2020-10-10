/**
 * @file iodata2D.h
 * @author Yu Duan (duanyu.nju@qq.com)
 * @brief Classes for output data and read data
 * @version 0.1
 * @date 2020-07-26
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#pragma once
#include <vector>
#include <chrono>
#include <ctime>
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include "config.h"
#include "particle2D.h"
#include "gsd.h"

#ifdef USE_MPI
#include "mpi.h"
#endif

#ifdef _MSC_VER
const std::string delimiter("\\");
#else
const std::string delimiter("/");
#endif

std::string add_suffix(const std::string& str, const std::string& suffix);

template <typename TPar>
int gather_particles(const std::vector<TPar>& p_arr, float** buf_gl,
                     MPI_Comm comm, bool oppsite_ori) {
  int my_rank;
  int tot_proc;
  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &tot_proc);
  int n_par = p_arr.size();
  float* buf = new float[3 * n_par];
  for (int j = 0; j < n_par; j++) {
    buf[j * 3 + 0] = float(p_arr[j].pos.x);
    buf[j * 3 + 1] = float(p_arr[j].pos.y);
    double theta = p_arr[j].get_ori();
    if (oppsite_ori) {
      if (theta > 0) {
        theta -= M_PI;
      } else {
        theta += M_PI;
      }
    }
    buf[j * 3 + 2] = theta;
  }

  int* n_par_arr = new int[tot_proc];
  MPI_Gather(&n_par, 1, MPI_INT, n_par_arr, 1, MPI_INT, 0, comm);
  int* recvcounts = new int[tot_proc]();
  int* displs = new int[tot_proc]();
  int n_par_gl = 0;
  if (my_rank == 0) {
    for (int i = 0; i < tot_proc; i++) {
      recvcounts[i] = n_par_arr[i] * 3;
      n_par_gl += n_par_arr[i];
    }
    displs[0] = 0;
    for (int i = 1; i < tot_proc; i++) {
      displs[i] = displs[i - 1] + recvcounts[i - 1];
    }
    *buf_gl = new float[n_par_gl * 3];
  }

  MPI_Gatherv(buf, n_par * 3, MPI_FLOAT, *buf_gl, recvcounts, displs, MPI_FLOAT, 0, comm);
  delete[] n_par_arr;
  delete[] recvcounts;
  delete[] displs;
  delete[] buf;
  return n_par_gl;
}

class BaseExporter {
public:
#ifdef USE_MPI
  BaseExporter(int sep, MPI_Comm group_comm);
#else
  BaseExporter(int sep);
#endif
  bool need_dump(int i_step) { return i_step % sep_ == 0; }

  bool is_root() { return my_rank_ == 0; }
protected:
  int sep_;
#ifdef USE_MPI
  MPI_Comm comm_;
#endif
  int my_rank_ = 0;
  int tot_proc_ = 1;
};

class Log : public BaseExporter {
public:
  Log(const std::string& fname,
      int sep,
      int n_par_gl,
      const std::string& open_flag,
      MPI_Comm group_comm);

  ~Log();

  void dump(int i_step);
  std::ofstream fout;
private:
  std::chrono::time_point<std::chrono::system_clock> t_start_;
  int n_par_;
  size_t simulation_steps_ = 0;
};

class Snap_GSD_2 : public BaseExporter {
public:
  Snap_GSD_2(const std::string& filename, int sep, const Vec_2<double>& gl_l,
             const std::string& open_flag, MPI_Comm group_comm);

  ~Snap_GSD_2();

  template <typename TPar>
  void dump(int i_step, const std::vector<TPar>& p_arr, bool oppsite_ori);
 
  void load_frame(int i_frame, float* buf, int buf_size);

private:
  gsd_handle* handle_ = nullptr;
};

template<typename TPar>
void Snap_GSD_2::dump(int i_step, const std::vector<TPar>& p_arr, bool oppsite_ori) {
  if (need_dump(i_step)) {
    float* buf_gl = nullptr;
    int n_par_gl = gather_particles(p_arr, &buf_gl, comm_, oppsite_ori);
    if (my_rank_ == 0) {
      size_t n_frame = gsd_get_nframes(handle_);
      size_t i_frame;
      if (n_frame == 0) {
        i_frame = 0;
      } else {
        const gsd_index_entry* chunk = gsd_find_chunk(handle_, n_frame - 1, "configuration/step");
        gsd_read_chunk(handle_, &i_frame, chunk);
        i_frame++;
      }
      std::cout << "dump frame " << i_frame << std::endl;
      gsd_write_chunk(handle_, "configuration/step", GSD_TYPE_UINT64, 1, 1, 0, &i_frame);
      gsd_write_chunk(handle_, "particles/N", GSD_TYPE_UINT32, 1, 1, 0, &n_par_gl);
      gsd_write_chunk(handle_, "particles/position", GSD_TYPE_FLOAT, n_par_gl, 3, 0, buf_gl);
      gsd_end_frame(handle_);
    }
    delete[] buf_gl;
  }
}

class XyzExporter_2 : public BaseExporter {
public:
#ifndef USE_MPI
  XyzExporter_2(const std::string& outfile, int sep,
                const Vec_2<double>& gl_l, const std::string& open_flag);
#else
  XyzExporter_2(const std::string& outfile, int sep,
                const Vec_2<double>& gl_l,
                const std::string& open_flag,
                MPI_Comm group_comm);
#endif

  template <typename TPar>
  void dump_pos_ori(int i_step, const std::vector<TPar>& par_arr);

private:
  std::ofstream fout_;
  Vec_2<double> gl_l_;
};

template <typename TPar>
void XyzExporter_2::dump_pos_ori(int i_step, const std::vector<TPar>& par_arr) {
  if (need_dump(i_step)) {
    float* buf_gl = nullptr;
    int n_par_gl = gather_particles(par_arr, &buf_gl, comm_, true);
    if (is_root) {
      fout_ << n_par_gl << "\n";
      // comment line
      fout_ << "Lattice=\"" << gl_l_.x << " 0 0 0 " << gl_l_.y << " 0 0 0 1\" "
        << "Properties=species:S:1:pos:R:2:mass:M:1 Time=" << i_step;
      for (int j = 0; j < n_par_gl; j++) {
        float x = buf_gl[j * 3];
        float y = buf_gl[j * 3 + 1];
        float theta = buf_gl[j * 3 + 2];
        fout_ << "\n" << "N\t"
          << x << "\t" << y << "\t" << theta;
      }
    }
    delete[] buf_gl;
  }
}