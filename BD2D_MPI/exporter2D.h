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

#ifdef USE_MPI
#include "mpi.h"
#endif

#ifdef _MSC_VER
const std::string delimiter("\\");
#else
const std::string delimiter("/");
#endif

/**
 * @brief Basic class for exporting data.
 *
 * Define the timming to dump data.
 */
class ExporterBase {
public:
  ExporterBase() : n_step_(0) {}

  ExporterBase(int start, int n_step, int sep) : start_(start), n_step_(n_step) {
    set_lin_frame(start, n_step, sep);
  }

  void set_lin_frame(int start, int n_step, int sep);

  bool need_export(const int i_step);

protected:
  int n_step_;    // total steps to run
  int start_ = 0; // The first step 
private:
  std::vector<int> frames_arr_; // frames that need to export
  std::vector<int>::iterator frame_iter_;
};

/**
 * @brief Exporter to output log
 *
 * Output the parameters after the initialization.
 * Output the beginning and endding time of the simulation.
 * Record time every certain time steps.
 */
class LogExporter : public ExporterBase {
public:
#ifdef USE_MPI
  LogExporter(const std::string& outfile, int start, int n_step, int sep,
    int np, MPI_Comm group_comm);
#else
  LogExporter(const std::string& outfile, int start, int n_step, int sep,
    int np);
#endif

  ~LogExporter();

  void record(int i_step);

  std::ofstream fout;
private:
  std::chrono::time_point<std::chrono::system_clock> t_start_;
  int n_par_;
#ifdef USE_MPI
  MPI_Comm comm_;
#endif
  int step_count_ = 0;
};


class XyzExporter_2 : public ExporterBase {
public:

  explicit XyzExporter_2(const std::string outfile, int start, int n_step, int sep,
    const Vec_2<double>& gl_l)
    : ExporterBase(start, n_step, sep), fout_(outfile), gl_l_(gl_l), sep_(sep) {}

  template <typename TPar>
  void dump_pos(int i_step, const std::vector<TPar>& par_arr);

  template <typename TPar>
  void dump_doub_pos(int i_step, const std::vector<TPar>& par_arr);

  template <typename TPar>
  void dump_pos_ori(int i_step, const std::vector<TPar>& par_arr);

private:
  std::ofstream fout_;
  Vec_2<double> gl_l_;
  int sep_;
};

template <typename TPar>
void XyzExporter_2::dump_pos(int i_step, const std::vector<TPar>& par_arr) {
  //if (need_export(i_step)) {
  if (i_step % sep_ == 0) {
    int n_par = par_arr.size();
    fout_ << n_par << "\n";
    // comment line
    fout_ << "Lattice=\"" << gl_l_.x << " 0 0 0 " << gl_l_.y << " 0 0 0 1\" "
      << "Properties=species:S:1:pos:R:2 Time=" << i_step;
    for (int j = 0; j < n_par; j++) {
      fout_ << "\n" << "N\t"
        << par_arr[j].pos.x << "\t" << par_arr[j].pos.y;
    }
    fout_ << std::endl;
  }
}

template<typename TPar>
void XyzExporter_2::dump_doub_pos(int i_step, const std::vector<TPar>& par_arr) {
  if (i_step % sep_ == 0) {
    int n_par = par_arr.size();
    fout_ << n_par * 2 << "\n";
    // comment line
    fout_ << "Lattice=\"" << gl_l_.x << " 0 0 0 " << gl_l_.y << " 0 0 0 1\" "
      << "Properties=species:S:1:pos:R:2 Time=" << i_step;
    for (int j = 0; j < n_par; j++) {
      fout_ << "\n" << "N\t"
        << par_arr[j].pos.x << "\t" << par_arr[j].pos.y;
      double dx = 0.01 * par_arr[j].u.x;
      double dy = 0.01 * par_arr[j].u.y;
      fout_ << "\n" << "O\t"
        << par_arr[j].pos.x + dx << "\t" << par_arr[j].pos.y + dy;
    }
    fout_ << std::endl;
  }
}

template <typename TPar>
void XyzExporter_2::dump_pos_ori(int i_step, const std::vector<TPar>& par_arr) {
  if (i_step % sep_ == 0) {
    int n_par = par_arr.size();
    fout_ << n_par << "\n";
    // comment line
    fout_ << "Lattice=\"" << gl_l_.x << " 0 0 0 " << gl_l_.y << " 0 0 0 1\" "
      << "Properties=species:S:1:pos:R:2:mass:M:1 Time=" << i_step;
    for (int j = 0; j < n_par; j++) {
      fout_ << "\n" << "N\t"
        << par_arr[j].pos.x << "\t" << par_arr[j].pos.y << "\t" << par_arr[j].get_ori();
    }
    fout_ << std::endl;
  }
}

/**
 * @brief Output snapshot as binary format.
 *
 * For each frame, the information of particles is saved as 3 * N float numbers.
 * 3 float number (x, y, theta) per particle.
 */
class SnapExporter : public ExporterBase {
public:
#ifdef USE_MPI
  explicit SnapExporter(const std::string outfile, int start, int n_step, int sep, MPI_Comm group_comm)
    : ExporterBase(start, n_step, sep), file_prefix_(outfile), comm_(group_comm) {}
#else
  explicit SnapExporter(const std::string outfile, int start, int n_step, int sep)
    : ExporterBase(start, n_step, sep), file_prefix_(outfile) {}
#endif
  template <typename TPar>
  void dump(int i_step, const std::vector<TPar>& p_arr);

private:
  int count_ = 0;
  std::string file_prefix_;
#ifdef USE_MPI
  MPI_File fh_{};
  MPI_Comm comm_;
#else
  std::ofstream fout_;
#endif
};

template<typename TPar>
void SnapExporter::dump(int i_step, const std::vector<TPar>& p_arr) {
  if (need_export(i_step)) {
    char filename[100];
    snprintf(filename, 100, "%s.%04d.bin", file_prefix_.c_str(), count_);
    count_++;
    int my_n = p_arr.size();
#ifdef USE_MPI
    MPI_File_open(comm_, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE,
      MPI_INFO_NULL, &fh_);
    int my_rank;
    MPI_Comm_rank(comm_, &my_rank);
    int tot_proc;
    MPI_Comm_size(comm_, &tot_proc);
    int my_origin;
    int* origin_arr = new int[tot_proc];
    int* n_arr = new int[tot_proc];
    MPI_Gather(&my_n, 1, MPI_INT, n_arr, 1, MPI_INT, 0, comm_);
    if (my_rank == 0) {
      origin_arr[0] = 0;
      for (int i = 1; i < tot_proc; i++) {
        origin_arr[i] = origin_arr[i - 1] + n_arr[i - 1];
      }
    }
    MPI_Scatter(origin_arr, 1, MPI_INT, &my_origin, 1, MPI_INT, 0, comm_);
    delete[] n_arr;
    delete[] origin_arr;

    MPI_Offset offset = my_origin * 3 * sizeof(float);
#else
    fout_.open(filename, std::ios::binary);
#endif
    float* buf = new float[3 * my_n];
    for (int j = 0; j < my_n; j++) {
      buf[j * 3 + 0] = p_arr[j].pos.x;
      buf[j * 3 + 1] = p_arr[j].pos.y;
      buf[j * 3 + 2] = p_arr[j].get_ori();
    }

#ifdef USE_MPI
    MPI_File_write_at(fh_, offset, buf, 3 * my_n, MPI_FLOAT, MPI_STATUSES_IGNORE);
    MPI_File_close(&fh_);
#else
    fout_.write(buf, sizeof(float) * my_n * 3);
    fout_.close()
#endif
      delete[] buf;
  }
}

