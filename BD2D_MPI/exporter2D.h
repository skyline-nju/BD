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

std::string add_suffix(const std::string& str, const std::string& suffix);

/**
 * @brief Basic class for exporting data.
 *
 * Define the timming to dump data.
 */
class ExporterBase {
public:
  ExporterBase(int start, int n_step, int sep) : start_(start), n_step_(n_step), sep_(sep) {
    set_lin_frame(start, n_step, sep);
  }

  void set_lin_frame(int start, int n_step, int sep);

  bool need_export(const int i_step);

protected:
  int n_step_;    // total steps to run
  int start_ = 0; // The first step
  int sep_;
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
  LogExporter(const std::string& outfile, int start, int n_step, int sep, int np);
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
#ifndef USE_MPI
  XyzExporter_2(const std::string &outfile, int start, int n_step, int sep,
                const Vec_2<double>& gl_l)
    : ExporterBase(start, n_step, sep), fout_(add_suffix(outfile, ".extxyz")), gl_l_(gl_l) {}
#else
  XyzExporter_2(const std::string &outfile, int start, int n_step, int sep,
                const Vec_2<double>& gl_l, MPI_Comm group_comm);
#endif
  template <typename TPar>
  void dump_pos(int i_step, const std::vector<TPar>& par_arr);

  template <typename TPar>
  void dump_doub_pos(int i_step, const std::vector<TPar>& par_arr);

  template <typename TPar>
  void dump_pos_ori(int i_step, const std::vector<TPar>& par_arr);

private:
  std::ofstream fout_;
  Vec_2<double> gl_l_;
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

class SnapExporter_2 : public ExporterBase {
public:
#ifndef USE_MPI
  SnapExporter_2(const std::string& filename, int start, int n_step, int sep,
                 const char* fileinfo);
#else
  SnapExporter_2(const std::string& filename, int start, int n_step, int sep,
                 const char* fileinfo, MPI_Comm group_comm);
#endif

  ~SnapExporter_2();

  template <typename TPar>
  void dump_pos(int i_step, const std::vector<TPar>& p_arr);

  template <typename TPar>
  void dump_pos_ori(int i_step, const std::vector<TPar>& p_arr);

  void write_info(const char* info);

  void write_data(const char* buf, size_t buf_size);

private:
  int count_ = 0;
#ifdef USE_MPI
  MPI_File fh_{};
  MPI_Comm comm_;
  MPI_Offset offset_;
  int my_rank_;
  int tot_proc_;
#else
  std::ofstream fout_;
#endif
 };

template<typename TPar>
void SnapExporter_2::dump_pos(int i_step, const std::vector<TPar>& p_arr) {
  if (i_step % sep_ == 0) {
    size_t n_par = p_arr.size();
    float* buf = new float[2 * n_par];
    for (int j = 0; j < n_par; j++) {
      buf[j * 2 + 0] = p_arr[j].pos.x;
      buf[j * 2 + 1] = p_arr[j].pos.y;
    }
    char frame_info[100];
    snprintf(frame_info, 100, "t=%d", i_step);
    write_info(frame_info);
    write_data((char*)buf, sizeof(buf[0]) * 2 * n_par);
  }
}

template<typename TPar>
void SnapExporter_2::dump_pos_ori(int i_step, const std::vector<TPar>& p_arr) {
  if (i_step % sep_ == 0) {
    size_t n_par = p_arr.size();
    float* buf = new float[3 * n_par];
    for (int j = 0; j < n_par; j++) {
      buf[j * 3 + 0] = p_arr[j].pos.x;
      buf[j * 3 + 1] = p_arr[j].pos.y;
      buf[j * 3 + 2] = p_arr[j].get_ori();
    }
    char frame_info[100];
    snprintf(frame_info, 100, "t=%d", i_step);
    write_info(frame_info);
    write_data((char*)buf, sizeof(buf[0]) * 3 * n_par);
  }
}
