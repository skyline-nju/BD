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
#include "iodata2D.h"
#include <string.h>
#include <sstream>
#include <cmath>

std::string add_suffix(const std::string& str, const std::string& suffix) {
  auto idx = str.find(suffix);
  std::string res;
  if (idx == std::string::npos) {
    res = str + suffix;
  } else {
    res = str;
  }
  return res;
}

#ifdef USE_MPI
BaseExporter::BaseExporter(int sep, MPI_Comm group_comm)
              : sep_(sep), comm_(group_comm) {
  MPI_Comm_rank(comm_, &my_rank_);
  MPI_Comm_size(comm_, &tot_proc_);
}
#else
BaseExporter::BaseExporter(int sep) : sep_(sep) {}
#endif

Log::Log(const std::string& fname, int sep, int n_par_gl,
         const std::string& open_flag, MPI_Comm group_comm)
         : BaseExporter(sep, group_comm), n_par_(n_par_gl) {
  t_start_ = std::chrono::system_clock::now();
  if (is_root()) {
    std::string filename = add_suffix(fname, ".log");
    if (open_flag == "new") {
      fout.open(filename);
    } else if (open_flag == "restart") {
      fout.open(filename, std::ios::app);
    }
    auto start_time = std::chrono::system_clock::to_time_t(t_start_);
    char str[100];
    tm now_time;
#ifdef _MSC_VER
    localtime_s(&now_time, &start_time);
#else
    localtime_r(&start_time, &now_time);
#endif
    std::strftime(str, 100, "%c", &now_time);
    fout << "Started simulation at " << str << "\n";
    fout << "Particle number=" << n_par_ << "\n";
    fout << "Proc number=" << tot_proc_ << "\n";
  }
}

void Log::dump(int i_step) {
  if (is_root() && need_dump(i_step) && i_step != 0) {
    const auto t_now = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = t_now - t_start_;
    const auto dt = elapsed_seconds.count();
    const auto hour = std::floor(dt / 3600);
    const auto min = std::floor((dt - hour * 3600) / 60);
    const int sec = dt - hour * 3600 - min * 60;
    fout << i_step << "\t" << hour << ":" << min << ":" << sec << std::endl;
  }
  simulation_steps_++;
}

Log::~Log() {
  if (is_root()) {
    const auto t_now = std::chrono::system_clock::now();
    auto end_time = std::chrono::system_clock::to_time_t(t_now);
    char str[100];
    tm now_time;
#ifdef _MSC_VER
    localtime_s(&now_time, &end_time);
#else
    localtime_r(&end_time, &now_time);
#endif
    std::strftime(str, 100, "%c", &now_time);
    fout << "Finished simulation at " << str << "\n";
    std::chrono::duration<double> elapsed_seconds = t_now - t_start_;
    double speed = simulation_steps_ * double(n_par_) / elapsed_seconds.count() / tot_proc_;
    fout << "speed=" << std::scientific << speed << " particle time step per second per core\n";
    fout << "----------------" << std::endl;
    fout.close();
  }
}

Snap_GSD_2::Snap_GSD_2(const std::string& filename, int sep, const Vec_2<double>& gl_l,
                       const std::string& open_flag, MPI_Comm group_comm)
                       : BaseExporter(sep, group_comm) {
  unsigned int version = gsd_make_version(1, 4);
  char fname[100];
  snprintf(fname, 100, "%s", add_suffix(filename, ".gsd").c_str());
  if (is_root()) {
    handle_ = new gsd_handle;
    if (open_flag == "new") {
      gsd_create(fname, "cpp", "hoomd", version);
      gsd_open(handle_, fname, GSD_OPEN_READWRITE);
      float box[6] = { gl_l.x, gl_l.y, 1, 0, 0, 0 };
      gsd_write_chunk(handle_, "configuration/box", GSD_TYPE_FLOAT, 6, 1, 0, box);
    } else if (open_flag == "restart") {
      gsd_open(handle_, fname, GSD_OPEN_READWRITE);
    } else {
      std::cout << "Wrong open flag, which must be one of 'new' or 'restart'!" << std::endl;
      exit(1);
    }
  }
}

Snap_GSD_2::~Snap_GSD_2() {
  if (is_root()) {
    gsd_close(handle_);
    delete handle_;
  }
}

void Snap_GSD_2::load_frame(int i_frame, float* buf, int buf_size) {
  if (my_rank_ == 0) {
    int n_frame = gsd_get_nframes(handle_);
    std::cout << "nframes = " << n_frame << std::endl;
    if (i_frame >= n_frame) {
      std::cout << i_frame << "should be less than total frames " << n_frame << std::endl;
      exit(1);
    } else if (i_frame == -1) {
      i_frame = n_frame - 1;
    }
    std::cout << "load " << i_frame << " frame" << std::endl;
    const gsd_index_entry* chunk_N = gsd_find_chunk(handle_, i_frame, "particles/N");
    unsigned int n;
    gsd_read_chunk(handle_, &n, chunk_N);
    if (n * 3 != buf_size) {
      std::cout << "particle number = " << n << " is not matched with buf size = ";
      std::cout << buf_size << std::endl;
      exit(1);
    }
    const gsd_index_entry* chunk = gsd_find_chunk(handle_, i_frame, "particles/position");
    gsd_read_chunk(handle_, buf, chunk);
  }
  MPI_Bcast(buf, buf_size, MPI_FLOAT, 0, comm_);
}

#ifndef USE_MPI
XyzExporter_2::XyzExporter_2(const std::string& outfile, int sep,
                             const Vec_2<double>& gl_l,
                             const std::string& open_flag)
                             : BaseExporter(sep), gl_l_(gl_l) {
  if (open_flag == "w") {
    fout_.open(add_suffix(outfile, ".extxyz"));
  } else {
    fout_.open(add_suffix(outfile, ".extxyz"), std::ios::app);
  }
}

#else
XyzExporter_2::XyzExporter_2(const std::string& outfile, int sep,
                             const Vec_2<double>& gl_l, 
                             const std::string& open_flag,
                             MPI_Comm group_comm)
                             : BaseExporter(sep, group_comm), gl_l_(gl_l) {
  if (is_root()) {
    if (open_flag == "new") {
      fout_.open(add_suffix(outfile, ".extxyz"));
    } else {
      fout_.open(add_suffix(outfile, ".extxyz"), std::ios::app);
    }
  }
}
#endif