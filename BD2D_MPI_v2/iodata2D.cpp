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

void ExporterBase::set_lin_frame(int start, int n_step, int sep) {
  n_step_ = n_step;
  for (auto i = start + sep; i <= n_step_; i += sep) {
    frames_arr_.push_back(i);
  }
  frame_iter_ = frames_arr_.begin();
}

bool ExporterBase::need_export(int i_step) {
  bool flag = false;
  if (!frames_arr_.empty() && i_step == (*frame_iter_)) {
    frame_iter_++;
    flag = true;
  }
  return flag;
}

#ifdef USE_MPI
LogExporter::LogExporter(const std::string& outfile, 
                         int start, int n_step, int sep, 
                         int np, MPI_Comm group_comm)
  : ExporterBase(start, n_step, sep), n_par_(np), comm_(group_comm) {
#else
LogExporter::LogExporter(const std::string& outfile,
                         int start, int n_step, int sep, int np)
  : ExporterBase(start, n_step, sep), n_par_(np) {
#endif
#ifdef USE_MPI
  int my_rank;
  MPI_Comm_rank(comm_, &my_rank);
  if (my_rank == 0) {
#endif
    if (start == 0) {
      fout.open(add_suffix(outfile, ".log"));
    } else {
      fout.open(add_suffix(outfile, ".log"), std::ios::app);
    }
    t_start_ = std::chrono::system_clock::now();
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
#ifdef USE_MPI
  }
#endif
}

LogExporter::~LogExporter() {
  int tot_proc = 1;
#ifdef USE_MPI
  int my_rank;
  MPI_Comm_rank(comm_, &my_rank);
  MPI_Comm_size(comm_, &tot_proc);
  if (my_rank == 0) {
#endif
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
    fout << "speed=" << std::scientific << step_count_ * double(n_par_) / elapsed_seconds.count() / tot_proc
      << " particle time step per second per core\n";
    fout.close();
#ifdef USE_MPI
}
#endif
}

void LogExporter::record(int i_step) {
  bool flag;
#ifdef USE_MPI
  int my_rank;
  MPI_Comm_rank(comm_, &my_rank);
  flag = my_rank == 0 && need_export(i_step);
#else
  flag = need_export(i_step);
#endif
  if (flag) {
    const auto t_now = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = t_now - t_start_;
    const auto dt = elapsed_seconds.count();
    const auto hour = std::floor(dt / 3600);
    const auto min = std::floor((dt - hour * 3600) / 60);
    const int sec = dt - hour * 3600 - min * 60;
    fout << i_step + start_ << "\t" << hour << ":" << min << ":" << sec << std::endl;
  }
  step_count_++;
}

#ifndef USE_MPI
XyzExporter_2::XyzExporter_2(const std::string& outfile,
  int start, int n_step, int sep, const Vec_2<double>& gl_l)
  : ExporterBase(start, n_step, sep), gl_l_(gl_l) {
  if (start == 0) {
    fout_(add_suffix(outfile, ".extxyz"));
  } else {
    fout_(add_suffix(outfile, ".extxyz"), std::ios::app);
  }
}

#else
XyzExporter_2::XyzExporter_2(const std::string& outfile, int start, int n_step, int sep,
  const Vec_2<double>& gl_l, MPI_Comm group_comm)
  : ExporterBase(start, n_step, sep), gl_l_(gl_l) {
  int my_rank;
  MPI_Comm_rank(group_comm, &my_rank);
  char filename[100];
  snprintf(filename, 100, "%s_n%d.extxyz", outfile.c_str(), my_rank);
  if (start == 0) {
    fout_.open(filename);
  } else {
    fout_.open(filename, std::ios::app);
  }
}
#endif

#ifndef USE_MPI
SnapExporter_2::SnapExporter_2(const std::string& filename, int start, int n_step, int sep,
                               const char* fileinfo)
  : ExporterBase(start, n_step, sep) {
  if (start == 0) {
    fout_(add_suffix(filename, ".bin"), std::ios::binary);
  } else {
    fout_(add_suffix(filename, ".bin"), std::ios::binary|std::ios::app);
  }
  write_info(fileinfo);
}
#else
SnapExporter_2::SnapExporter_2(const std::string& filename, int start, int n_step, int sep,
                               const char* fileinfo, MPI_Comm group_comm)
  : ExporterBase(start, n_step, sep), comm_(group_comm), offset_(0) {
  MPI_Comm_rank(comm_, &my_rank_);
  MPI_Comm_size(comm_, &tot_proc_);
  if (start == 0) {
    MPI_File_open(comm_, add_suffix(filename, ".bin").c_str(),
                  MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &fh_);
    MPI_File_set_size(fh_, 0);
    write_info(fileinfo);
  } else {
    MPI_File_open(comm_, add_suffix(filename, ".bin").c_str(),
      MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &fh_);
    MPI_Offset file_size;
    MPI_File_get_size(fh_, &file_size);
    offset_ += file_size;
  }
}
#endif

SnapExporter_2::~SnapExporter_2() {
#ifndef USE_MPI
  fout_.close();
#else
  MPI_File_close(&fh_);
#endif
}

void SnapExporter_2::write_info(const char* info) {
  int info_size = static_cast<int>(strlen(info));
#ifndef USE_MPI
  fout_.write((char*)&info_size, 4);
  fout_.write(info, info_size);
#else
  if (my_rank_ == 0) {
    MPI_File_write_at(fh_, offset_, &info_size, 1, MPI_INT, MPI_STATUSES_IGNORE);
    MPI_File_write_at(fh_, offset_ + 4, info, info_size, MPI_CHAR, MPI_STATUSES_IGNORE);
  }
  MPI_Barrier(comm_);
  offset_ += (4 + static_cast<unsigned long long>(info_size));
#endif
}

void SnapExporter_2::write_data(const char* buf, size_t buf_size) {
#ifndef USE_MPI
  fout_.write((char*)buf_size, sizeof(buf_size));
  fout_.write(buf, buf_size);
#else
  size_t* buf_size_arr = new size_t[tot_proc_]{};
  MPI_Gather(&buf_size, 1, MPI_UINT64_T, buf_size_arr, 1, MPI_UINT64_T, 0, comm_);
  MPI_Offset* offset_arr = new MPI_Offset[tot_proc_]{};
  size_t frame_size = 0;
  if (my_rank_ == 0) {
    offset_arr[0] = 0;
    for (int i = 1; i < tot_proc_; i++) {
      offset_arr[i] = offset_arr[i - 1] + buf_size_arr[i - 1];
    }
    frame_size = offset_arr[tot_proc_ - 1] + buf_size_arr[tot_proc_ - 1];
    MPI_File_write_at(fh_, offset_, &frame_size, 1, MPI_UINT64_T, MPI_STATUSES_IGNORE);
  }
  offset_ += sizeof(frame_size);
  MPI_Offset my_offset;
  MPI_Scatter(offset_arr, 1, MPI_INT64_T, &my_offset, 1, MPI_INT64_T, 0, comm_);
  MPI_Bcast(&frame_size, 1, MPI_UINT64_T, 0, comm_);
  MPI_File_write_at(fh_, offset_+ my_offset, buf, buf_size, MPI_CHAR, MPI_STATUSES_IGNORE);
  offset_ += frame_size;
  delete[] buf_size_arr;
  delete[] offset_arr;
#endif
}

void load_last_frame(const std::string& filein, float* buf, int &t_last) {
  std::ifstream fin(add_suffix(filein, ".bin"), std::ios::binary);
  fin.seekg(0, std::ios::end);
  size_t file_size = fin.tellg();
  fin.seekg(0, std::ios::beg);
  
  int info_size;
  fin.read((char*)&info_size, 4);
  char info[200];
  //fin.read(info, info_size);
  //std::cout << info << std::endl;
  fin.seekg(info_size, std::ios::cur);
  //memset(info, '\0', sizeof(info));

  // get t and size of the last frame
  size_t frame_size = 0;
  while (fin.tellg() < file_size) {
    fin.read((char*)&info_size, 4);
    std::streampos pos1 = fin.tellg();
    fin.seekg(info_size, std::ios::cur);
    fin.read((char*)&frame_size, sizeof(frame_size));
    std::streampos pos2 = fin.tellg();
    if (pos2 == file_size - frame_size) {
      fin.seekg(pos1, std::ios::beg);
      fin.read(info, info_size);
      char* info_buf = nullptr;
#ifdef _MSC_VER
      char* ptr = strtok_s(info, "=", &info_buf);
      ptr = strtok_s(NULL, "=", &info_buf);
#else
      char* ptr = strtok_r(info, "=", &info_buf);
      ptr = strtok_r(NULL, "=", &info_buf);
#endif
      t_last = atoi(ptr);
      fin.seekg(pos2, std::ios::beg);
      fin.read((char*)buf, frame_size);
    } else {
      fin.seekg(frame_size, std::ios::cur);
    }
  }
  fin.close();
}
