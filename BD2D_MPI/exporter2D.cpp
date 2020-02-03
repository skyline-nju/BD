#include "exporter2D.h"

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
  int my_rank;
  MPI_Comm_rank(comm_, &my_rank);
  if (my_rank == 0) {
    fout.open(outfile);
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
  }
}
#else
LogExporter::LogExporter(const std::string& outfile,
                                   int start, int n_step, int sep, int np)
  : ExporterBase(start, n_step, sep), n_par_(np) {
  fout.open(outfile);
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
}
#endif

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
    const auto hour = int(dt / 3600);
    const auto min = int((dt - hour * 3600) / 60);
    const int sec = dt - hour * 3600 - min * 60;
    fout << i_step << "\t" << hour << ":" << min << ":" << sec << std::endl;
  }
  step_count_++;
}

