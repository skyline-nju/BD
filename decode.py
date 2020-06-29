"""
Decode the .bin file where the data are saved as binary format.
2020/2/8
"""
import numpy as np
import struct
import matplotlib.pyplot as plt
import os


def get_para(info, beg=1):
    str_list = info.split(";")
    para_dict = {}
    for s in str_list[beg:]:
        s1, s2 = s.split("=")
        try:
            para_dict[s1] = int(s2)
        except ValueError:
            try:
                para_dict[s1] = float(s2)
            except ValueError:
                para_dict[s1] = s2
    return para_dict


def get_t(info):
    para_dict = get_para(info, 0)
    return para_dict["t"]


def get_para_from_file(file_in, beg=0):
    with open(file_in, "rb") as f:
        buf = f.read(4)
        info_size, = struct.unpack("i", buf)
        buf = f.read(info_size)
        info = buf.decode()
        para_dict = get_para(info)
    return para_dict


def show_file_info(fin):
    with open(fin, "rb") as f:
        buf = f.read(4)
        info_size, = struct.unpack("i", buf)
        buf = f.read(info_size)
        info = buf.decode()
        print(info)


def get_file_size(f):
    f.seek(0, 2)
    file_size = f.tell()
    f.seek(0, 0)
    return file_size


def get_n_frames(fin):
    with open(fin, "rb") as f:
        file_size = get_file_size(f)
        buf = f.read(4)
        info_size, = struct.unpack("i", buf)
        f.seek(info_size, 1)
        n = 0

        while f.tell() < file_size:
            buf = f.read(4)
            info_size, = struct.unpack("i", buf)
            f.seek(info_size, 1)
            buf = f.read(8)
            data_size, = struct.unpack("Q", buf)
            f.seek(data_size, 1)
            n += 1
    return n


def read_pos(fin, sep, start=0):
    with open(fin, "rb") as f:
        file_size = get_file_size(f)
        buf = f.read(4)
        info_size, = struct.unpack("i", buf)
        buf = f.read(info_size)
        info = buf.decode()
        print(info)
        para_dict = get_para(info)

        while f.tell() < file_size:
            buf = f.read(4)
            info_size, = struct.unpack("i", buf)
            buf = f.read(info_size)
            info = buf.decode()
            t = int(info.split("=")[1])
            buf = f.read(8)
            data_size, = struct.unpack("Q", buf)
            if (t % sep == 0 and t >= start):
                buf = f.read(data_size)
                data = struct.unpack("%df" % (data_size // 4), buf)
                x, y = np.array(data).reshape(para_dict["N"], 2).T
                yield t, x, y
            else:
                f.seek(data_size, 1)


def read_pos_theta(fin, sep, t_start=0, frame_start=0):
    with open(fin, "rb") as f:
        file_size = get_file_size(f)
        buf = f.read(4)
        info_size, = struct.unpack("i", buf)
        buf = f.read(info_size)
        info = buf.decode()
        print(info)
        para_dict = get_para(info)
        print("file size =", file_size)
        n_frame = 0
        while f.tell() < file_size:
            buf = f.read(4)
            info_size, = struct.unpack("i", buf)
            buf = f.read(info_size)
            info = buf.decode()
            t = int(info.split("=")[1])
            buf = f.read(8)
            data_size, = struct.unpack("Q", buf)
            # print("data size", data_size)
            n_frame += 1
            if (t % sep == 0 and t >= t_start and frame_start < n_frame):
                buf = f.read(data_size)
                data = struct.unpack("%df" % (data_size // 4), buf)
                try:
                    x, y, theta = np.array(data).reshape(para_dict["N"], 3).T
                except ValueError:
                    x, y, theta = np.array(data).reshape(para_dict["N"]*2, 3).T
                yield t, x, y, theta
            else:
                f.seek(data_size, 1)


def get_one_frame(file_in, t1=None):
    fin = open(file_in, "rb")
    fin_size = get_file_size(fin)
    buf_0 = fin.read(4)
    info_size, = struct.unpack("i", buf_0)
    buf_1 = fin.read(info_size)
    while fin.tell() < fin_size:
        buf_2 = fin.read(4)
        info_size, = struct.unpack("i", buf_2)
        buf_3 = fin.read(info_size)
        info = buf_3.decode()
        t = int(info.split("=")[1])
        buf_4 = fin.read(8)
        data_size, = struct.unpack("Q", buf_4)
        print("t =", t, end="\r")
        if (t1 is None and fin.tell() + data_size == fin_size) or t == t1:
            buf_5 = fin.read(data_size)
            if t1 is None:
                t1 = t
            break
        else:
            fin.seek(data_size, 1)
    fin.close()

    file_out = file_in.rstrip(".bin") + "_t%d.bin" % t1
    fout = open(file_out, "wb")
    fout.write(buf_0)
    fout.write(buf_1)
    fout.write(buf_2)
    fout.write(buf_3)
    fout.write(buf_4)
    fout.write(buf_5)
    fout.close()


def get_one_image_frame(file_in, t1=None):
    para_dict = get_para_from_file(file_in)
    fin = open(file_in, "rb")
    fin_size = get_file_size(fin)
    buf_0 = fin.read(4)
    info_size, = struct.unpack("i", buf_0)
    buf_1 = fin.read(info_size)
    while fin.tell() < fin_size:
        buf_2 = fin.read(4)
        info_size, = struct.unpack("i", buf_2)
        buf_3 = fin.read(info_size)
        info = buf_3.decode()
        t = int(info.split("=")[1])
        buf_4 = fin.read(8)
        data_size, = struct.unpack("Q", buf_4)
        print("t =", t, end="\r")
        if (t1 is None and fin.tell() + data_size == fin_size) or t == t1:
            buf_5 = fin.read(data_size)
            N = data_size // 12
            data = struct.unpack("%df" % (N * 3), buf_5)
            x, y, theta = np.array(data).reshape(N, 3).T
            x_new, y_new, theta_new = np.zeros((3, N * 2), dtype="float32")
            x_new[:N] = x
            x_new[N:] = para_dict["Lx"] * 2 - x
            y_new[:N] = y
            y_new[N:] = y
            theta_new[:N] = theta
            theta_new[N:] = np.pi - theta
            # data_new = np.array([x_new, y_new, theta_new]).T
            buf_4 = struct.pack("Q", data_size * 2)
            if t1 is None:
                t1 = t
            break
        else:
            fin.seek(data_size, 1)
    fin.close()

    file_out = file_in.rstrip(".bin") + "_t%d.bin" % t1
    fout = open(file_out, "wb")
    fout.write(buf_0)
    fout.write(buf_1)
    fout.write(buf_2)
    fout.write(buf_3)
    fout.write(buf_4)
    for i in range(N * 2):
        buf = struct.pack("fff", x_new[i], y_new[i], theta_new[i])
        fout.write(buf)
    # fout.write(buf_5)
    fout.close()


if __name__ == "__main__":
    # # fname = "AmABP_Lx40_Ly20_p0.7_v0_C12.bin"
    # fname = "D:\\data\\ABP_test\\PBC_MPI\\open_mpi\\AmABP_Lx150_Ly75_p0.55_v-180_C12.bin"
    # # show_file_info(fname)
    # frames = read_pos_theta(fname, True)
    # for x, y, theta in frames:
    #     plt.plot(x, y, ".")
    #     plt.show()
    #     plt.close()
    # fname = "BD2D_MPI_v2\\AmABP_Lx50_Ly150_p0.01_v50_C6_Dr0.6.bin"
    # para_dict = get_para_from_file(fname)
    # print(para_dict)

    # fname = "D:\\data\\ABP_test\\Ly150_ini_ordered\\AmABP_Lx2100_Ly150_p0.5_v-50_C6_Dr0.8.bin"
    # fname = "D:\\data\\ABP_test\\Ly150_ini_ordered\\AmABP_Lx2100_Ly150_p0.45_v-50_C6_Dr0.8_t22260000.bin"
    fname = "G:\\data\\AmABP\\Lx2400\\ini_ordered\\AmABP_Lx2400_Ly200_p0.55_v-180_C12_Dr3.bin"

    get_one_frame(fname, 422135144)

    # os.chdir(r"D:\data\AmABP\cell")
    # file_in = "Cell_Lx400_Ly100_p0.4_a1_b1_Dr0.1.bin"
    # get_one_image_frame(file_in, 2378000)
