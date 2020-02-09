"""
Decode the .bin file where the data are saved as binary format.
2020/2/8
"""
import numpy as np
import struct
import matplotlib.pyplot as plt


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


def get_para_from_file(file_in, beg=1):
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


def read_pos(fin, show_frame_info=False, return_t=False):
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
            if show_frame_info:
                print(info)
            buf = f.read(8)
            data_size, = struct.unpack("Q", buf)
            buf = f.read(data_size)
            data = struct.unpack("%df" % (data_size // 4), buf)
            x, y = np.array(data).reshape(para_dict["N"], 2).T
            if not return_t:
                yield x, y
            else:
                t = get_t(info)
                yield t, x, y


def read_pos_theta(fin, show_frame_info=False, return_t=False):
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
            if show_frame_info:
                print(info)
            buf = f.read(8)
            data_size, = struct.unpack("Q", buf)
            buf = f.read(data_size)
            data = struct.unpack("%df" % (data_size // 4), buf)
            x, y, theta = np.array(data).reshape(para_dict["N"], 3).T
            if not return_t:
                yield x, y, theta
            else:
                t = get_t(info)
                yield t, x, y, theta


if __name__ == "__main__":
    # fname = "AmABP_Lx40_Ly20_p0.7_v0_C12.bin"
    fname = "D:\\data\\ABP_test\\PBC_MPI\\open_mpi\\AmABP_Lx150_Ly75_p0.55_v-180_C12.bin"
    # show_file_info(fname)
    frames = read_pos_theta(fname, True)
    for x, y, theta in frames:
        plt.plot(x, y, ".")
        plt.show()
        plt.close()
