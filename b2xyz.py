"""
Transfer .bin file to .extxyz file.
2020/2/8
"""
import decode
import numpy as np


def get_last_t(file_in):
    f = open(file_in, "r")
    line = f.readline()
    while line != "":
        n = int(line.rstrip("\n"))
        t = int(f.readline().rstrip("\n").split("Time=")[1])
        for i in range(n):
            f.readline()
        print("reading %s: N=%d, t=%d" % (file_in, n, t), end="\r")
        line = f.readline()
    f.close()
    return t


def save_as_extxyz(file_in, sep, mod="w", Janus=False):
    if not Janus:
        file_out = file_in.replace(".bin", ".extxyz")
    else:
        file_out = file_in.rstrip(".bin") + "_Janus.extxyz"
    if mod == "a":
        t_start = get_last_t(file_out) + sep
        fout = open(file_out, "a")
    else:
        t_start = 0
        fout = open(file_out, "w")
    para_dict = decode.get_para_from_file(file_in)
    N = para_dict["N"]
    Lx = para_dict["Lx"]
    Ly = para_dict["Ly"]
    comment_line = "Lattice=\"%g 0 0 0 %g 0 0 0 1\" " % (Lx, Ly)
    if "theta" in para_dict["data"]:
        comment_line += "Properties=species:S:1:pos:R:2:mass:M:1 "
        frames = decode.read_pos_theta(file_in, sep, t_start)
        for i_frame, frame in enumerate(frames):
            t, x, y, theta = frame
            if not Janus:
                theta += np.pi
                mask = theta > np.pi
                theta[mask] -= 2 * np.pi
                fout.write("%d\n%s\n" % (N, comment_line + "Time=%d" % t))
                lines = [
                    "N\t%.3f\t%.3f\t%.3f\n" % (x[i], y[i], theta[i])
                    for i in range(x.size)
                ]
            else:
                x2 = x + 0.01 * np.cos(theta)
                y2 = y + 0.01 * np.sin(theta)
                fout.write("%d\n%s\n" % (N * 2, comment_line + "Time=%d" % t))
                lines = [
                    "N\t%.3f\t%.3f\nO\t%.3f\t%.3f\n" % (x[i], y[i], x2[i], y2[i])
                    for i in range(x.size)
                ]
            fout.writelines(lines)
            print("frame", i_frame, end="\r")
    else:
        comment_line += "Properties=species:S:1:pos:R:2 "
        frames = decode.read_pos(file_in, sep, t_start)
        for i_frame, frame in enumerate(frames):
            t, x, y = frame
            fout.write("%d\n%s\n" % (N, comment_line + "Time=%d" % t))
            lines = [
                "N\t%.3ff\t%.3f\n" % (x[i], y[i])
                for i in range(x.size)
            ]
            fout.writelines(lines)
            print("frame", "th frame")
    fout.close()


if __name__ == "__main__":
    # fname = "D:\\data\\ABP_test\\Ly150\\AmABP_Lx2100_Ly150_p0.2_v-50_C6_Dr0.8_t51120000.bin"
    # fname = "D:\\data\\ABP_test\\Ly150\\AmABP_Lx2100_Ly150_p0.15_v-50_C6_Dr0.8.bin"
    # fname = "D:\\data\\ABP_test\\Ly150_ini_ordered\\AmABP_Lx2100_Ly150_p0.4_v-50_C6_Dr0.8.bin"
    # fname = "D:\\data\\ABP_test\\Ly300\\AmABP_Lx2100_Ly300_p0.4_v-50_C6_Dr0.8.bin"
    # fname = "D:\\code\\BD\\BD2D_MPI_v2\\AmABP_Lx100_Ly80_p0.5_v0_C12_Dr3.bin"
    # fname = "D:\\data\\ABP_test\\Ly600\\AmABP_Lx1050_Ly600_p0.5_v-50_C6_Dr0.8.bin"
    # fname = "D:\\data\\ABP_test\\Ly600\\AmABP_Lx1050_Ly1050_p0.2_v-50_C6_Dr0.8.bin"

    # fname = "BD2D_MPI_v2\\AmABP_Lx50_Ly150_p0.01_v-50_C6_Dr0.6.bin"
    fname = "D:\\data\\ABP_test\\Ly150_slow\\AmABP_Lx2100_Ly150_p0.2_v-50_C6_Dr0.8_t51120000.bin"
    save_as_extxyz(fname, 4000, "w", True)

    # import glob
    # path = 'G:\\data\\AmABP\\200_100'
    # # path = 'D:\\code\\BD\\BD2D_MPI_v2'
    # files = glob.glob("%s\\*.bin" % path)
    # for fname in files:
    #     save_as_extxyz(fname, 10000)

    # fname = "D:\\data\\ABP_test\\Ly600\\AmABP_Lx1050_Ly1050_p0.2_v-50_C6_Dr0.8.extxyz"
    # get_last_t(fname)