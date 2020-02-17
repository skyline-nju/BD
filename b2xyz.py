"""
Transfer .bin file to .extxyz file.
2020/2/8
"""
import decode
import numpy as np


def save_as_extxyz(file_in):
    file_out = file_in.replace(".bin", ".extxyz")
    fout = open(file_out, "w")
    para_dict = decode.get_para_from_file(file_in)
    N = para_dict["N"]
    Lx = para_dict["Lx"]
    Ly = para_dict["Ly"]
    comment_line = "Lattice=\"%g 0 0 0 %g 0 0 0 1\" " % (Lx, Ly)
    if "theta" in para_dict["data"]:
        comment_line += "Properties=species:S:1:pos:R:2:mass:M:1 "
        frames = decode.read_pos_theta(file_in, return_t=True)
        for i_frame, frame in enumerate(frames):
            t, x, y, theta = frame
            theta += np.pi
            mask = theta > np.pi
            theta[mask] -= 2 * np.pi
            fout.write("%d\n%s\n" % (N, comment_line + "Time=%d" % t))
            lines = [
                "N\t%f\t%f\t%f\n" % (x[i], y[i], -theta[i])
                for i in range(x.size)
            ]
            fout.writelines(lines)
            print("frame", i_frame, end="\r")
    else:
        comment_line += "Properties=species:S:1:pos:R:2 "
        frames = decode.read_pos(file_in, return_t=True)
        for i_frame, frame in enumerate(frames):
            t, x, y = frame
            fout.write("%d\n%s\n" % (N, comment_line + "Time=%d" % t))
            lines = [
                "N\t%f\t%f\n" % (x[i], y[i])
                for i in range(x.size)
            ]
            fout.writelines(lines)
            print("frame", "th frame")
    fout.close()


if __name__ == "__main__":
    # fname = "D:\\data\\ABP_test\\Ly150\\AmABP_Lx2100_Ly150_p0.55_v-50_C6_Dr0.8.bin"
    fname = "D:\\code\\BD\\BD2D_MPI_v2\\AmABP_Lx100_Ly80_p0.5_v0_C12_Dr3.bin"
    save_as_extxyz(fname)

    # import glob
    # path = 'D:\\data\\ABP_test\\Ly25'
    # # path = 'D:\\code\\BD\\BD2D_MPI_v2'
    # files = glob.glob("%s\\*.bin" % path)
    # for fname in files:
    #     save_as_extxyz(fname)
