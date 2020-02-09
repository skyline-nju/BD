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
        for frame in frames:
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
    else:
        comment_line += "Properties=species:S:1:pos:R:2 "
        frames = decode.read_pos(file_in, return_t=True)
        for frame in frames:
            t, x, y = frame
            fout.write("%d\n%s\n" % (N, comment_line + "Time=%d" % t))
            lines = [
                "N\t%f\t%f\n" % (x[i], y[i])
                for i in range(x.size)
            ]
            fout.writelines(lines)
    fout.close()


if __name__ == "__main__":
    fname = "D:\\data\\ABP_test\\PBC_MPI\\AmABP_Lx1050_Ly50_p0.55_v-180_C12.bin"
    save_as_extxyz(fname)
