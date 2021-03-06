"""
Transfer .bin file to .extxyz file.
2020/2/8
"""
import decode
import numpy as np
import os


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
    print(para_dict)
    N = para_dict["N"]
    Lx = para_dict["Lx"]
    Lx2 = int(os.path.basename(file_in).split("_")[1].lstrip("Lx"))
    if Lx != Lx2:
        Lx = Lx2
        N *= 2
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
                    "N\t%.3f\t%.3f\nO\t%.3f\t%.3f\n" %
                    (x[i], y[i], x2[i], y2[i]) for i in range(x.size)
                ]
            fout.writelines(lines)
            print("frame", i_frame, end="\r")
    else:
        comment_line += "Properties=species:S:1:pos:R:2 "
        frames = decode.read_pos(file_in, sep, t_start)
        for i_frame, frame in enumerate(frames):
            t, x, y = frame
            fout.write("%d\n%s\n" % (N, comment_line + "Time=%d" % t))
            lines = ["N\t%.3ff\t%.3f\n" % (x[i], y[i]) for i in range(x.size)]
            fout.writelines(lines)
            print("frame", "th frame")
    fout.close()


def handle_files(folder, sep=10000):
    import glob
    import os
    files_in = glob.glob(folder + os.path.sep + "*.bin")
    files_fault = []
    for f in files_in:
        fout = f.replace(".bin", ".extxyz")
        if os.path.exists(fout):
            if os.stat(fout).st_mtime > os.stat(f).st_mtime:
                continue
            mode = "a"
        else:
            mode = "w"
        # try:
        #     save_as_extxyz(f, sep, mode)
        # except KeyError:
        #     files_fault.append(f)
        save_as_extxyz(f, sep, mode)

    print("The follwing files are failed:")
    for f in files_fault:
        print(f)


if __name__ == "__main__":
    folder = "D:\\data\\AmABP\\ini_ordered2\\Lx1200"
    # folder = "D:\\data\\AmABP\\ini_rand2"
    # folder = "G:\\data\\AmABP2\\ini_ordered\\Lx2400"

    handle_files(folder, sep=1)
    # fname = folder + "\\AmABP_Lx200_Ly100_p0.2_v-180_C12_Dr3.bin"
    # # n = decode.get_n_frames(fname)
    # # print("n = %d" % n)
    # frames = decode.read_pos_theta(fname, 1)
    # for i, frame in enumerate(frames):
    #     print("i =", i)
