import numpy as np
import matplotlib.pyplot as plt
import os
import decode
import space_time


if __name__ == "__main__":
    # os.chdir("D:\\data\\AmABP\\phase_digram\\C12_200_100")
    os.chdir("D:\\data\\AmABP\\Lx4800\\ini_rand")

    Lx = 2400
    Ly = 200
    phi = 0.2
    v = -180
    C = 12
    fname = "AmABP_Lx%d_Ly%d_p%g_v%d_C%d_Dr3.bin" % (Lx, Ly, phi, v, C)
    # print(decode.get_n_frames(fname))
    frames = decode.read_pos_theta(fname, 1, 400*1e5)
    dx = 2
    for frame in frames:
        t, x, y, theta = frame
        rho, px, py = space_time.coarse_grain(x, y, theta, Lx, Ly, dx)
        plt.plot(rho)
        plt.show()
        plt.close()

