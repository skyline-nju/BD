import numpy as np
import matplotlib.pyplot as plt
import os
import decode
import glob


def coarse_grain(x, y, theta, Lx, Ly, dx=1):
    inv_dx = 1. / dx
    nx = Lx // dx
    area = dx * Ly
    rho, px, py = np.zeros((3, nx))
    vx = -np.cos(theta)
    vy = -np.sin(theta)
    for j in range(x.size):
        k = int(x[j] * inv_dx)
        if k < 0:
            k = nx - 1
        elif k >= nx:
            k = 0
        rho[k] += 1
        px[k] += vx[j]
        py[k] += vy[j]
    return rho / area, px / area, py / area


def eval_space_time(fin, dx=1, frame_beg=0):
    para_dict = decode.get_para_from_file(fin)
    bins = para_dict["Lx"] // dx
    rho_x, px_x, py_x = np.zeros((3, bins))
    frames = decode.read_pos_theta(fin, sep=1, frame_start=frame_beg)
    for i, (t, x, y, theta) in enumerate(frames):
        rho, px, py = coarse_grain(x, y, theta, para_dict["Lx"],
                                   para_dict["Ly"], dx)
        if i == 0:
            rho_x, px_x, py_x = rho, px, py
        else:
            rho_x = np.append(rho_x, rho, axis=0)
            px_x = np.append(px_x, px, axis=0)
            py_x = np.append(py_x, py, axis=0)
    n = rho_x.size // bins
    rho_x = rho_x.reshape(n, bins)
    px_x = px_x.reshape(n, bins)
    py_x = py_x.reshape(n, bins)
    return rho_x, px_x, py_x


def plot_space_time_img(f_npz, tmax=None, dt=0.5, show=False):
    # para = decode.get_para_from_file(f_npz)
    npzfile = np.load(f_npz)
    rho_x = npzfile["rho_x"][:tmax]
    px_x = npzfile["px_x"][:tmax]
    print(rho_x.shape)
    if tmax is None:
        tmax = rho_x.shape[0]
        print("max t =", tmax)
    fig, (ax1, ax2) = plt.subplots(
        ncols=1,
        nrows=2,
        constrained_layout="true",
        sharex=True,
        figsize=(16, 6))
    extent = [0, tmax * dt, 1, 0]
    im1 = ax1.imshow(rho_x.T, aspect="auto", extent=extent)
    cb1 = plt.colorbar(im1, ax=ax1)
    cb1.set_label(r"$\rho$", fontsize="large")
    ux_x = px_x / rho_x
    im2 = ax2.imshow(
        ux_x.T, cmap="bwr", aspect="auto", extent=extent, vmin=-1, vmax=1)
    cb2 = plt.colorbar(im2, ax=ax2)
    cb2.set_label(r"$p_x$", fontsize="large")
    ax2.set_xlabel(r"$t/\tau$", fontsize="x-large")
    ax1.set_ylabel(r"$x/L_x$", fontsize="x-large")
    ax2.set_ylabel(r"$x/L_x$", fontsize="x-large")
    if show:
        plt.show()
    plt.savefig(f_npz.replace(".npz", ".png"))
    plt.close()
    npzfile.close()


def handle_files(folder, show=False):
    os.chdir(folder)
    if not os.path.exists("space_time"):
        os.mkdir("space_time")
    files_in = glob.glob("*.bin")
    for f in files_in:
        f_npz = "space_time\\" + f.replace(".bin", ".npz")
        if os.path.exists(f_npz):
            if os.stat(f_npz).st_mtime > os.stat(f).st_mtime:
                continue
            data = np.load(f_npz)
            rho_x, px_x, py_x = data["rho_x"], data["px_x"], data["py_x"]
            data.close()
            rho_x2, px_x2, py_x2 = eval_space_time(
                f, dx=4, frame_beg=rho_x.shape[0])
            rho_x = np.append(rho_x, rho_x2, axis=0)
            px_x = np.append(px_x, px_x2, axis=0)
            py_x = np.append(py_x, py_x2, axis=0)
        else:
            rho_x, px_x, py_x = eval_space_time(f, dx=4)
        np.savez_compressed(f_npz, rho_x=rho_x, px_x=px_x, py_x=py_x)
    files_npz = glob.glob("space_time\\*.npz")
    for f_npz in files_npz:
        plot_space_time_img(f_npz, tmax=None, dt=0.5, show=show)


def plot_profile(f_npz, beg):
    npzfile = np.load(f_npz)
    rho_x = npzfile["rho_x"][beg:] * np.pi / 4
    px_x = npzfile["px_x"][beg:]
    plt.ion()
    n = rho_x.shape[0]
    for i in range(n):
        plt.plot(rho_x[i])
        plt.ylim(0.1, 1.1)
        plt.title(r"$\phi=%.2f$" % np.mean(rho_x))
        plt.pause(0.1)
        plt.clf()
    plt.close()


if __name__ == "__main__":
    # ini_mode = "rand"
    # Lx = 3600
    # Ly = 100
    # phi = 0.2
    # os.chdir("D:\\data\\AmABP\\Lx%d\\ini_%s" % (Lx, ini_mode))
    # fname = "AmABP_Lx%d_Ly%d_p%g_v-180_C12_Dr3.bin" % (Lx, Ly, phi)
    # plot_space_time_img(fname, None)
    folder = r"G:\data\AmABP\Lx2400\ini_ordered"
    # folder = r"G:\data\AmABP\Lx3600\ini_rand"
    handle_files(folder, False)

    # f_npz = "G:\\data\\AmABP\\Lx2400\\ini_rand\\space_time\\AmABP_Lx2400_Ly100_p0.55_v-180_C12_Dr3.npz"

    # plot_profile(f_npz, 0)

