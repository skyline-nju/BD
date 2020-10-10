import os
from gsd import hoomd
import glob
import numpy as np


def convert(fin):
    if not os.path.exists("tmp"):
        os.mkdir("tmp")
    fout = "tmp/%s" % (fin)
    traj_old = hoomd.open(name=fin, mode="rb")
    print(len(traj_old))

    traj_new = hoomd.open(name=fout, mode="wb")
    for i, f in enumerate(traj_old):
        s = hoomd.Snapshot()
        s.configuration.step = i
        s.particles.N = f.particles.N
        s.particles.position = f.particles.position
        if i == 0:
            s.configuration.box = f.configuration.box
        traj_new.append(s)
    traj_new.close()
    traj_old.close()


def get_one_frame(fin, iframe=-1):
    if not os.path.exists("last"):
        os.mkdir("last")
    fout = "last/%s" % fin
    traj_old = hoomd.open(name=fin, mode="rb")
    nframes = len(traj_old)
    if iframe >= nframes:
        print("iframe=%d should be less than nframes=%d" % (iframe, nframes))
    print("nframes =", nframes)
    traj_new = hoomd.open(name=fout, mode="wb")
    frame = traj_old[iframe]
    traj_new.append(frame)
    traj_new.close()
    traj_old.close()


def double_snap(fin, ori="x"):
    s = os.path.basename(fin).rstrip(".gsd").split("_")
    Lx = int(s[1].lstrip("Lx"))
    Ly = int(s[2].lstrip("Ly"))
    phi = float(s[3].lstrip("p"))
    v = float(s[4].lstrip("v"))
    C = float(s[5].lstrip("C"))
    Dr = float(s[6].lstrip("Dr"))
    if ori == "x":
        Lx *= 2
    elif ori == "y":
        Ly *= 2
    fout = "AmABP_Lx%d_Ly%d_p%.3f_v%g_C%g_Dr%g.gsd" % (Lx, Ly, phi, v, C, Dr)
    traj_old = hoomd.open(name=fin, mode="rb")
    frame_old = traj_old[0].particles.position
    traj_old.close()
    N_old = frame_old.shape[0]
    N_new = N_old * 2

    s = hoomd.Snapshot()
    s.configuration.step = 0
    s.particles.N = N_new
    s.configuration.box = [Lx, Ly, 5, 0, 0, 0]
    frame_new = np.zeros((N_new, 3))
    frame_new[:N_old] = frame_old
    frame_new[N_old:] = frame_old
    if ori == "x":
        frame_new[N_old:, 0] += Lx // 2
    elif ori == "y":
        frame_new[N_old:, 1] += Ly // 2
    s.particles.position = frame_new
    traj_new = hoomd.open(name=fout, mode="wb")
    traj_new.append(s)
    traj_new.close()


def double_snap_1more(fin, ori="x"):
    s = os.path.basename(fin).rstrip(".gsd").split("_")
    Lx = int(s[1].lstrip("Lx"))
    Ly = int(s[2].lstrip("Ly"))
    phi = float(s[3].lstrip("p"))
    v = float(s[4].lstrip("v"))
    C = float(s[5].lstrip("C"))
    Dr = float(s[6].lstrip("Dr"))
    if ori == "x":
        Lx *= 2
    elif ori == "y":
        Ly *= 2
    fout = "AmABP_Lx%d_Ly%d_p%.3f_v%g_C%g_Dr%g.gsd" % (Lx, Ly, phi, v, C, Dr)
    traj_old = hoomd.open(name=fin, mode="rb")
    frame_old = traj_old[0].particles.position
    traj_old.close()
    N_old = frame_old.shape[0]
    N_new = N_old * 2 + 1

    s = hoomd.Snapshot()
    s.configuration.step = 0
    s.particles.N = N_new
    s.configuration.box = [Lx, Ly, 5, 0, 0, 0]
    frame_new = np.zeros((N_new, 3))
    frame_new[:N_old] = frame_old
    frame_new[N_old:N_new-1] = frame_old
    frame_new[-1] = np.array([922.5, 54.5, 0.])
    if ori == "x":
        frame_new[N_old:N_new-1, 0] += Lx // 2
    elif ori == "y":
        frame_new[N_old:N_new-1, 1] += Ly // 2
    s.particles.position = frame_new
    traj_new = hoomd.open(name=fout, mode="wb")
    traj_new.append(s)
    traj_new.close()


if __name__ == "__main__":
    # # dest_dir = r"D:/data/AmABP2/tmp2"
    # dest_dir = r"G:/data/AmABP2/ini_ordered/Lx=1200"
    # os.chdir(dest_dir)
    # files = glob.glob("AmABP_Lx1200_Ly100_p0.550_v-180_C12_Dr3.gsd")
    # for fin in files:
    #     # convert(fin)
    #     get_one_frame(fin, -30)

    dest_dir = r"G:/data/AmABP2/ini_ordered/Lx=1200/last"
    os.chdir(dest_dir)
    fin = "AmABP_Lx1200_Ly100_p0.550_v-180_C12_Dr3.gsd"
    double_snap_1more(fin, "y")
