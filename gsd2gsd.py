import os
from gsd import hoomd
import glob


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


if __name__ == "__main__":
    os.chdir(r"D:/data/AmABP2/ini_ordered//tmp")
    files = glob.glob("*.gsd")
    for fin in files:
        # convert(fin)
        get_one_frame(fin)
