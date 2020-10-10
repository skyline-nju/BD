import numpy as np
import os
import struct
from gsd import hoomd
import glob


def read_bin(fin):
    with open(fin, "rb") as f:
        f.seek(0, 2)
        file_size = f.tell()
        f.seek(0, 0)
        buf = f.read(4)
        info_size, = struct.unpack("i", buf)
        buf = f.read(info_size)
        info = buf.decode()
        print(info)
        print("file size =", file_size)
        n_frame = 0
        while f.tell() < file_size:
            buf = f.read(4)
            info_size, = struct.unpack("i", buf)
            buf = f.read(info_size)
            info = buf.decode()
            # t = int(info.split("=")[1])
            buf = f.read(8)
            data_size, = struct.unpack("Q", buf)
            n_frame += 1
            buf = f.read(data_size)
            data = struct.unpack("%df" % (data_size // 4), buf)
            yield np.array(data).reshape(data_size // 12, 3)


def convert(fin, fout=None):
    basename = os.path.basename(fin)
    Lx = int(basename.split("_")[1].lstrip("Lx"))
    Ly = int(basename.split("_")[2].lstrip("Ly"))

    if fout is None:
        fout = fin.replace(".bin", ".gsd")
    traj = hoomd.open(name=fout, mode="wb")
    frames = read_bin(fin)
    for i, frame in enumerate(frames):
        s = hoomd.Snapshot()
        s.configuration.step = i
        s.particles.N = frame.shape[0]
        frame[:, 2] += np.pi
        mask = frame[:, 2] > np.pi
        frame[:, 2][mask] -= np.pi * 2
        s.particles.position = frame
        if i == 0:
            s.configuration.box = [Lx, Ly, 5, 0, 0, 0]
        traj.append(s)
    traj.close()


if __name__ == "__main__":
    # folder_in = "G:/data/AmABP/phase_diagram/phi0.55_200_100"
    # folder_out = "D:/data/AmABP/phase_diagram/phi0.55_200_100"
    # folder_in = "G:/data/AmABP/Lx600/ini_ordered"
    folder_in = "G:/data/AmABP/Lx2400/ini_rand"
    folder_out = "D:/data/AmABP2/tmp2"

    files = glob.glob("%s/*.bin" % folder_in)
    for fin in files:
        phi = float(os.path.basename(fin).split("_")[3].lstrip("p"))
        fout = "%s/%s" % (folder_out, os.path.basename(fin).replace(
            ".bin", ".gsd").replace("p%g" % phi, "p%.3f" % phi))
        if os.path.exists(
                fout) and os.stat(fout).st_mtime > os.stat(fin).st_mtime:
            continue
        try:
            convert(fin, fout)
        except UnicodeDecodeError:
            print("UnicodeDecodeError for", fin)
