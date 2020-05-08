"""
cal order parameters.
2020/4/16
"""

import decode
import numpy as np


def cal_polarity(file_in, beg, sep=10000):
    frames = decode.read_pos_theta(file_in, sep, beg)
    vm_sum = 0
    count = 0
    for frame in frames:
        t, x, y, theta = frame
        vx = np.sin(theta)
        vy = np.cos(theta)
        vm_sum += np.sqrt(np.mean(vx) ** 2 + np.mean(vy) ** 2)
        count += 1
    return vm_sum / count


def handle_files(folder, Dr):
    import glob
    import os
    files = glob.glob(folder + os.path.sep + "*Dr%g.bin" % Dr)
    v_list, C_list, p_list = [], [], []
    for f in files:
        try:
            p = cal_polarity(f, 3e6, 1e4)
            para_list = os.path.basename(f).rstrip(".extxyz").split("_")
            # phi = float(para_list[3].lstrip("p"))
            v = float(para_list[4].lstrip("v-"))
            C = float(para_list[5].lstrip("C"))
            v_list.append(v)
            C_list.append(C)
            p_list.append(p)
        except ZeroDivisionError:
            pass

    for i in range(len(v_list)):
        print(v_list[i], C_list[i], p_list[i])


if __name__ == "__main__":
    # folder = "D:\\data\\ABP_test\\phi04"
    folder = "D:\\data\\ABP_test\\phi04\\Rc2"
    handle_files(folder, 3)
