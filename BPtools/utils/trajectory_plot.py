import matplotlib.pyplot as plt
import numpy as np
import io


def traj_to_img(trajectory, label):
    x = [t[0] for t in trajectory]
    y = [t[1] for t in trajectory]

    args = [x, y]

    plt.figure()
    # y_target, x_target
    plt.plot(args[1], args[0], label=label)
    plt.title(label)
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return buf


def trajs_to_img(real, gen, label):
    x_real = [t[0] for t in real]
    y_real = [t[1] for t in real]

    x_gen = [t[0] for t in gen]
    y_gen = [t[1] for t in gen]

    args = [x_real, y_real, x_gen, y_gen]

    fig = plt.figure()
    # y_target, x_target
    plt.plot(args[1], args[0], label='Real')
    plt.plot(args[3], args[2], label='Decoded')

    plt.title(label)
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return buf