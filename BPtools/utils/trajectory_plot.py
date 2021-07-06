import matplotlib.pyplot as plt
import numpy as np
import io
from matplotlib.offsetbox import AnnotationBbox
from sklearn import manifold
from torch import ones, float


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


def trajs_to_img_2(label, **kwargs):
    XX = []
    YY = []
    fig = plt.figure()
    for key, value in kwargs.items():
        xx = [t[0] for t in value]
        yy = [t[1] for t in value]
        XX.append(xx)
        YY.append(yy)
        plt.plot(yy, xx, label=key)

    plt.title(label)
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return buf



def plot_target_pred(target, prediction, label=None, category=None):
    x_target = [t[0] for t in target]
    y_target = [t[1] for t in target]

    x_pred = [t[0] for t in prediction]
    y_pred = [t[1] for t in prediction]

    fig = plt.figure()
    ax1 = fig.add_axes((0.1, 0.2, 0.8, 0.7))
    y_loss = np.mean((np.array(y_target) - np.array(y_pred))**2)
    y_mean = np.mean(np.array(y_target) - np.array(y_pred))
    y_std = np.std(np.array(y_target) - np.array(y_pred))

    x_loss = np.mean((np.array(x_target) - np.array(x_pred)) ** 2)
    x_mean = np.mean(np.array(x_target) - np.array(x_pred))
    x_std = np.std(np.array(x_target) - np.array(x_pred))
    plt.plot(y_target, x_target, label="target", marker="+")
    # plt.gca().set_aspect("equal")
    plt.plot(y_pred, x_pred, label="decoded", marker="x")
    # ax1.set_xlabel("Y-coordinate")
    # ax1.set_ylabel("X-coordinate")
    txt = ''
    if label is not None:
        txt = '\n' + "Label: " + str(label) + "Category: " + str(category)

    ax1.set_title("Y loss: " + str(y_loss) + "; ErrMean(Y): " + str(y_mean)+"; ErrStd(Y): " + str(y_std) + "\n"
                  + "X loss: " + str(x_loss) + "; ErrMean(X): " + str(x_mean)+"; ErrStd(X): " + str(x_std) + txt)
    # print(x_pred)
    # print(y_pred)
    # print(x_target)
    # print(y_target)
    plt.legend()
    plt.show()
    del fig


def plot_trajectory_dist(preds, target, expected, w, mu):
    x_trg = np.array([t[0] for t in target])
    y_trg = np.array([t[1] for t in target])

    x_exp = np.array([t[0] for t in expected])
    y_exp = np.array([t[1] for t in expected])

    x_preds = []
    y_preds = []
    # fig = plt.figure()
    # plt.show()
    f, axarr = plt.subplots(2)
    # ax1 = f.add_axes((0.1, 0.2, 0.8, 0.7))
    f.tight_layout(pad=1.5)
    plt.subplots_adjust(top=0.90, wspace=3.5, left=0.105)

    y_loss = round(np.mean((np.array(y_trg) - np.array(y_exp))**2), 5)
    y_mean = round(np.mean(np.array(y_trg) - np.array(y_exp)), 5)
    y_std = round(np.std(np.array(y_trg) - np.array(y_exp)), 5)

    x_loss = round(np.mean((np.array(x_trg) - np.array(x_exp)) ** 2), 5)
    x_mean = round(np.mean(np.array(x_trg) - np.array(x_exp)), 5)
    x_std = round(np.std(np.array(x_trg) - np.array(x_exp)), 5)

    for pred, alpha in zip(preds, w):
        x_pred = [t[0] for t in pred]

        y_pred = [t[1] for t in pred]
        alpha = np.exp(-np.dot(alpha, alpha)/2) / 2
        color = "mediumslateblue"
        axarr[0].plot(y_pred, x_pred, color=color, alpha=alpha, linewidth=0.1)

        # x_preds.append(x_pred)
        # y_preds.append(y_pred)

    axarr[0].plot(y_trg, x_trg, label="target", color="red")
    axarr[0].plot(y_exp, x_exp, label="expected", color="mediumspringgreen", linewidth=0.6)  # "paleturquoise"

    x_min = np.min(x_trg) - 0.3
    x_max = np.max(x_trg) + 0.3
    x_0 = (x_min + x_max) / 2

    axarr[0].set_title("Y loss: " + str(y_loss) + "; ErrMean(Y): " + str(y_mean) + "; ErrStd(Y): " + str(y_std) + "\n"
                  + "X loss: " + str(x_loss) + "; ErrMean(X): " + str(x_mean) + "; ErrStd(X): " + str(x_std))

    axarr[0].legend()
    axarr[0].set_ylim(x_0 - 2.0, x_0 + 2.0)

    # x, y labels
    axarr[0].set_xlabel("Longitudinal coordinate: y (m)")
    axarr[0].set_ylabel("Lateral coordinate: x (m)")
    axarr[1].set_xlabel("Components of the context vector")
    axarr[1].set_ylabel("Values of the context vector")

    xs = np.arange(0, mu.shape[0], 1)
    axarr[1].bar(xs, mu, label='hidden')
    for x, y in zip(xs, mu):
        label = "{:.2f}".format(y)

        axarr[1].annotate(label, (x, y), textcoords="offset points", xytext=(0, 10 if y < 0 else -10), ha='center')

    axarr[1].autoscale()

    axarr[1].legend()

    return axarr
    # plt.show()
    # x_preds = np.array(x_preds)
    # y_preds = np.array(y_preds)
    # print(x_preds.shape, y_preds.shape)


# def load_files():
#     path = RESULTS_PATH
#
#
# def prepare(targ, preg):
#     return target, prediction

def boundary_for_grid(grid):
    # N, 1, 16, 128
    N, c, A, B = grid.shape
    bound = ones((N,c,A+2,B+2), dtype=float).to(grid.device) / 2.0
    bound[:,:,1:A+1,1:B+1] = grid[:,:,:,:]
    return bound
