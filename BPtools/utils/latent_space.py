import matplotlib.pyplot as plt
import numpy as np
import io
from matplotlib.offsetbox import AnnotationBbox
from sklearn import manifold
import torch
from PIL import Image
from matplotlib import cm



class myBhattacharyya():
    def __init__(self):
        pass

    def __call__(self, x1, x2):
        mu1, s1 = np.split(x1, 2)
        mu2, s2 = np.split(x2, 2)
        s = (s1 + s2) / 2

        return np.dot(mu1 - mu2, (mu1 - mu2) / s) / 8 + 0.5 * (np.sum(np.log(s)) - 0.5*(np.sum(np.log(s1)) +
                                                                                        np.sum(np.log(s2))))


def computeTSNEProjectionOfLatentSpace(X, X_encoded, perplex=30, components=2, display=False, metric="euclidean"):
    # Compute latent space representation
    # print("Computing latent space projection...")
    # X_encoded = encoder(X)

    # Compute t-SNE embedding of latent space
    print("Computing t-SNE embedding...")
    tsne = manifold.TSNE(n_components=components, init='pca', random_state=0, perplexity=perplex, metric=metric)
    X_tsne = tsne.fit_transform(X_encoded)

    # Plot images according to t-sne embedding
    if display:
        print("Plotting t-SNE visualization...")
        fig, ax = plt.subplots()
        imscatter(X_tsne[:, 0], X_tsne[:, 1], Data=X, ax=ax, zoom=0.6)
        plt.show()
    else:
        return X_tsne


def imscatter(x, y, ax, Data, zoom=None):
    images = []
    for i in range(len(x)):
        x0, y0 = x[i], y[i]
        # Convert to image
        imageSize = 0
        # img = Data[i] * 255.
        # img = img.astype(np.uint8).reshape([28, 28])  # nemkell
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # Note: OpenCV uses BGR and plt uses RGB
        # image = OffsetImage(img, zoom=zoom)  # nemkell
        im = Image.fromarray(np.uint8(cm.gist_earth(Data[i]) * 255))
        image = plt.figure()
        plt.plot(im)
        # canvas = FigureCanvas(image)
        # canvas.draw()
        # X = fig2data(image)
        # grab the pixel buffer and dump it into a numpy array
        # X = np.array(canvas.renderer.buffer_rgba())

        ab = AnnotationBbox(image, (x0, y0), xycoords='data', frameon=False)
        images.append(ax.add_artist(ab))

    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()


def plot_latent_interpolation(mu1, mu2, decoder, n_inter):
    f, interpol = plt.subplots(n_inter + 2, 1, sharex='all', sharey='all')

    reconstruction = decoder(mu1).to("cpu").detach().numpy()
    img = Image.fromarray(np.uint8(cm.gist_earth(reconstruction) * 255))
    interpol[0].plot(img)

    for n in range(1, n_inter+1):
        mu_n = (n * mu2 + (n_inter - n + 1) * mu1) / (n_inter + 1)
        reconstruction_n = decoder(mu_n).to("cpu").detach().numpy()
        img_n = Image.fromarray(np.uint8(cm.gist_earth(reconstruction_n) * 255))
        interpol[n].plot(img)

    reconstruction = decoder(mu2).to("cpu").detach().numpy()
    img = Image.fromarray(np.uint8(cm.gist_earth(reconstruction) * 255))
    interpol[n_inter + 1].plot(img)
    plt.show()


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def sampler(mu, logvar):
    std = logvar.mul(0.5).exp_()
    eps = torch.FloatTensor(std.size()).normal_().to(std.device)
    return eps.mul(std).add_(mu), eps
