from sklearn import svm
from BPtools.utils.trajectory_plot import *
from BPtools.utils.latent_space import *
from BPtools.utils.models import EncoderBN, VarDecoderConv1d_3, Discriminator, AdvAE, VAEDataModul

import torch
from mpl_toolkits.mplot3d import Axes3D


# load trainer
# model_state_dict = torch.load('logadvvae_disc_lay1/advae_model_state_dict_proba')
model_state_dict = torch.load('log_WITHOUTLOGVARadvvae_n_disc_lay2/model_state_dict_WITHOUTLOGVARadvvae_n_disc_lay2')
# model_state_dict = torch.load('log_WITH3GAUSS2advvae_n_disc_lay2/model_state_dict_WITH3GAUSS2advvae_n_disc_lay2')
# model_state_dict = torch.load('log_advvae_n_disc_lay2/model_state_dict_advvae_n_disc_lay2')
print(model_state_dict.keys())
enc = EncoderBN(2, 60, 10)
dec = VarDecoderConv1d_3(2, 60, 10)
disc = Discriminator(10, 2)

my_model = AdvAE(enc, dec, disc)
print(my_model.state_dict().keys())
my_model.load_state_dict(model_state_dict)
my_model.to("cuda")

# load data
my_dm = VAEDataModul(path="c:/repos/full_data/X_Yfull_dataset.npy", split_ratio=0.1)
my_dm.prepare_data()
my_dm.setup()

# load labels
labels = np.load("c:/repos/full_data/X_Yfull_labels.npy")
V = labels.shape[0]
# train
labels_tr = np.reshape(labels[0:int((1 - 2 * 0.1) * V)], (-1, 3))
labels_tr = np.argmax(labels_tr, axis=1) - 1.0
# test
labels_t = np.reshape(labels[int((1 - 2 * 0.1) * V):int((1 - 0.1) * V)], (-1, 3))
labels_t = np.argmax(labels_t, axis=1) - 1.0
# valid
labels_v = np.reshape(labels[int((1 - 0.1) * V):], (-1, 3))
labels_v = np.argmax(labels_v, axis=1) - 1.0

# eval the model
# train
pred, mu, logvar, hidden = my_model(my_dm.train_dataloader())
mu = mu.to("cpu").detach().numpy()
std = logvar.mul(0.5).exp_().to("cpu").detach().numpy()
pred = pred.to("cpu").detach().numpy()
target = my_dm.train_dataloader().to("cpu").detach().numpy()
# test
_, mu_t, _, _ = my_model(my_dm.test_dataloader())
mu_t = mu_t.to("cpu").detach().numpy()
# valid
_, mu_v, _, _ = my_model(my_dm.val_dataloader())
mu_v = mu_v.to("cpu").detach().numpy()

# fit SVM
C = 1
for C in [0.1,0.1,1,10,20,30,40]:
    clf = svm.SVC(C=C, decision_function_shape='ovo')
    clf.fit(mu, labels_tr)
    print("fit is done")
    predicted_labels = clf.predict(mu_v)
    print("prediction is done")
    good = 0
    bad = 0
    for p,t in zip(predicted_labels, labels_v[1:324]):
        if p == t:
            good += 1
        else:
            bad += 1
    result = good / (good + bad)
    print("SVC: ", "C= ", C, "   ", good / (good + bad))
#
# # make masks
# mask_left = labels_tr == -1.0
# mask_keep = labels_tr == 0.0
# mask_right = labels_tr == 1.0
# perplexity = 10
#
# # fit t-SNE
# comp = 2
# for perplexity in range(15, 25, 2):
#     embedded = computeTSNEProjectionOfLatentSpace(pred, mu, perplex=perplexity, components=comp)
#     if comp == 2:
#         fig = plt.figure()
#         plt.scatter(embedded[:, 0][mask_left], embedded[:, 1][mask_left],  color="red",
#                    label="left", s=2)
#         plt.scatter(embedded[:, 0][mask_keep], embedded[:, 1][mask_keep],
#                    color="green", label="keep", s=2)
#         plt.scatter(embedded[:, 0][mask_right], embedded[:, 1][mask_right],
#                    color="orange", label="right", s=2)
#         plt.legend()
#         plt.show()
#
#     if comp == 3:
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
#         ax.scatter(embedded[:, 0][mask_left], embedded[:, 1][mask_left], embedded[:, 2][mask_left], color="red",
#                    label="left", s=2)
#         ax.scatter(embedded[:, 0][mask_keep], embedded[:, 1][mask_keep], embedded[:, 2][mask_keep], color="green",
#                    label="keep", s=2)
#         ax.scatter(embedded[:, 0][mask_right], embedded[:, 1][mask_right], embedded[:, 2][mask_right], color="orange",
#                    label="right", s=2)
#         ax.set_xlabel('X feature')
#         ax.set_ylabel('Y feature')
#         ax.set_zlabel('Z feature')
#         plt.legend()
#         plt.show()
#
#
#
