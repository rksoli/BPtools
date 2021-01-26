from BPtools.utils.trajectory_plot import *
from BPtools.utils.latent_space import *
from BPtools.utils.models import EncoderBN, VarDecoderConv1d_3, Discriminator, AdvAE, VAEDataModul
import os
import torch

RESULTS_PATH = "d:/phd/konf_2021"
name_l = "aae_traj8_val"
c = 10
# load trainer   log_advvae_disc_lay8/
model_state_dict = torch.load('model_state_dict_advvae_disc_lay0')
enc = EncoderBN(2, 60, 10)
dec = VarDecoderConv1d_3(2, 60, 10)
disc = Discriminator(10, 0)

my_model = AdvAE(enc, dec, disc)
my_model.load_state_dict(model_state_dict)
my_model.to("cuda")

# load data
my_dm = VAEDataModul(path="c:/repos/full_data/X_Yfull_dataset.npy", split_ratio=0.1)
my_dm.prepare_data()
my_dm.setup()

target_t = my_dm.val_dataloader().to("cpu").detach().numpy()
target_t = np.transpose(target_t, (0, 2, 1))
_, mu_t, logvar_t, hidden = my_model(my_dm.val_dataloader())
expected_t = my_model.vae.dec(mu_t)
expected_t = np.transpose(expected_t.to("cpu").detach().numpy(), (0, 2, 1))

hidden_list = []
weight_list = []
pred_list = []
for i in range(1):
    z, w = sampler(mu_t, logvar_t)
    w = w.to("cpu").detach().numpy()
    s_pred = my_model.vae.dec(z)
    z = z.to("cpu").detach().numpy()
    hidden_list.append(z)
    weight_list.append(w)
    s_pred = s_pred.to("cpu").detach().numpy()
    s_pred = np.transpose(s_pred, (0, 2, 1))
    pred_list.append(s_pred)

mu_t = mu_t.to("cpu").detach().numpy()
pred_list = np.transpose(np.array(pred_list), (1, 0, 2, 3))
weight_list = np.transpose(np.array(weight_list), (1, 0, 2))
print(pred_list.shape)
print(weight_list.shape)
i = 0
for traj, trg, exp, w, m in zip(pred_list, target_t, expected_t, weight_list, mu_t):

    if i in [5, 9, 39]:
        axarray = plot_trajectory_dist(traj, trg, exp, w, m)
        if not os.path.exists(RESULTS_PATH + "/" + name_l + "/"):
            os.makedirs(RESULTS_PATH + "/" + name_l + "/")
        # plt.show()
        plt.savefig(
            RESULTS_PATH + "/" + name_l + "/" + "context" + str(c) + "fig_" + str(i) + ".pdf")
        del axarray
    i += 1
