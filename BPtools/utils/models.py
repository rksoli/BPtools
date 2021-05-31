from BPtools.core.bpmodule import *
from BPtools.core.bpdatamodule import *
from BPtools.utils.trajectory_plot import *
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
# from torch import functional as F


class VariationalAutoEncoder(BPModule):
    def __init__(self, encoder, decoder):
        super(VariationalAutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def sampler(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(std.device)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        # if self.training:
        #     err = torch.FloatTensor(x.size()).normal_(torch.tensor(0.0), torch.tensor(0.1)).to(x.device)
        #     x = x + err
        mu, logvar = self.encoder(x)
        z = self.sampler(mu, logvar)
        pred = self.decoder(z)  # return h
        return {"output": pred, "mu": mu, "logvar": logvar}, z

    def training_step(self, optim_configuration, step):
        self.train()
        self.optimizer_zero_grad(0, 0, optim_configuration, 0)
        kwargs, z = self(self.trainer.dataloaders["train"])
        loss = self.trainer.criterion(**kwargs, target=self.trainer.dataloaders["train"])
        loss.backward()
        optim_configuration.step()
        self.trainer.losses["train"].append(loss.item())
        # self.trainer.writer.add_scalar('train loss', loss.item(), step)

    def validation_step(self, step):
        self.eval()
        self.freeze()
        kwargs, z = self(self.trainer.dataloaders["valid"])
        loss = self.trainer.criterion(**kwargs, target=self.trainer.dataloaders["valid"])
        self.trainer.losses["valid"].append(loss.item())

        # Images
        picture_indexes = [4, 8, 38]
        img_batch = np.zeros((3, 3, 480, 640))
        i = 0
        if step % 100 == 0:
            for n in picture_indexes:
                real = np.transpose(np.array(self.trainer.dataloaders["valid"].to('cpu')), (0, 2, 1))[n]
                out = np.transpose(np.array(kwargs['output'].to('cpu')), (0, 2, 1))[n]

                img_real_gen = trajs_to_img(real, out, "Real and generated. N= " + str(n))
                img_real_gen = PIL.Image.open(img_real_gen)
                img_real_gen = ToTensor()(img_real_gen)
                img_batch[i] = img_real_gen[0:3]
                i = i + 1
            self.trainer.writer.add_images("Real & Out", img_batch, step)

        self.unfreeze()
        # self.trainer.writer.add_scalar('valid loss', loss.item(), step)

    def test_step(self, *args, **kwargs):
        pass

    def configure_optimizers(self):
        return optim.Adam(self.parameters())

    def prepare_data(self, **kwargs):
        self.setup()

    def setup(self, path="data/X_Yfull_dataset.npy"):
        data = np.load(path)
        seq_length = data.shape[3]
        feature_dim = data.shape[2]
        q = 0.1
        data = np.reshape(data, (-1, feature_dim, seq_length))
        V = data.shape[0]

        self.trainer.dataloaders["train"] = np.reshape(data[0:int((1 - 2 * q) * V)], (-1, feature_dim, seq_length))
        self.trainer.dataloaders["test"] = np.reshape(data[int((1 - 2 * q) * V):int((1 - q) * V)],
                                    (-1, feature_dim, seq_length))
        self.trainer.dataloaders["valid"] = np.reshape(data[int((1 - q) * V):], (-1, feature_dim, seq_length))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.trainer.dataloaders["train"] = torch.tensor(self.trainer.dataloaders["train"]).float().to(device)
        self.trainer.dataloaders["test"] = torch.tensor(self.trainer.dataloaders["test"]).float().to(device)
        self.trainer.dataloaders["valid"] = torch.tensor(self.trainer.dataloaders["valid"]).float().to(device)

        self.trainer.dataloaders["train"] = self.trainer.dataloaders["train"] - self.trainer.dataloaders["train"][:, :, 0][:, :, None]
        self.trainer.dataloaders["test"] = self.trainer.dataloaders["test"] - self.trainer.dataloaders["test"][:, :, 0][:, :, None]
        self.trainer.dataloaders["valid"] = self.trainer.dataloaders["valid"] - self.trainer.dataloaders["valid"][:, :, 0][:, :, None]


class VarEncoderConv1d(nn.Module):
    def __init__(self, input_channels, seq_length, context_dim):
        super(VarEncoderConv1d, self).__init__()
        self.input_dim = input_channels * seq_length
        self.context_dim = context_dim
        self.precontext_dim = (25 + context_dim)//2 - 2
        self.encoder_common = nn.Sequential(
            ##### 1
            nn.Conv1d(in_channels=input_channels, out_channels=6, kernel_size=4),  # 57 * 6
            nn.PReLU(6),
            ##### 2
            nn.Conv1d(in_channels=6, out_channels=12, kernel_size=6),  # 52 * 12
            nn.PReLU(12),
            ##### 3
            nn.Conv1d(in_channels=12, out_channels=8, kernel_size=8),  # 45 * 8
            nn.PReLU(8),
            nn.MaxPool1d(kernel_size=3, padding=0, dilation=1),  # 15 * 8
            # A kimenet 8 hosszú, 7 csatornás jel, 56 dimenziós
        )
        self.conv_mu = nn.Sequential(
            nn.Conv1d(in_channels=8, out_channels=5, kernel_size=5),  # 11 * 5
            nn.PReLU(5),
            nn.MaxPool1d(kernel_size=2, padding=0, dilation=1),  # 5 * 5 = 25
        )
        self.conv_logvar = nn.Sequential(
            nn.Conv1d(in_channels=8, out_channels=5, kernel_size=5),  # 11 * 5
            nn.PReLU(5),
            nn.MaxPool1d(kernel_size=2, padding=0, dilation=1),  # 5 * 5 = 25
        )
        self.linearpremu = nn.Linear(25, self.precontext_dim)
        self.linearprelogvar = nn.Linear(25, self.precontext_dim)
        self.prelu_mu = nn.PReLU(1)
        self.prelu_logvar = nn.PReLU(1)
        self.linear_mu = nn.Linear(self.precontext_dim, self.context_dim)
        self.linear_logvar = nn.Linear(self.precontext_dim, self.context_dim)

    def encoder_mu(self, h1):
        mu = self.conv_mu(h1).view(-1, 25)
        mu = self.prelu_mu(self.linearpremu(mu))
        # mu = self.linearpremu(mu)
        return self.linear_mu(mu)

    def encoder_logvar(self, h1):
        logvar = self.conv_logvar(h1).view(-1, 25)
        logvar = self.prelu_logvar(self.linearprelogvar(logvar))
        return self.linear_logvar(logvar)

    def forward(self, x):
        h1 = self.encoder_common(x)
        return self.encoder_mu(h1), self.encoder_logvar(h1)


class VarDecoderConv1d_3(nn.Module):
    def __init__(self, output_channels, seq_length, context_dim):
        super(VarDecoderConv1d_3, self).__init__()
        self.tnumber = lambda L_in, padd, kern, strid, dil, opadd: strid*(L_in-1)-2*padd+dil*(kern-1)+opadd+1
        self.seq_length = seq_length
        self.output_channels = output_channels
        self.input_dim = output_channels * seq_length
        self.context_dim = context_dim
        self.linear = nn.Linear(context_dim, context_dim)  # a context dimenzióhoz hozzá kell adni 3-at, ha caterogical
        #  decoder1 requires a vector of 10 dimensions
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=1, out_channels=3, padding=2, kernel_size=3, stride=1, dilation=1,
                               output_padding=0),  #
            nn.PReLU(3),
            nn.AdaptiveAvgPool1d(10),
            nn.ConvTranspose1d(in_channels=3, out_channels=8, padding=2, kernel_size=5, stride=2, dilation=1,
                               output_padding=0),  #
            nn.PReLU(8),
            nn.AdaptiveAvgPool1d(15),
            nn.ConvTranspose1d(in_channels=8, out_channels=4, padding=2, kernel_size=5, stride=2, dilation=1,
                               output_padding=0),  #
            nn.PReLU(4),
            nn.AdaptiveAvgPool1d(30),
            nn.ConvTranspose1d(in_channels=4, out_channels=self.output_channels, padding=0, kernel_size=5, stride=1,
                               dilation=1, output_padding=0),  # 2 * 60
            nn.AdaptiveAvgPool1d(self.seq_length)
        #    nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder1(self.linear(x.unsqueeze(1)))


class VAEDataModul(BPDataModule):
    def __init__(self, path, split_ratio):
        super(VAEDataModul).__init__()
        self.path = path
        self.seq_length = None
        self.feature_dim = None
        self.data = None
        self.split_ratio = split_ratio
        self.ngsim_train = None
        self.ngsim_test = None
        self.ngsim_val = None

    def prepare_data(self, *args, **kwargs):
        data = np.load(self.path)
        self.seq_length = data.shape[3]
        self.feature_dim = data.shape[2]
        q = 0.1
        self.data = np.reshape(data, (-1, self.feature_dim, self.seq_length))
        self.set_has_prepared_data(True)

    def setup(self, stage: Optional[str] = None):
        V = self.data.shape[0]
        feature_dim = self.feature_dim
        seq_length = self.seq_length
        q = self.split_ratio

        # todo
        self.data[:, 1, :] = 0.05 * self.data[:, 1, :]

        self.ngsim_train = np.reshape(self.data[0:int((1 - 2 * q) * V)], (-1, feature_dim, seq_length))
        self.ngsim_test = np.reshape(self.data[int((1 - 2 * q) * V):int((1 - q) * V)],
                                                      (-1, feature_dim, seq_length))
        self.ngsim_val = np.reshape(self.data[int((1 - q) * V):], (-1, feature_dim, seq_length))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.ngsim_train = torch.tensor(self.ngsim_train).float().to(device)
        self.ngsim_test = torch.tensor(self.ngsim_test).float().to(device)
        self.ngsim_val = torch.tensor(self.ngsim_val).float().to(device)

        self.ngsim_train = self.ngsim_train - self.ngsim_train[:, :, 0][:, :, None]
        self.ngsim_test = self.ngsim_test - self.ngsim_test[:, :, 0][:, :, None]
        self.ngsim_val = self.ngsim_val - self.ngsim_val[:, :, 0][:, :, None]
        self.set_has_setup_test(True)
        self.set_has_setup_fit(True)

    def train_dataloader(self, *args, **kwargs):
        # return DataLoader(self.ngsim_train, batch_size=self.ngsim_train.shape[0])
        return self.ngsim_train

    def val_dataloader(self, *args, **kwargs):
        return self.ngsim_val

    def test_dataloader(self, *args, **kwargs):
        return self.ngsim_test


class Discriminator(nn.Module):
    def __init__(self, hidden, lay_num=1):
        super(Discriminator, self).__init__()
        self.hidden = hidden
        self.mod_list = nn.ModuleList()
        for n in range(lay_num):
            self.mod_list.append(nn.Linear(self.hidden, self.hidden))
            self.mod_list.append(nn.BatchNorm1d(self.hidden))
            self.mod_list.append(nn.LeakyReLU(0.2))
        self.disc = nn.Sequential(
            # 2
            nn.Linear(self.hidden, self.hidden // 2),
            nn.BatchNorm1d(self.hidden // 2),
            nn.LeakyReLU(0.2),
            # 3
            nn.Linear(self.hidden//2, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        for layer in self.mod_list:
            z = layer(z)
        return self.disc(z)


class Discriminator2(nn.Module):
    def __init__(self, hidden, hidden2, lay_num=1):
        super(Discriminator2, self).__init__()
        self.hidden = hidden
        self.hidden2 = hidden2
        self.mod_list = nn.ModuleList()
        for n in range(lay_num):
            self.mod_list.append(nn.Linear(self.hidden, self.hidden2))
            self.mod_list.append(nn.BatchNorm1d(self.hidden2))
            self.mod_list.append(nn.LeakyReLU(0.2))
        self.disc = nn.Sequential(
            # 2
            nn.Linear(self.hidden2, self.hidden),
            nn.BatchNorm1d(self.hidden),
            nn.LeakyReLU(0.2),
            # 3
            nn.Linear(self.hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        for layer in self.mod_list:
            z = layer(z)
        return self.disc(z)


class VAE(nn.Module):
    def __init__(self, enc, dec):
        super(VAE, self).__init__()
        self.enc = enc
        self.dec = dec# VarDecoderConv1d_3(2, 60, 10)

    def sampler(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(std.device)
        # kipróbáljuk, milyen lesz a tanulás, ha nem mintavételezünk.
        return eps.mul(std).add_(mu)

    def forward(self, x):
        if self.training:
            err = torch.FloatTensor(x.size()).normal_(torch.tensor(0.0), torch.tensor(0.1)).to(x.device)
            x = x + err
        mu, logvar = self.enc(x)
        z = self.sampler(mu, logvar)
        pred = self.dec(z)  # return h
        return pred, mu, logvar, z


class AdvAE(BPModule):
    def __init__(self, encoder, decoder, disc):
        super(AdvAE, self).__init__()
        self.vae = VAE(encoder, decoder)
        self.disc = disc
        self.losses_keys = ['disc train', 'generator train', 'disc valid', 'generator valid']
        self.bce = nn.BCELoss()

    def sampler(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(std.device)
        # kipróbáljuk, milyen lesz a tanulás, ha nem mintavételezünk.
        return eps.mul(std).add_(mu)

    def forward(self, x):
        return self.vae(x)  # pred, mu, logvar, z

    def training_step(self, optim_configuration, step):
        self.train()

        self.set_dropout(drop_rate=0)
        pred, mu, logvar, z = self(self.trainer.dataloaders["train"])

        ### Disc
        # mu, logvar = self.encoder(self.trainer.dataloaders["train"])
        # z = self.sampler(mu, logvar)
        z_real = torch.FloatTensor(z.size()).normal_().to(z.device)
        # itt még hozzáadok 2 normális eloszlást
        # z_real += torch.FloatTensor(z.size()).normal_(torch.tensor(3.0),torch.tensor(1.0)).to(z.device)
        # z_real += torch.FloatTensor(z.size()).normal_(torch.tensor(-3.0),torch.tensor(1.0)).to(z.device)

        d_real = self.disc(z_real)
        d_fake = self.disc(z)

        loss_real = self.bce(d_real, torch.ones_like(d_real))
        loss_fake = self.bce(d_fake, torch.zeros_like(d_fake))
        disc_loss = (loss_real + loss_fake) / 2
        # if step > 2000:
        disc_loss.backward(retain_graph=True)
        optim_configuration[0][2].step()
        optim_configuration[1][2].step()
        self.optimizer_zero_grad(0, 0, optim_configuration, step)

        ### Generator
        # mu, logvar = self.encoder(self.trainer.dataloaders["train"])
        z = self.sampler(mu, logvar)
        d_fake = self.disc(z)
        gen_loss = self.bce(d_fake, torch.ones_like(d_fake))
        # if step > 2000:
        gen_loss.backward()
        optim_configuration[0][0].step()
        optim_configuration[1][0].step()
        self.optimizer_zero_grad(0, 0, optim_configuration, step)

        ### Reconstruction
        pred, mu, logvar, z = self(self.trainer.dataloaders["train"])

        recon_loss_vae = self.trainer.criterion(pred, mu, logvar, target=self.trainer.dataloaders["train"])
        recon_loss_vae.backward()
        # opt_vae
        optim_configuration[0][1].step()
        optim_configuration[1][1].step()
        self.optimizer_zero_grad(0, 0, optim_configuration, step)

        self.trainer.losses["train"].append(recon_loss_vae.item())
        self.trainer.losses["disc train"].append(disc_loss.item())
        self.trainer.losses["generator train"].append(gen_loss.item())

    def validation_step(self, step):
        self.eval()
        self.freeze()

        self.set_dropout(drop_rate=0)
        # Reconstruction
        pred, mu, logvar, z = self(self.trainer.dataloaders["valid"])

        var = logvar.exp_()
        var = torch.mean(var, axis=(0,1))
        self.trainer.writer.add_scalar("Mean Variance", var, step)


        valid_recon_loss = self.trainer.criterion(pred, mu, logvar, target=self.trainer.dataloaders["valid"])
        self.trainer.losses["valid"].append(valid_recon_loss.item())

        # Disc
        z_real = torch.FloatTensor(z.size()).normal_().to(z.device)
        # itt még hozzáadok 2 normális eloszlást
        # z_real += torch.FloatTensor(z.size()).normal_(3, 1).to(z.device)
        # z_real += torch.FloatTensor(z.size()).normal_(-3, 1).to(z.device)

        d_real = self.disc(z_real)
        d_fake = self.disc(z)
        loss_real = self.bce(d_real, torch.ones_like(d_real))
        loss_fake = self.bce(d_fake, torch.zeros_like(d_fake))
        valid_disc_loss = (loss_real + loss_fake) / 2
        self.trainer.losses['disc valid'].append(valid_disc_loss.item())

        # Generator
        gen_loss = self.bce(d_fake, torch.ones_like(d_fake))
        self.trainer.losses['generator valid'].append(gen_loss.item())

        # Images
        picture_indexes = [4, 8, 38, 15, 55, 64]
        img_batch = np.zeros((len(picture_indexes), 3, 480, 640))
        i = 0
        if step % 100 == 0:
            for n in picture_indexes:
                real = np.transpose(np.array(self.trainer.dataloaders["valid"].to('cpu')), (0, 2, 1))[n]
                out = np.transpose(np.array(pred.to('cpu')), (0, 2, 1))[n]

                img_real_gen = trajs_to_img(real, out, "Real and generated. N= " + str(n))
                img_real_gen = PIL.Image.open(img_real_gen)
                img_real_gen = ToTensor()(img_real_gen)
                img_batch[i] = img_real_gen[0:3]
                i=i+1
            self.trainer.writer.add_images("Real & Out", img_batch, step)

        self.unfreeze()

    def configure_optimizers(self):
        opt_encoder = optim.Adam(self.vae.enc.parameters(), lr=0.0005)
        opt_vae = optim.Adam(self.vae.parameters(), lr=0.001)
        opt_disc = optim.SGD(self.disc.parameters(), lr=0.0005)

        sch_enc = torch.optim.lr_scheduler.CosineAnnealingLR(opt_encoder, T_max=1500)
        sch_vae = torch.optim.lr_scheduler.MultiStepLR(opt_vae, milestones=[8000, 80000, 120000, 170000], gamma=0.5)
        sch_disc = torch.optim.lr_scheduler.CosineAnnealingLR(opt_disc, T_max=1000)
        return [opt_encoder, opt_vae, opt_disc], [sch_enc, sch_vae, sch_disc]

    def optimizer_zero_grad(
            self, epoch: int, batch_idx: int, optimizer: Union[optim.Optimizer, List], optimizer_idx: int):
        for opt in optimizer[0]:
            opt.zero_grad()

    def setup(self, path="data/X_Yfull_dataset.npy"):
        data = np.load(path)
        seq_length = data.shape[3]
        feature_dim = data.shape[2]
        q = 0.1
        data = np.reshape(data, (-1, feature_dim, seq_length))
        V = data.shape[0]

        self.trainer.dataloaders["train"] = np.reshape(data[0:int((1 - 2 * q) * V)], (-1, feature_dim, seq_length))
        self.trainer.dataloaders["test"] = np.reshape(data[int((1 - 2 * q) * V):int((1 - q) * V)],
                                    (-1, feature_dim, seq_length))
        self.trainer.dataloaders["valid"] = np.reshape(data[int((1 - q) * V):], (-1, feature_dim, seq_length))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.trainer.dataloaders["train"] = torch.tensor(self.trainer.dataloaders["train"]).float().to(device)
        self.trainer.dataloaders["test"] = torch.tensor(self.trainer.dataloaders["test"]).float().to(device)
        self.trainer.dataloaders["valid"] = torch.tensor(self.trainer.dataloaders["valid"]).float().to(device)

        self.trainer.dataloaders["train"] = self.trainer.dataloaders["train"] - self.trainer.dataloaders["train"][:, :, 0][:, :, None]
        self.trainer.dataloaders["test"] = self.trainer.dataloaders["test"] - self.trainer.dataloaders["test"][:, :, 0][:, :, None]
        self.trainer.dataloaders["valid"] = self.trainer.dataloaders["valid"] - self.trainer.dataloaders["valid"][:, :, 0][:, :, None]

    def set_dropout(self, model=None,  drop_rate=0.2):
        model = self if model is None else model
        for name, child in model.named_children():
            if isinstance(child, nn.Dropout):
                child.p = drop_rate
            self.set_dropout(child, drop_rate=drop_rate)


class EncoderBN(nn.Module):
    def __init__(self, input_channels, seq_length, context_dim):
        super(EncoderBN, self).__init__()
        self.input_dim = input_channels * seq_length
        self.context_dim = context_dim
        self.precontext_dim = (25 + context_dim)//2 - 2
        self.pdrop = 0.2
        self.encoder_common = nn.Sequential(
            ##### 1
            nn.Conv1d(in_channels=input_channels, out_channels=6, kernel_size=4),  # 57 * 6
            nn.BatchNorm1d(6),
            nn.PReLU(6),
            ##### 2
            nn.Conv1d(in_channels=6, out_channels=12, kernel_size=6),  # 52 * 12
            nn.BatchNorm1d(12),
            nn.PReLU(12),
            ##### 3
            nn.Conv1d(in_channels=12, out_channels=8, kernel_size=8),  # 45 * 8
            nn.BatchNorm1d(8),
            nn.PReLU(8),
            nn.MaxPool1d(kernel_size=3, padding=0, dilation=1),  # 15 * 8
            # A kimenet 8 hosszú, 7 csatornás jel, 56 dimenziós
        )
        self.conv_mu = nn.Sequential(
            nn.Dropout(self.pdrop),
            nn.Conv1d(in_channels=8, out_channels=5, kernel_size=5),  # 11 * 5
            nn.BatchNorm1d(5),
            nn.PReLU(5),
            nn.MaxPool1d(kernel_size=2, padding=0, dilation=1),  # 5 * 5 = 25
        )
        self.conv_logvar = nn.Sequential(
            nn.Dropout(self.pdrop),
            nn.Conv1d(in_channels=8, out_channels=5, kernel_size=5),  # 11 * 5
            nn.BatchNorm1d(5),
            nn.PReLU(5),
            nn.MaxPool1d(kernel_size=2, padding=0, dilation=1),  # 5 * 5 = 25
        )
        self.linearpremu = nn.Linear(25, self.precontext_dim)
        self.linearprelogvar = nn.Linear(25, self.precontext_dim)
        self.prelu_mu = nn.PReLU(1)
        self.prelu_logvar = nn.PReLU(1)
        self.linear_mu = nn.Linear(self.precontext_dim, self.context_dim)
        self.linear_logvar = nn.Linear(self.precontext_dim, self.context_dim)

    def encoder_mu(self, h1):
        mu = self.conv_mu(h1).view(-1, 25)
        mu = self.prelu_mu(self.linearpremu(mu))
        # mu = self.linearpremu(mu)
        return self.linear_mu(mu)

    def encoder_logvar(self, h1):
        logvar = self.conv_logvar(h1).view(-1, 25)
        logvar = self.prelu_logvar(self.linearprelogvar(logvar))
        return self.linear_logvar(logvar)

    def forward(self, x):
        h1 = self.encoder_common(x)
        return self.encoder_mu(h1), self.encoder_logvar(h1)


class Encoder2D(nn.Module):
    def __init__(self, kernel=2):
        super(Encoder2D, self).__init__()
        self.conv = nn.Sequential(
            # input shape: N, 1, 32, 256
            nn.Conv2d(1, 2, kernel_size=(4, 16), stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2, 4, kernel_size=(4, 16), stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(4, 8, kernel_size=(8, 16), stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),
            # out shape: N, 8, 23, 229
        )
        self.mu = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(8, 16), stride=(2,4), padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            #([10, 16, 4, 24])

            nn.Conv2d(16, 27, kernel_size=(4, 24), stride=(2, 4), padding=0),
            nn.Sigmoid()
            # ([10, 27, 1, 1])
        )
        self.logvar = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(8, 16), stride=(2, 4), padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            # ([10, 16, 4, 24])

            nn.Conv2d(16, 27, kernel_size=(4, 24), stride=(2, 4), padding=0),
            nn.Sigmoid()
            # ([10, 27, 1, 1])
        )

    def forward(self, x):
        h = self.conv(x)
        return self.mu(h), self.logvar(h)  # .squeeze(1).squeeze(1)


class Decoder2D(nn.Module):
    def __init__(self, latent_dim=27):
        super(Decoder2D, self).__init__()
        feature = 32
        self.convtr = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feature, kernel_size=(4, 16), stride=1, padding=0),
            nn.BatchNorm2d(feature),
            nn.LeakyReLU(0.2),
            # N, 32, 4, 16

            nn.ConvTranspose2d(feature, feature // 2, kernel_size=(4, 32), stride=2, padding=1),
            nn.BatchNorm2d(feature // 2),
            nn.LeakyReLU(0.2),


            nn.ConvTranspose2d(feature // 2, feature // 4, kernel_size=(4, 10), stride=2, padding=1),
            nn.BatchNorm2d(feature // 4),
            nn.LeakyReLU(0.2),


            nn.ConvTranspose2d(feature // 4, 1, kernel_size=(4, 8), stride=2, padding=1),
            nn.Sigmoid()
            # N, 1, 32, 256

        )

    def forward(self, l):
        return self.convtr(l)


class Decoder2D2(nn.Module):
    def __init__(self, latent_dim=155):
        super(Decoder2D2, self).__init__()
        feature = 64
        self.convtr = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feature, kernel_size=(2, 8), stride=2, padding=0),
            nn.BatchNorm2d(feature),
            nn.LeakyReLU(0.2),
            # N, 32, 4, 16

            nn.ConvTranspose2d(feature, feature // 2, kernel_size=(3, 3), stride=3, padding=1),
            nn.BatchNorm2d(feature // 2),
            nn.LeakyReLU(0.2),


            nn.ConvTranspose2d(feature // 2, feature // 4, kernel_size=(4, 4), stride=3, padding=(1, 1)),
            nn.BatchNorm2d(feature // 4),
            nn.LeakyReLU(0.2),


            nn.ConvTranspose2d(feature // 4, 1, kernel_size=(2, 4), stride=2, padding=1),
            nn.AdaptiveAvgPool2d((16, 128)),
            nn.Sigmoid()
            # N, 1, 16, 128

        )

    def forward(self, l):
        return self.convtr(l)


class Encoder2D2(nn.Module):
    def __init__(self, kernel=2):
        super(Encoder2D2, self).__init__()
        # todo 1: ez mindig nulla lesz?

        p_drop = 0.5 if self.training else 0.0
        self.conv = nn.Sequential(
            # input shape: N, 1, 32, 256
            nn.Conv2d(1, 4, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(4, 8, kernel_size=(5, 5), stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(8, 10, kernel_size=(8, 8), stride=(1, 2), padding=1),
            nn.BatchNorm2d(10),
            nn.LeakyReLU(0.2),
        )
        # N, 10, 9, 61
        self.mu = nn.Sequential(
            nn.Dropout(p_drop),
            nn.Conv2d(10, 10, kernel_size=1, padding=1),
            nn.BatchNorm2d(10),
            nn.LeakyReLU(0.2),

            nn.Conv2d(10, 6, kernel_size=(8, 8), stride=(1, 2), padding=(1, 2)),
            nn.BatchNorm2d(6),
            nn.LeakyReLU(0.2),

            nn.Conv2d(6, 4, kernel_size=(4, 4), stride=1, padding=(1, 2)),
            nn.LeakyReLU(0.2),

            nn.Conv2d(4, 1, kernel_size=1, padding=0)
        )
        # N, 1, 5, 31
        self.logvar = nn.Sequential(
            nn.Dropout(p_drop),
            nn.Conv2d(10, 10, kernel_size=1, padding=1),
            nn.BatchNorm2d(10),
            nn.LeakyReLU(0.2),

            nn.Conv2d(10, 6, kernel_size=(8, 8), stride=(1, 2), padding=(1, 2)),
            nn.BatchNorm2d(6),
            nn.LeakyReLU(0.2),

            nn.Conv2d(6, 4, kernel_size=(4, 4), stride=1, padding=(1, 2)),
            nn.LeakyReLU(0.2),

            nn.Conv2d(4, 1, kernel_size=1, padding=0)
        )
        # N, 1, 5, 31

    def forward(self, x):
        h = self.conv(x)
        return self.mu(h).view(-1, 155, 1, 1), self.logvar(h).view(-1, 155, 1, 1)  # .squeeze(1).squeeze(1)


class VAE2D(BPModule):
    def __init__(self, encoder, decoder):
        super(VAE2D, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.losses_keys = ['train', 'valid']

    def sampler(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(std.device)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        # if self.training:
        #     err = torch.FloatTensor(x.size()).normal_(torch.tensor(0.0), torch.tensor(0.1)).to(x.device)
        #     x = x + err
        mu, logvar = self.encoder(x)
        z = self.sampler(mu, logvar)
        pred = self.decoder(z)  # return h
        return {"output": pred, "mu": mu, "logvar": logvar}, z

    def training_step(self, optim_configuration, step):
        self.train()
        self.optimizer_zero_grad(0, 0, optim_configuration, 0)
        epoch_loss = 0
        for batch in self.trainer.dataloaders["train"]:
            kwargs, z = self(batch)
            loss = self.trainer.criterion(**kwargs, target=batch)
            loss.backward()
            optim_configuration.step()
            epoch_loss = epoch_loss + loss.item()
        self.trainer.losses["train"].append(epoch_loss/len(self.trainer.dataloaders["train"]))
        # self.trainer.writer.add_scalar('train loss', loss.item(), step)

    def validation_step(self, step):
        self.eval()
        self.freeze()
        epoch_loss = 0
        for batch in self.trainer.dataloaders["valid"]:
            kwargs, z = self(batch)
            loss = self.trainer.criterion(**kwargs, target=batch)
            epoch_loss = epoch_loss + loss.item()

        var = kwargs["logvar"].exp_()
        mean_of_var = torch.mean(var, axis=(0,1))
        std_of_var = torch.std(var, axis=(0,1))
        self.trainer.writer.add_scalars('latent var.', {'std': std_of_var,
                                                        'mean of latent var.': mean_of_var}, step)

            # Images
        with torch.no_grad():
            # if step % 10 == 1:
            img_fake_grid = make_grid(boundary_for_grid(kwargs["output"][:16]), normalize=True, nrow=2)
            img_real_grid = make_grid(boundary_for_grid(batch[:16]), normalize=True, nrow=2)

            self.trainer.writer.add_image("Occupancy Real Images", img_real_grid)
            self.trainer.writer.add_image("Occupancy Fake Images", img_fake_grid, step)

        self.trainer.losses["valid"].append(epoch_loss/len(self.trainer.dataloaders["valid"]))
        self.unfreeze()
        # self.trainer.writer.add_scalar('valid loss', loss.item(), step)

    def test_step(self, *args, **kwargs):
        pass

    def configure_optimizers(self):
        return optim.Adam(self.parameters())


class GridVAE_DataModul(BPDataModule):
    def __init__(self, path, split_ratio):
        super(GridVAE_DataModul).__init__()
        self.path = path
        self.seq_length = None
        self.feature_dim = None
        self.data = None
        self.split_ratio = split_ratio
        self.ngsim_train = None
        self.ngsim_test = None
        self.ngsim_val = None
        # todo: batch size, and to BPDataModule too

    def prepare_data(self, *args, **kwargs):
        data = np.load(self.path, allow_pickle=True)
        # data = np.concatenate(data)
        # print(data.shape)
        # print(data.dtype)
        self.data = np.expand_dims(np.concatenate(data), axis=1)
        self.data = self.data[:, :, 7:23, 63:191]
        print(self.data.shape)
        print(self.data.dtype)
        self.set_has_prepared_data(True)

    def setup(self, stage: Optional[str] = None):
        N = self.data.shape[0]
        T = int(self.split_ratio * N)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ngsim_train = torch.tensor(self.data[T+1:N]).float()
        self.ngsim_val = torch.tensor(self.data[0:T]).float()
        self.ngsim_train = self.ngsim_train[torch.randperm(self.ngsim_train.shape[0])].to(device)
        self.ngsim_val = self.ngsim_val[torch.randperm(self.ngsim_val.shape[0])].to(device)

        self.ngsim_val = torch.split(self.ngsim_val, 4000)
        self.ngsim_train = torch.split(self.ngsim_train, 4000)
        self.set_has_setup_test(True)
        self.set_has_setup_fit(True)

    def train_dataloader(self, *args, **kwargs):
        # return DataLoader(self.ngsim_train, batch_size=self.ngsim_train.shape[0])
        return self.ngsim_train

    def val_dataloader(self, *args, **kwargs):
        return self.ngsim_val

    def test_dataloader(self, *args, **kwargs):
        return self.ngsim_test