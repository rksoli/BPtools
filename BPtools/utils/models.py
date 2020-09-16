from BPtools.core.bpmodule import *


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
        if self.training:
            err = torch.FloatTensor(x.size()).normal_(torch.tensor(0.0), torch.tensor(0.1)).to(x.device)
            x = x + err
        mu, logvar = self.encoder(x)
        z = self.sampler(mu, logvar)
        pred = self.decoder(z)  # return h
        return pred, mu, logvar, z

    def training_step(self, optim_configuration):
        self.train()
        self.optimizer_zero_grad(0, 0, optim_configuration, 0)
        results = self(self.trainer.dataloaders["train"])
        loss = self.trainer.criterion(results,)

    def validation_step(self, *args, **kwargs):
        pass

    def test_step(self, *args, **kwargs):
        pass

    def configure_optimizers(self):
        return optim.Adam(self.parameters())




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
        self.linear = nn.Linear(context_dim, 10)  # a context dimenzióhoz hozzá kell adni 3-at, ha caterogical
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
