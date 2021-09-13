from BPtools.core.bpmodule import *
from BPtools.core.bpdatamodule import *
from BPtools.utils.trajectory_plot import *
# from torchvision.transforms import ToTensor
# from torchvision.utils import make_grid
# from torch import functional as F


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