from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Phase:
    """
    Training is typically separated into two phases: training and validation.
    This class logs metrics to tensorboard and stores data related to the phases.

    Example usage:

    def make_phases(train, valid, bs=32, n_jobs=0):
        return [
            Phase('train', DataLoader(train, bs, shuffle=True, num_workers=n_jobs)),
            Phase('valid', DataLoader(valid, bs, num_workers=n_jobs), grad=False)
        ]
    """

    def __init__(self, name: str, loader: DataLoader, tensorboard_writer: SummaryWriter, gradients=True):
        self.name = name
        self.loader = loader
        self.tb_writer = tensorboard_writer
        self.gradients = gradients
        self.batch_loss = None
        self.batch_idx = 0
        self.rolling_loss = 0
        self.losses = []
