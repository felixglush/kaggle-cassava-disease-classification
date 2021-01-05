from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Phase:
    """
    Training is typically separated into two phases: training and validation.
    """

    def __init__(self, name: str, loader: DataLoader, gradients=True, every_epoch=True):
        self.name = name
        self.loader = loader
        self.is_training = gradients
        self.batch_idx = 0
        self.accuracy = 0
        self.best_loss = float('inf')  # for the entire fold (i.e. min of self.losses)
        self.running_loss = 0
        self.epoch_losses = []
        self.every_epoch = every_epoch
        self.latest_preds = None

    def reset(self):
        self.accuracy = 0
        self.running_loss = 0
        self.epoch_losses = []

    @property
    def last_epoch_loss(self):
        return self.epoch_losses[-1] if self.epoch_losses else None

    def update_loss(self, loss):
        self.epoch_losses.append(loss)
        self.running_loss += loss

    def average_epoch_loss(self):
        return self.running_loss / (self.batch_idx + 1)
