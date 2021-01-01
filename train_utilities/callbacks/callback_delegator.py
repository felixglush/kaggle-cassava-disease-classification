import re
from typing import List

from train_utilities.callbacks.callback_base import CallbackBase


def to_snake_case(string):
    """Converts CamelCase string into snake_case."""

    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', string)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()


def classname(obj):
    return obj.__class__.__name__


class CallbackGroupDelegator(CallbackBase):
    """
        This invokes callbacks. Example usage:

        cb = CallbacksGroup([
            RollingLoss(),
            Accuracy(),
            Scheduler(
                OneCycleSchedule(t=len(phases[0].loader) * epochs),
                mode='batch'
            ),
            StreamLogger(),
            ProgressBar()
        ])

        Then we can call `cb.<function>(**kwargs)` and each callback in the list will call its implementation of <function>.

        Example: `cb.epoch_started(epoch=epoch)` will call epoch_started (if implemented) for each of RollingLoss, Accuracy,...
    """

    def __init__(self, callbacks: List):
        self.callbacks = callbacks
        # {"rolling_loss": RollingLoss(), etc}
        self.named_callbacks = {to_snake_case(classname(cb)): cb for cb in self.callbacks}

    """ Calls each callback in the group with the supplied method """

    def invoke(self, method, **kwargs):
        # 'method' is a string and we need the function attribute itself
        for cb in self.callbacks:
            # equivalent to cb.<method>(kwargs)
            getattr(cb, method)(**kwargs)

    def fold_started(self, **kwargs): self.invoke('fold_started', **kwargs)

    def fold_ended(self, **kwargs): self.invoke('fold_ended', **kwargs)

    def training_started(self, **kwargs): self.invoke('training_started', **kwargs)

    def training_ended(self, **kwargs): self.invoke('training_ended', **kwargs)

    def epoch_started(self, **kwargs): self.invoke('epoch_started', **kwargs)

    def phase_started(self, **kwargs): self.invoke('phase_started', **kwargs)

    def phase_ended(self, **kwargs): self.invoke('phase_ended', **kwargs)

    def epoch_ended(self, **kwargs): self.invoke('epoch_ended', **kwargs)

    def batch_started(self, **kwargs): self.invoke('batch_started', **kwargs)

    def batch_ended(self, **kwargs): self.invoke('batch_ended', **kwargs)

    def before_forward_pass(self, **kwargs): self.invoke('before_forward_pass', **kwargs)

    def after_forward_pass(self, **kwargs): self.invoke('after_forward_pass', **kwargs)

    def before_backward_pass(self, **kwargs): self.invoke('before_backward_pass', **kwargs)

    def after_backward_pass(self, **kwargs): self.invoke('after_backward_pass', **kwargs)
