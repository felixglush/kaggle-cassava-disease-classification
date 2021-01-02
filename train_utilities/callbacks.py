import re
from typing import List

"""
This file defines a base class all callbacks inherit, a delegator, and the callbacks themselves.
"""


class CallbackBase:
    """ A base class that all callbacks inherit. """

    def fold_started(self, **kwargs): pass

    def fold_ended(self, **kwargs): pass

    def phase_started(self, **kwargs): pass

    def phase_ended(self, **kwargs): pass

    def training_started(self, **kwargs): pass

    def training_ended(self, **kwargs): pass

    def training_interrupted(self, **kwargs): pass

    def epoch_started(self, **kwargs): pass

    def epoch_ended(self, **kwargs): pass

    def batch_started(self, **kwargs): pass

    def batch_ended(self, **kwargs): pass

    def before_forward_pass(self, **kwargs): pass

    def after_forward_pass(self, **kwargs): pass

    def before_backward_pass(self, **kwargs): pass

    def after_backward_pass(self, **kwargs): pass


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
            EarlyStopping(),
            Checkpoint(),
            MetricLogger()
        ])

        Then we can call `cb.<function>(**kwargs)` and each callback  will call its implementation of <function>.
    """

    def __init__(self, callbacks: List):
        self.callbacks = callbacks
        # {"rolling_loss": RollingLoss(), etc}
        self.named_callbacks = {to_snake_case(classname(cb)): cb for cb in self.callbacks}

    def __getitem__(self, item):
        item = to_snake_case(item)
        if item in self.named_callbacks:
            return self.named_callbacks[item]
        raise KeyError(f'callback name is not found: {item}')

    def invoke(self, method, **kwargs):
        """ Calls each callback in the group with the supplied method """
        # 'method' is a string and we need the function attribute itself
        for cb in self.callbacks:
            # equivalent to cb.<method>(kwargs)
            getattr(cb, method)(**kwargs)

    def fold_started(self, **kwargs):
        self.invoke('fold_started', **kwargs)

    def fold_ended(self, **kwargs):
        self.invoke('fold_ended', **kwargs)

    def training_started(self, **kwargs):
        self.invoke('training_started', **kwargs)

    def training_ended(self, **kwargs):
        self.invoke('training_ended', **kwargs)

    def training_interrupted(self, **kwargs):
        self.invoke('training_interrupted', **kwargs)

    def epoch_started(self, **kwargs):
        self.invoke('epoch_started', **kwargs)

    def phase_started(self, **kwargs):
        self.invoke('phase_started', **kwargs)

    def phase_ended(self, **kwargs):
        self.invoke('phase_ended', **kwargs)

    def epoch_ended(self, **kwargs):
        self.invoke('epoch_ended', **kwargs)

    def batch_started(self, **kwargs):
        self.invoke('batch_started', **kwargs)

    def batch_ended(self, **kwargs):
        self.invoke('batch_ended', **kwargs)

    def before_forward_pass(self, **kwargs):
        self.invoke('before_forward_pass', **kwargs)

    def after_forward_pass(self, **kwargs):
        self.invoke('after_forward_pass', **kwargs)

    def before_backward_pass(self, **kwargs):
        self.invoke('before_backward_pass', **kwargs)

    def after_backward_pass(self, **kwargs):
        self.invoke('after_backward_pass', **kwargs)


"""
Example usage below. Note how epoch_started and epoch_ended in Test1/2 get different arguments but when
passed as `cb.epoch_started(epoch=1, msg="hey test1", test2=123)`, each callback acts only on the args
passed into its respective implementation of epoch_started/ended. 
See cbtest_example.py
"""


class Test1(CallbackBase):
    def fold_started(self, fold, metrics, **kwargs):
        print("Test 1 fold_started", fold, metrics)

    def fold_ended(self, fold, metrics, **kwargs):
        print("Test 1 fold_ended", fold, metrics)

    def epoch_started(self, epoch, msg, **kwargs):
        print("Test1 epoch_started", epoch, msg)

    def epoch_ended(self, epoch, msg, **kwargs):
        print("Test1 epoch_ended", epoch, msg)


class Test2(CallbackBase):
    def fold_started(self, fold, metrics, **kwargs):
        print("Test 2 fold_started", fold, metrics)

    def epoch_started(self, epoch, t, **kwargs):
        print("Test2 epoch_started", epoch, t)

    def epoch_ended(self, epoch, t2, **kwargs):
        print("Test2 epoch_ended", epoch, t2)


class EarlyStopping(CallbackBase):
    def __init__(self):
        pass

    def epoch_ended(self, **kwargs):
        pass


class ModelCheckpoint(CallbackBase):
    def __init__(self):
        pass

    def epoch_ended(self, **kwargs):
        pass


class MetricLogger(CallbackBase):
    def __init__(self):
        pass


class ProgressBar(CallbackBase):
    def __init__(self):
        pass

