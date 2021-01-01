class Callback:
    """
    A base class that all callbacks inherit.
    """

    def phase_started(self, **kwargs): pass

    def phase_ended(self, **kwargs): pass

    def training_started(self, **kwargs): pass

    def training_ended(self, **kwargs): pass

    def epoch_started(self, **kwargs): pass

    def epoch_ended(self, **kwargs): pass

    def batch_started(self, **kwargs): pass

    def batch_ended(self, **kwargs): pass

    def before_forward_pass(self, **kwargs): pass

    def after_forward_pass(self, **kwargs): pass

    def before_backward_pass(self, **kwargs): pass

    def after_backward_pass(self, **kwargs): pass