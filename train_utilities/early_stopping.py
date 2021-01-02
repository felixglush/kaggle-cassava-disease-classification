"""
    Monitors improvement (decrease) of a metric over the last best.
    Flags for an early stop if no improvement has been seen within a grace period.
"""


class EarlyStopping():
    """
    Args:

    """

    def __init__(self, metric, logger, best_score=None, counter=0, patience=4, delta_improve=0):
        self.metric = metric
        self.best_score = best_score
        self.logger = logger
        self.patience = patience
        self.counter = counter
        self.delta_improve = delta_improve
        self.stop = False

    def __call__(self, score):
        if self.best_score is None:  # initialize
            self.best_score = score
        elif score >= self.best_score - self.delta_improve:  # no improvement
            self.counter += 1
            if self.counter >= self.patience:
                self.logger.info(
                    f'Metric {self.metric} has not seen improvement in {self.patience} epochs. Early stop.')
                self.stop = True
        else:  # improvement occured
            self.best_score = score
            self.counter = 0