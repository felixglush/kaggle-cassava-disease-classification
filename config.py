class Configuration:
    def __init__(self):
        self.seed = 123
        self.data_dir = './data'
        self.train_img_dir = './data/train_images'
        self.test_img_dir = './data/test_images'
        self.data_csv = './data/train.csv'
        self.target_col = 'label'
        self.num_classes = 5
        self.img_size = 512
        self.print_every = 10

        self.model_arch = 'tf_efficientnet_b4_ns'
        # self.model_arch = 'efficientnet_b3a'
        # self.model_arch = 'seresnet50'
        # run settings
        self.hyperparameter_tuning = True
        self.debug = False
        self.train = True
        self.inference = True

        self.train_bs = 16
        self.valid_bs = 64
        self.is_amsgrad = False
        self.grad_accumulator_steps = 4
        # `lr`: 0.001285         # adam/adaboost
        self.lr_test = True
        self.lr = 0.02 # SGD
        self.max_lr = 0.8
        self.min_lr = 1e-3
        self.weight_decay = 0.000125  # Adam
        self.momentum = 0.9  # SGD
        self.patience = 8  # for early stopping
        self.max_norm_grad = 1.
        self.fold_num = 5
        self.epochs = 8

        # loss function params
        self.t1 = 0.8
        self.t2 = 1.
        self.smoothing = 0.05

        # lr scheduler settings
        self.T_max = self.epochs  # CosineAnnealingLR
        self.T_mult = 2  # CosineAnnealingWarmRestarts
        self.T_0 = 10  # CosineAnnealingWarmRestarts
        self.schedule_verbosity = False
        self.num_workers = 5
