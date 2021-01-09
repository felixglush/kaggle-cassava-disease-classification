class Configuration:
    def __init__(self):

        self.seed = 123
        self.data_dir = './data'
        self.train_img_dir = './data/train_images'
        self.test_img_dir = './data/test_images'
        self.data_csv = './data/train.csv'
        self.save_dir = './trained-models'
        self.target_col = 'label'
        self.num_classes = 5
        self.img_size = 512
        self.print_every = 10

        self.model_arch = 'tf_efficientnet_b4_ns'

        # run settings
        self.hyperparameter_tuning = True
        self.debug = False
        self.train = True
        self.inference = True

        self.train_bs = 16
        self.valid_bs = 64
        self.is_amsgrad = False
        self.grad_accumulator_steps = 4
        # lr: 0.001285         # adam/adaboost
        self.lr = 0.2  # SGD selected
        self.lr_test = False
        self.max_lr = 0.9
        self.min_lr = 1e-4
        self.weight_decay = 0.000125  # Adam
        self.momentum = 0.9  # SGD
        self.loss_patience = 6  # for early stopping
        self.max_norm_grad = 1.
        self.fold_num = 5
        self.epochs = 25

        # lr scheduler settings
        # supported: ['ReduceLROnPlateau', 'CosineAnnealingLR',  'CosineAnnealingWarmRestarts',  'StepLR']
        self.scheduler = 'CosineAnnealingWarmRestarts'
        self.factor = 0.2  # ReduceLROnPlateau
        self.patience = 4  # ReduceLROnPlateau
        self.gamma = 1.025  # StepLR for range test
        self.step_size_lr = 1  # StepLR for range test
        self.eps = 1e-6  # ReduceLROnPlateau
        self.T_max = 10  # CosineAnnealingLR
        self.T_mult = 2  # CosineAnnealingWarmRestarts
        self.T_0 = 10  # CosineAnnealingWarmRestarts
        self.wait_epochs_schd = 2  # how many epochs to wait until starting lr schedule
        self.schedule_verbosity = False
        self.num_workers = 0