from ray import tune
class Config:
    seed = 123
    data_dir = './data'
    train_img_dir = data_dir + '/train_images'
    test_img_dir = data_dir + '/test_images'
    data_csv = data_dir + '/train.csv'
    save_dir = './trained-models'
    target_col = 'label'
    
    model_arch =  'tf_efficientnet_b4_ns'
    img_size = 512

    # run settings
    hyperparameter_tuning = True
    debug = False
    train = True
    inference = True
    
    train_bs = 16
    valid_bs = 32
    is_amsgrad = False
    accum_iter = 8          # Gradient accumulation
    # lr = 0.001285         # adam/adaboost
    lr = 0.5286             # SGD selected with LR Range Test.
    max_lr = 0.9            # SGD with 1cycle 
    min_lr = 1e-4 
    weight_decay = 0.000125 # Adam
    momentum = 0.9          # SGD
    loss_patience = 6       # for early stopping
    max_norm_grad = 1.    
    fold_num = 5
    epochs = 25
    print_every = 10
    
    # lr scheduler settings
    # supported: ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts', 'StepLR']
    scheduler = 'CosineAnnealingWarmRestarts'
    factor = 0.2     # ReduceLROnPlateau
    patience = 4     # ReduceLROnPlateau
    gamma = 1.025    # StepLR for range test
    step_size_lr = 1 # StepLR for range test
    eps = 1e-6       # ReduceLROnPlateau
    T_max = 10       # CosineAnnealingLR
    T_mult = 2       # CosineAnnealingWarmRestarts
    T_0 = 10         # CosineAnnealingWarmRestarts
    wait_epochs_schd = 2 # how many epochs to wait until starting CosineAnnealingWarmRestarts
    schd_verbose = False
    num_workers = 0
    verbose_step = 1
    