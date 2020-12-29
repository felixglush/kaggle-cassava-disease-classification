from ray import tune
class Config:
    # run settings
    hyperparameter_tuning = True
    debug = False
    train = True
    inference = True
    
    train_bs = 16
    is_amsgrad = False
    # gradient accumulation --> effectively larger batch size when larger batches don't fit in memory
    accum_iter = 8
    lr = 0.0012185
    weight_decay = 0.000125
    
    
    fold_num = 4
    seed = 123
    model_arch =  'tf_efficientnet_b4_ns'
    img_size = 512
    epochs = 20
    valid_bs = 32
    print_every = 10
    max_norm_grad = 1.
    # lr scheduler settings
    # supported: ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']
    scheduler = 'CosineAnnealingWarmRestarts'
    factor=  0.2 # ReduceLROnPlateau
    patience = 4 # ReduceLROnPlateau
    eps = 1e-6 # ReduceLROnPlateau
    T_max = 10 # CosineAnnealingLR
    T_mult = 1
    T_0 = 10 # CosineAnnealingWarmRestarts
    #lr = 1e-4

    min_lr = 1e-5
    
    num_workers = 0
    verbose_step = 1
    target_col = 'label'

    data_dir = './data'
    train_img_dir = data_dir + '/train_images'
    test_img_dir = data_dir + '/test_images'
    data_csv = data_dir + '/train.csv'
    save_dir = './trained-models'