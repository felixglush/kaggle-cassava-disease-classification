
class Trainer():
    def __init__(self, folds_df, holdout_loader, logger, tensorboard_writer, device, settings, checkpoint_params):
        self.holdout_loader = holdout_loader
        self.checkpoint_params = checkpoint_params
        self.settings = settings
        self.folds_df = folds_df
        self.logger = logger
        self.tensorboard_writer = tensorboard_writer
        self.device = device
        self.epoch = None
        self.fold = None
        self.train_loader = None
        self.valid_loader = None
        self.holdout_loader = None
        self.model = None
        # etc

    
    def fit():
        pass
    
    """
        Trains a model corresponding to a particular fold over epochs
        
        Returns a DataFrame consisting of only the the rows used for validation along with the model's predictions
    """
    def train_fold(train_folds_df, fold, 
                     device, basename, 
                     holdout_dataloader, holdout_targets, 
                     tb_writer, checkpoint=None):
    model_checkpoint_name = basename + f'/{Config.model_arch}_fold{fold}.pth'
    
    # -------- DATASETS AND LOADERS --------
    # select one of the folds, create train & validation set loaders
    train_df, valid_df = get_data_dfs(train_folds_df, fold)
    train_dataloader, valid_dataloader = get_loaders(train_df, valid_df,
                                                     Config.train_bs, 
                                                     Config.data_dir + '/train_images')
    
    # make model and optimizer
    model, optimizer = setup_model_optimizer(Config.model_arch, 
                                           Config.lr, 
                                           Config.is_amsgrad, 
                                           num_labels=train_folds_df.label.nunique(), 
                                           weight_decay=Config.weight_decay,
                                           momentum=Config.momentum,
                                           fc_layer={"middle_fc": False, "middle_fc_size": 0},
                                           device=device,
                                           checkpoint=checkpoint)
    
    scheduler, criterion = get_schd_crit(optimizer)
    
    accuracy = 0.
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    
    early_stop = EarlyStopping('val_loss', LOGGER, patience=Config.loss_patience)

    for e in range(Config.epochs):
        epoch_start_time = time.time()
        LOGGER.info(f'Training epoch {e+1}/{Config.epochs}')
        
        # -------- TRAIN --------
        avg_training_loss = train_epoch(train_dataloader, model, 
                                      criterion, optimizer, 
                                      scheduler, GradScaler(), 
                                      Config.accum_iter, LOGGER,
                                      device, tb_writer, fold, e)

        # -------- VALIDATE --------
        avg_validation_loss, preds = valid_epoch(valid_dataloader, model, 
                                                 criterion, LOGGER, device, 
                                                 tb_writer, fold, e)
        
        train_losses.append(avg_training_loss)
        val_losses.append(avg_validation_loss)

        # -------- SCORE METRICS & LOGGING FOR THIS EPOCH --------
        validation_labels = valid_df[Config.target_col].values
        accuracy = accuracy_score(y_true=validation_labels, y_pred=preds)
       
        epoch_elapsed_time = time.time() - epoch_start_time
        
        tb_writer.add_scalar(f'Avg Epoch Train Loss Fold {fold}', avg_training_loss, e)
        tb_writer.add_scalar(f'Avg Epoch Val Loss Fold {fold}', avg_validation_loss, e)
        tb_writer.add_scalar(f'Epoch Val Accuracy Fold {fold}', accuracy, e)
        
        LOGGER.info(f'\nEpoch training summary:\n Fold {fold+1}/{Config.fold_num} | ' + \
                    f'Epoch: {e+1}/{Config.epochs} | ' + \
                    f'Epoch time: {epoch_elapsed_time} sec\n' + \
                    f'Training loss: {avg_training_loss} | ' + \
                    f'Validation loss: {avg_validation_loss} | ' + \
                    f'Accuracy: {accuracy}')
        
        early_stop(avg_validation_loss)
        if early_stop.stop: break
            
        # --------SAVE MODEL --------
        if avg_validation_loss < best_val_loss: 
            best_val_loss = avg_validation_loss
            torch.save({'model_state': model.state_dict(), 
                        'optimizer_state': optimizer.state_dict(),
                        'accuracy': accuracy, 
                        'preds': preds,
                        'val_loss': best_val_loss,
                        'fold': fold,
                        'epochs_no_improve': early_stop.counter,
                        'epoch_stopped_at': e
                       }, model_checkpoint_name)
            LOGGER.info(f'Saved model!')
        LOGGER.info('----------------')
        
        # -------- UPDATE LR --------
        if scheduler and e > 2:
            if Config.scheduler == 'ReduceLROnPlateau':
                scheduler.step(avg_validation_loss)
        gc.collect()

    # -------- TEST ON HOLDOUT SET --------
    # load best model
    checkpoint = torch.load(model_checkpoint_name)
    model.load_state_dict(checkpoint['model_state']) 
    holdout_loss, holdout_preds = valid_epoch(holdout_dataloader, model, 
                                              criterion, LOGGER, device, 
                                              tb_writer, fold, holdout=True)
    holdout_accuracy = accuracy_score(y_true=holdout_targets, y_pred=holdout_preds)
    
    tb_writer.add_scalar(f'Fold {fold} holdout accuracy', holdout_accuracy, fold)
    tb_writer.add_scalar(f'Fold {fold} holdout loss', holdout_loss, fold)
    
    valid_df['prediction'] = checkpoint['preds']
    
    del model
    del optimizer
    del train_dataloader
    del valid_dataloader
    return valid_df, checkpoint['accuracy'], holdout_accuracy
        
    
    """
        Runs through the training dataset-
    """
    def train_loop():
        pass
    
    
    """
        Runs through the validation dataset
    """
    def valid_loop():
        pass
    
    def test():
        pass
    
   