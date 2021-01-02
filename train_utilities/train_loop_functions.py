import numpy as np
import torch
from torch.cuda.amp import autocast
from config import Config
import gc 
from model import Model
from tqdm import tqdm
from scipy import stats

# for each sample in this batch, take the maximum predicted class
def process_model_output(predictions, output, batch_size):
    batch_sample_preds = np.array([torch.argmax(output, 1).detach().cpu().numpy()]) # sample preds for the batch
    assert batch_sample_preds.shape == (1, batch_size) 
    predictions = np.concatenate((predictions, batch_sample_preds), axis=None)
    return predictions

# loops over data with gradient scaling and accumulation
def train_epoch(dataloader, model, criterion, optimizer, scheduler, scaler, accum_iter, logger, 
                device, tb_writer, fold, epoch):
    model.train()
    running_loss = 0.0 # for this epoch, across all batches
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader)) 
    
    for batch_idx, (images, labels) in progress_bar:    
        progress_bar.set_description(f"[TRAIN] Processing batch {batch_idx+1}")
        images = images.to(device)
        labels = labels.to(device)
        with autocast():
            predictions = model(images)
            assert len(predictions) == len(images) == len(labels)
            loss = criterion(predictions, labels)
        loss = loss / accum_iter # loss is normalized across the accumulated batches
        
        running_loss += loss.detach().item()
        
        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        # Backward passes under autocast are not recommended.
        # Backward ops run in the same dtype autocast chose for corresponding forward ops.
        # See https://pytorch.org/docs/stable/amp.html#gradient-scaling for why scaling is helpful
        scaler.scale(loss).backward()
        
        # increase LR ... (LR Range test)
        #if scheduler: scheduler.step()
        
        total_norm = 0.
        
        if scheduler and epoch > Config.wait_epochs_schd and Config.scheduler == "CosineAnnealingWarmRestarts":
            scheduler.step()
        
        # Gradient accumulation (larger effective batch size)
        if (batch_idx + 1) % accum_iter == 0 or (batch_idx + 1) == len(dataloader):
            # if want to implement gradient clipping, see this first. need to unscale gradients first.
            # https://pytorch.org/docs/stable/notes/amp_examples.html#working-with-unscaled-gradients
        
            # Unscales the gradients of optimizer's assigned params in-place
            scaler.unscale_(optimizer)

            # Monitor gradients for explosions
            for p in list(filter(lambda p: p.grad is not None, model.parameters())):
                param_norm = p.grad.data.norm(2).item() # norm of the gradient tensor
                total_norm += param_norm ** 2
            total_norm = np.sqrt(total_norm)
            
            total_batches_processed = epoch * len(dataloader) + batch_idx
            tb_writer.add_scalar(f'Train loss fold {fold}', running_loss / (batch_idx + 1), total_batches_processed)
            tb_writer.add_scalar(f'Gradient fold {fold}', total_norm, total_batches_processed)
            tb_writer.add_scalar(f'Learning Rate', optimizer.param_groups[0]['lr'], total_batches_processed) 
            
            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
            #torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_norm_grad)
        
            # scaler.step() first unscales (if they're not already) the gradients of the optimizer's assigned params.
            # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.
            scaler.step(optimizer)

            # Updates the scale for next iteration.
            scaler.update()
            optimizer.zero_grad()
        gc.collect()
    logger.info(f'[TRAIN] batch loss: {running_loss / (batch_idx + 1)}')
    return running_loss / len(dataloader)

def valid_epoch(dataloader, model, criterion, logger, device, tb_writer, fold, epoch=None, holdout=False):
    model.eval()
    running_loss = 0.0 # for this epoch, across all batches
    predictions = np.array([])
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader)) 

    for batch_idx, (images, labels) in progress_bar:
        progress_bar.set_description(f"[VAL] Processing batch {batch_idx+1}")
        
        images = images.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            output = model(images)
            # output: [batch_size, # classes] -> [batch_size, 5]
        loss = criterion(output, labels)
        running_loss += loss.detach().item()
        
        if (batch_idx + 1) % Config.print_every == 0:
            if not holdout:
                total_batches_processed = epoch * len(dataloader) + batch_idx
                tb_writer.add_scalar(f'Validation loss fold {fold}', running_loss / (batch_idx + 1), total_batches_processed)
            else:
                tb_writer.add_scalar(f'Holdout loss fold {fold}', running_loss / (batch_idx + 1), fold)
                
        # for each sample in this batch, take the maximum predicted class
        predictions = process_model_output(predictions, output, batch_size=images.size(0))

        gc.collect()
        
    logger.info(f'[VAL] batch loss: {running_loss / (batch_idx+1)}')    
    return running_loss / len(dataloader), predictions

# runs inference on all trained models, averages/majority votes the result   
def ensemble_inference(states, model_arch, num_labels, dataloader, num_samples, device, mode='vote', test=True):
    predictions = np.zeros((num_samples, len(states)))
    progress_bar = tqdm(range(len(states)), total=len(states))
    for state_idx in progress_bar:
        model = Model(model_arch, num_labels, False, 0, pretrained=True).to(device)
        model.load_state_dict(states[state_idx])
        model.eval()
       
        start = 0
        for batch_idx, data in enumerate(dataloader):
            if test: images = data
            else: images, _ = data
            
            end = start + len(images)
            
            images = images.to(device)

            with torch.no_grad(): 
                output = model(images)
                            
            batch_sample_preds = np.array([torch.argmax(output, 1).detach().cpu().int().numpy()])
            predictions[start:end, state_idx] = np.reshape(batch_sample_preds, (len(images),))
            
            start = end
            
        del model
        gc.collect()
        
    if mode == 'avg':
        return np.round(np.mean(predictions, axis=1), decimals=0)
    elif mode == 'vote':
        return stats.mode(predictions, axis=1)[0]