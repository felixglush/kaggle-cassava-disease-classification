import numpy as np
import torch
from torch.cuda.amp import autocast
from config import Config
# for each sample in this batch, take the maximum predicted class
def process_model_output(predictions, output, batch_size):
    predicted_class_per_sample = np.array([torch.argmax(output, 1).detach().cpu().numpy()])
    assert predicted_class_per_sample.shape == (1, batch_size) 
    predictions = np.concatenate((predictions, predicted_class_per_sample), axis=None)
    return predictions

# loops over data with gradient scaling and accumulation
def train_epoch(dataloader, model, criterion, optimizer, scheduler, scaler, accum_iter, logger, device):
    model.train()
    batch_losses = []

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        with autocast():
            predictions = model(images)
            loss = criterion(predictions, labels)
        
        batch_losses.append(loss.item())

        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        # Backward passes under autocast are not recommended.
        # Backward ops run in the same dtype autocast chose for corresponding forward ops.
        # See https://pytorch.org/docs/stable/amp.html#gradient-scaling for why scaling is helpful
        scaler.scale(loss).backward()
        total_norm = 0.
        
        if batch_idx + 1 % accum_iter == 0 or batch_idx + 1 == len(dataloader):
            # if want to implement gradient clipping, see this first. need to unscale gradients first.
            # https://pytorch.org/docs/stable/notes/amp_examples.html#working-with-unscaled-gradients
        
            # Unscales the gradients of optimizer's assigned params in-place
            scaler.unscale_(optimizer)

            # TEMPORARY
            # get gradients to check for explosions and determine clipping value
            for p in list(filter(lambda p: p.grad is not None, model.parameters())):
                param_norm = p.grad.data.norm(2).item() # norm of the gradient tensor
                total_norm += param_norm ** 2
            total_norm = np.sqrt(total_norm)

            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_norm_grad)
        
            # scaler.step() first unscales (if they're not already) the gradients of the optimizer's assigned params.
            # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.
            scaler.step(optimizer)

            # Updates the scale for next iteration.
            scaler.update()
            optimizer.zero_grad()
                
        if batch_idx + 1 % Config.print_every == 0 or batch_idx + 1 == len(dataloader):
            logger.info(f'[TRAIN] batch {batch_idx+1}/{len(dataloader)} loss: {loss} | grad: {total_norm}')

    return batch_losses

def valid_epoch(dataloader, model, criterion, accum_iter, logger, device):
    model.eval()
    batch_losses = []
    predictions = np.array([])
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            output = model(images)
            # output: [batch_size, # classes] -> [batch_size, 5]
        loss = criterion(output, labels)
        
        batch_losses.append(loss)
        # for each sample in this batch, take the maximum predicted class
        predictions = process_model_output(predictions, output, batch_size=images.size(0))
        
        if batch_idx + 1 % Config.print_every == 0 or batch_idx + 1 == len(dataloader):
            logger.info(f'[VAL] batch {batch_idx+1}/{len(dataloader)} loss: {loss / accum_iter}')
        
    return batch_losses, predictions
    
def inference(model, dataloader, device):
    model.eval()
    predictions = np.array([])
    #targets = np.array([])
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        
        with torch.no_grad():
            output = model(images)
        
        # for each sample in this batch, take the maximum predicted class
        predictions = process_model_output(predictions, output, batch_size=images.size(0))
        
        #targets  = np.concatenate((targets, labels), axis=None)
    
    return predictions