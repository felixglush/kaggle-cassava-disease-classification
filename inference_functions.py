import numpy as np
from tqdm import tqdm
import torch
from lightning_objects import Model


# runs inference on all trained models, averages/majority votes the result
def ensemble_inference(states, model_arch, num_labels, dataloader, num_samples, device, kaggle, mode='vote'):
    predictions = np.zeros((num_samples, len(states)))
    progress_bar = tqdm(range(len(states)), total=len(states))
    for state_idx in progress_bar:
        model = Model(model_arch, num_labels, fc_nodes=0, pretrained=True).to(device)
        model.load_state_dict(states[state_idx])
        model.eval()

        start = 0
        for batch_idx, data in enumerate(dataloader):
            if kaggle:
                images = data
            else:
                images, labels = data

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
