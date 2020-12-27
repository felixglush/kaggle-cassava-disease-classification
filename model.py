import torch
import sys
sys.path.append('../pytorch-image-models')
import timm # pytorch-image-models implementations

# EfficientNet noisy student: https://arxiv.org/pdf/1911.04252.pdf. Implementation from
# https://github.com/rwightman/pytorch-image-models.
class Model(torch.nn.Module):
    def __init__(self, model_arch, n_classes, middle_fc=False, middle_fc_size=0, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        # replace classifier with a Linear in_features->n_classes layer
        in_features = self.model.classifier.in_features
        
        if middle_fc and middle_fc_size > 0:
            self.model.classifier = torch.nn.Linear(in_features, middle_fc_size)
            self.model.fc2 = torch.nn.Linear(middle_fc_size, n_classes)
        else:
            self.model.classifier = torch.nn.Linear(in_features, n_classes)
        
    def forward(self, x):
        return self.model(x)