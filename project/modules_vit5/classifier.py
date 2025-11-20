import torch
from torch import nn

from project.modules_vit5.base import PreTrainedModel
from utils.registry import registry

class Classifier(PreTrainedModel):
    def __init__(self):
        super().__init__()
        self.device = registry.get_args("device")
        self.writer = registry.get_writer("common")
    
    def forward(self, dec_last_hidden_state):
        return self.classifier(dec_last_hidden_state)