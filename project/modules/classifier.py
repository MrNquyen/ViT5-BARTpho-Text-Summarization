import torch
from torch import nn

from utils.registry import registry

class Classifier(nn.Module):
    def __init__(self, hidden_size, num_choices):
        super().__init__()
        self.num_choices = num_choices
        self.hidden_size = hidden_size
        self.device = registry.get_args("device")
        self.writer = registry.get_writer("common")

        #~ Build
        self.load_weight()

    #-- Build
    def load_weight(self):
        self.classifier = nn.Linear(self.hidden_size, self.num_choices)

    def get_fixed_embed(self):
        return self.classifier.weight

    def get_vocab_size(self):
        return self.get_fixed_embed().size(0)

    
    def forward(self, inputs):
        return self.classifier(inputs)