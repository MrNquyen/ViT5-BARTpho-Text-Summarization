import torch
import numpy as np

from torch import nn

from project.modules_vit5.base import PreTrainedModel
from utils.registry import registry
from utils.vocab import CustomVocab

class Encoder(PreTrainedModel):
    def __init__(self, module_type, mode):
        super().__init__(module_type)
        if mode=="ocr_description":
            self.max_length = self.type_config["max_length"]
        elif mode=="gt":
            self.max_length = self.type_config["max_dec_length"]


    def forward(self, texts):
        """
            Text Embedding the batch of texts provided
            Args:
                - texts: Batch of input texts

            Return:
                - torch.Tensor: text_embed of the given texts - BS, max_length, hidden_size
        """
        inputs = self.tokenize(texts)
        text_embed = self.text_embedding(inputs)
        return text_embed


    


    