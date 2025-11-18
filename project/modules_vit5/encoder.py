import torch
import numpy as np

from torch import nn

from project.modules.base import PreTrainedModel
from utils.registry import registry
from utils.vocab import CustomVocab

class Encoder(PreTrainedModel):
    def __init__(self, ):
        super().__init__(module_type="encoder")
        if module_type=="ocr_description":
            self.max_length = type_config["max_length"]
        elif module_type=="gt":
            self.max_length = type_config["max_dec_length"]


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


    


    