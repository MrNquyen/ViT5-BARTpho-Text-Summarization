import torch
import math
from torch import nn
from torch.nn import functional as F
from utils.registry import registry 
from icecream import ic

from project.modules.base import PreTrainedModel
from utils.module_utils import _batch_gather, _get_causal_mask


class Decoder(PreTrainedModel):
    def __init__(self):
        super().__init__(_type="decoder")


    def forward(
        self,
        prev_inds: torch.Tensor,
        input_embed: torch.Tensor,
        input_attention_mask: torch.Tensor
    ):
        
        vit5_dec_output = decoder(
            input_ids=prev_inds,
            encoder_hidden_states=input_embed,
            encoder_attention_mask=input_attention_mask
        )
        vit5_dec_last_hidden_state = vit5_dec_output.last_hidden_state
        results = {
            "vit5_dec_last_hidden_state": vit5_dec_last_hidden_state
        }

        return results

    