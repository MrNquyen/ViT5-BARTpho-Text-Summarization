import torch
import math
from torch import nn
from torch.nn import functional as F
from utils.registry import registry 
from icecream import ic

from utils.module_utils import _batch_gather, _get_causal_mask
from transformers.models.bart.modeling_bart import shift_tokens_right


class Decoder(nn.Module):
    def __init__(self, tokenizer, decoder):
        super().__init__()
        self.tokenizer = tokenizer
        self.decoder = decoder

    def _shift_right(self, x):
        return shift_tokens_right(
            x,
            pad_token_id=self.tokenizer.pad_token_id,
            decoder_start_token_id=self.decoder.config.decoder_start_token_id,
        )

    def forward(
        self,
        input_embed: torch.Tensor,
        input_attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor
    ):
        #-- Decoder
        bartpho_dec_output = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=input_embed,
            encoder_attention_mask=input_attention_mask,
            return_dict=True,
        )

        bartpho_dec_last_hidden_state = bartpho_dec_output.last_hidden_state
        results = {
            "bartpho_dec_last_hidden_state": bartpho_dec_last_hidden_state
        }

        return results

    