import torch
from torch import nn
from project.modules.decoder import Decoder
from project.modules.encoder import Encoder 
from utils.registry import registry
from utils.utils import count_nan
from utils.module_utils import _batch_gather
from project.modules.classifier import Classifier
from torch.nn import functional as F
from icecream import ic
from tqdm import tqdm
import math
import time


class TransformerSummarizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_config = registry.get_config("model_attributes")
        self.device = registry.get_args("device")
        self.writer = registry.get_writer("common")
        self.hidden_size = self.model_config["hidden_size"]
        self.build()

    #-- BUILD
    def build(self):
        self.writer.LOG_INFO("=== Build writer ===")
        self.build_writer()

        self.writer.LOG_INFO("=== Build model init ===")
        self.build_model_init()

        self.writer.LOG_INFO("=== Build model layers ===")
        self.build_layers()

        self.writer.LOG_INFO("=== Build model outputs ===")
        self.build_output()

        self.writer.LOG_INFO("=== Build adjust learning rate ===")
        self.adjust_lr()
    

    def build_writer(self):
        self.writer = registry.get_writer("common")


    def build_layers(self):
        self.decoder = Decoder()
        self.encoder_description = Encoder()
        self.encoder_summary = Encoder(self.encoder_description.tokenizer)

    # Need to modify
    def build_output(self):
        num_choices = self.encoder_summary.get_vocab_size()
        self.classifier = Classifier(self.hidden_size, num_choices)
    

    def build_model_init(self):
        #~ Finetune module is the module has lower lr than others module
        self.finetune_modules = []


    def adjust_lr(self):
        #~ Word Embedding
        # self.add_finetune_modules(self.classifier)
        # self.add_finetune_modules(self.decoder.encoder)
        pass

    #-- ADJUST LEARNING RATE
    def add_finetune_modules(self, module: nn.Module):
        self.finetune_modules.append({
            'module': module,
            'lr_scale': self.model_config["adjust_optimizer"]["lr_scale"],
        })

    def get_optimizer_parameters(self, config_optimizer):
        """
            -----
            Function:
                - Modify learning rate
                - Fine-tuning layer has lower learning rate than others
        """
        optimizer_param_groups = []
        base_lr = config_optimizer["params"]["lr"]
        scale_lr = config_optimizer["lr_scale"]
        base_lr = float(base_lr)
        
        # collect all the parameters that need different/scaled lr
        finetune_params_set = set()
        for m in self.finetune_modules:
            optimizer_param_groups.append({
                "params": list(m['module'].parameters()),
                "lr": base_lr * scale_lr
            })
            finetune_params_set.update(list(m['module'].parameters()))
        # remaining_params are those parameters w/ default lr
        remaining_params = [
            p for p in self.parameters() if p not in finetune_params_set
        ]
        # put the default lr parameters at the beginning
        # so that the printed lr (of group 0) matches the default lr
        optimizer_param_groups.insert(0, {"params": remaining_params})
        
        # check_overlap(finetune_params_set, remaining_params)
        return optimizer_param_groups
        
    def init_random(self, shape):
        return torch.normal(
            mean=0.0, 
            std=0.1, 
            size=shape
        ).to(self.device)
        
    #-- FORWARD
    def forward(
            self,
            batch
        ):
        gt_captions = batch["list_captions"]
        gt_caption_tokens = batch["list_caption_tokens"]
        ocr_descriptions = batch["list_ocr_descriptions"]

        #-- Get inputs
        ocr_description_inputs = self.encoder_description.tokenize(ocr_descriptions)
        ocr_description_embed = self.encoder_description.text_embedding(ocr_description_inputs)

        #-- Get summary inds
        gt_caption_input_inds = self.encoder_summary.get_word_inds(gt_caption_tokens)

        #-- Get inds
        batch_size = len(gt_captions)
        if self.training:
            results = self.forward_mmt(
                prev_inds= gt_caption_input_inds,
                input_embed=ocr_description_embed,
                fixed_ans_emb=self.classifier.get_fixed_embed(),
                input_attention_mask=ocr_description_inputs["attention_mask"]
            )
            scores = self.forward_output(results=results)
            return scores, gt_caption_input_inds, gt_caption_input_inds
        else:
            num_dec_step = self.decoder.max_length
            # Init prev_ids with <s> idx at begin, else where with <pad> (at idx 0)
            start_idx = self.encoder_description.get_cls_token_id() 
            pad_idx = self.encoder_description.get_pad_token_id()
            fixed_ans_emb = self.classifier.get_fixed_embed()
            prev_inds = torch.full((batch_size, num_dec_step), pad_idx).to(self.device)
            prev_inds[:, 0] = start_idx
            scores = None
            for i in tqdm(range(1, num_dec_step)):
                results = self.forward_mmt(
                    prev_inds= prev_inds,
                    input_embed=ocr_description_embed,
                    fixed_ans_emb=fixed_ans_emb,
                    input_attention_mask=ocr_description_inputs["attention_mask"]
                )
                scores = self.forward_output(results)
                argmax_inds = scores.argmax(dim=-1)
                prev_inds[:, i] = argmax_inds[:, -1]
            return scores, prev_inds, gt_caption_input_inds

    def forward_mmt(self, prev_inds, input_embed, fixed_ans_emb, input_attention_mask):
        """
            Forward to mmt layer
        """
        results = self.decoder(
            prev_inds= prev_inds,
            input_embed=input_embed,
            fixed_ans_emb=fixed_ans_emb,
            input_attention_mask=input_attention_mask
        )
        return results
        
    def forward_output(self, results):
        """
        Calculate scores for ocr tokens and common word at each timestep

            Parameters:
            ----------
            results: dict
                - The result output of decoder step

            Return:
            ----------
        """
        mmt_dec_output = results["mmt_dec_output"]
        fixed_scores = self.classifier(mmt_dec_output)
        return fixed_scores

