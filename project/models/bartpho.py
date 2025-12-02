import torch
from torch import nn
from project.modules_bart.decoder import Decoder
from project.modules_bart.encoder import Encoder 
from utils.registry import registry
from utils.utils import count_nan
from utils.module_utils import _batch_gather
from project.modules_bart.classifier import Classifier
from torch.nn import functional as F
from icecream import ic
from tqdm import tqdm
import math
import time

from transformers import AutoConfig, AutoTokenizer, BartForConditionalGeneration


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
        
        self.writer.LOG_INFO("=== Load Pretrained ===")
        self.load_pretrained()

        self.writer.LOG_INFO("=== Build model layers ===")
        self.build_layers()

        self.writer.LOG_INFO("=== Build adjust learning rate ===")
        self.adjust_lr()
    

    def build_writer(self):
        self.writer = registry.get_writer("common")


    def load_pretrained(self):
        self.model_name = self.model_config["pretrained"]
        ic(self.model_name)
        config = AutoConfig.from_pretrained(self.model_name)

        #-- Load pretrained
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        self.model = BartForConditionalGeneration.from_pretrained(
            self.model_name, 
            config=config
        ).to(self.device)

        #-- Load Encoder, Decoder, Classifier
        self.model.gradient_checkpointing_enable()
        self.encoder = self.model.model.encoder
        self.decoder = self.model.model.decoder
        self.classifier = self.model.lm_head


    
    def build_layers(self):
        #~ Load Config
        encoder_config = self.model_config["encoder"]
        decoder_config = self.model_config["decoder"]

        self.max_length = encoder_config["max_length"]
        self.max_dec_length = encoder_config["max_dec_length"]

        #-- Load Layer
        self.encoder_description = Encoder(tokenizer=self.tokenizer, encoder=self.encoder, max_length=self.max_length)
        self.encoder_summary = Encoder(tokenizer=self.tokenizer, encoder=self.encoder, max_length=self.max_dec_length)
        self.decoder = Decoder(
            tokenizer=self.tokenizer, 
            decoder=self.decoder
        )
        self.classifier = Classifier(self.classifier)


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
        ocr_description_attention_mask = ocr_description_inputs["attention_mask"]


        #-- Get summary ids
        gt_caption_inputs = self.encoder_summary.tokenize(gt_captions)
        gt_caption_input_ids = gt_caption_inputs["input_ids"]


        #-- Others params
        batch_size = ocr_description_embed.size(0)
        vocab_size = self.classifier.get_vocab_size()

            #~: Labels:     Tôi  là  AI  .  <EOS>
        labels_input_ids = gt_caption_input_ids.clone()
        labels_input_ids[labels_input_ids == self.encoder_summary.tokenizer.pad_token_id] = -100

        #-- Get ids
        if self.training:
            #~ Decoder input: shift right (prepend pad_token, remove last token)
            #~ Example: [Tôi, là, AI, ., <eos>] -> [<pad>, Tôi, là, AI, .]
            shift_decoder_input_ids = self.decoder._shift_right(gt_caption_input_ids.clone())
            decoder_attention_mask = (shift_decoder_input_ids != self.encoder_summary.tokenizer.pad_token_id).long()

            results = self.forward_mmt(
                input_embed=ocr_description_embed,
                input_attention_mask=ocr_description_attention_mask,
                decoder_input_ids=shift_decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask
            )
            scores = self.forward_output(results=results)
            return scores, gt_caption_input_ids, labels_input_ids
        
        else:
            #~ Greedy Search
            eos_id = self.encoder_summary.tokenizer.eos_token_id
            pad_id = self.encoder_summary.tokenizer.pad_token_id
            start_id = self.decoder.decoder.config.decoder_start_token_id

            with torch.no_grad():
                scores = torch.zeros((batch_size, self.max_dec_length, vocab_size), device=self.device)
                decoder_input_ids = torch.full(
                    (batch_size, 1),
                    fill_value=start_id,
                    dtype=torch.long,
                    device=self.device
                )

                for step in range(self.max_dec_length):
                    decoder_attention_mask = (decoder_input_ids != pad_id).long()
                    results = self.forward_mmt(
                        input_embed=ocr_description_embed,
                        input_attention_mask=ocr_description_attention_mask,
                        decoder_input_ids=decoder_input_ids,
                        decoder_attention_mask=decoder_attention_mask
                    )
                    step_scores = self.forward_output(results=results)
                    argmax_inds = step_scores.argmax(dim=-1).unsqueeze(-1)

                    #~ Assign
                    scores[:, step, :] = step_scores[:, -1, :]
                    decoder_input_ids = torch.concat([decoder_input_ids, argmax_inds[:, -1]], dim=1)

                # gen_ids = decoder_input_ids[:, 1:] #-- Ignore the first pad token
                return scores, decoder_input_ids, labels_input_ids
            


    def forward_mmt(self, input_embed, input_attention_mask, decoder_input_ids, decoder_attention_mask):
        """
            Forward to mmt layer
        """
        results = self.decoder(
            input_embed=input_embed,
            input_attention_mask=input_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
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
        bartpho_dec_last_hidden_state = results["bartpho_dec_last_hidden_state"]
        fixed_scores = self.classifier(bartpho_dec_last_hidden_state)
        return fixed_scores

