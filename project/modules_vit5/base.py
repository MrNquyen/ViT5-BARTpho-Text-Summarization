import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig, T5ForConditionalGeneration
from typing import List
from utils.registry import registry
from icecream import ic

#========= TEXT ENCODER ===========
class PreTrainedModel(nn.Module):
    def __init__(self, module_type="encoder"):
        super().__init__()
        #-- Load config and args
        self.writer = registry.get_writer("common")
        self.model_config = registry.get_config("model_attributes")
        self.type_config = self.model_config[module_type]
        self.device = registry.get_args("device")
        self.max_length = self.type_config["max_length"]
        self.hidden_size = self.model_config["hidden_size"]
        self.load_pretrained()


    def load_pretrained(self):
        self.model_name = self.type_config["pretrained"]
        config = AutoConfig.from_pretrained(self.model_name)

        #-- Load pretrained
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.model_name, 
            config=config
        ).to(self.device)

        #-- Load Encoder, Decoder, Classifier
        self.model.gradient_checkpointing_enable()
        self.encoder = self.model.encoder
        self.decoder = self.model.decoder
        self.classifier = self.model.lm_head

        for param in self.encoder.parameters():
            param.requires_grad = False
        self.writer.LOG_INFO("Freeze the encoder params")


    #-- Tokenize
    def tokenize(self, texts: List[str]):
        """
            Args:
                - texts: (str): Batch of texts

            Return:
                - dict: Text input has 'input_ids', 'attention_mask'

            Example: 
                - Return a dict = {
                    'input_ids': ..., 
                    'attention_mask': ...,
                }
        """
        inputs = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"        # return PyTorch tensors directly
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs # 'input_ids', 'token_type_ids', 'attention_mask'
    

    def text_embedding(self, inputs):
        """
            Args:
                - inputs: (str): Input is a output from tokenizer

            Return:
                - Tensor: Tensor of text embeb features

            Example: 
                - inputs = {
                    'input_ids': ..., 
                    'token_type_ids': ..., 
                    'attention_mask': ...,
                }
        """
        with torch.no_grad():
            encoder_outputs = self.encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            return encoder_outputs.last_hidden_state



    #-- Decode
    def batch_decode(self, pred_inds):
        return self.tokenizer.batch_decode(pred_inds, skip_special_tokens=True)

    #-- Common function
    def get_vocab_size(self):
        return len(self.tokenizer.get_vocab())

    #-- Get token
    def get_pad_token(self):
        return self.tokenizer.pad_token
    
    def get_unk_token(self):
        return self.tokenizer.unk_token
    
    def get_cls_token(self):
        return self.tokenizer.cls_token
    
    def get_eos_token(self):
        return self.tokenizer.eos_token
    
    #-- Get token id
    def get_pad_token_id(self):
        return self.tokenizer.pad_token_id
    
    def get_unk_token_id(self):
        return self.tokenizer.unk_token_id
    
    def get_cls_token_id(self):
        return self.tokenizer.cls_token_id
    
    def get_eos_token_id(self):
        return self.tokenizer.eos_token_id




