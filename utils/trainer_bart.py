import torch
import warnings
import os
import numpy as np

from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from utils.configs import Config
from project.dataset.dataset import get_loader
from utils.model_utils import get_optimizer_parameters, lr_lambda_update, lr_lambda_update_epoch
from utils.module_utils import _batch_padding, _batch_padding_string
from utils.logger import Logger
from utils.metrics import metric_calculate
from utils.utils import save_json, count_nan
from utils.registry import registry
from project.models.bartpho import TransformerSummarizer
from icecream import ic
from utils.utils import check_requires_grad

# ~Trainer~
class Trainer():
    def __init__(self, config, args):
        self.args = args
        self.device = args.device
        self.config = Config(config)

        print("Build Logger")
        self.writer = Logger(name="all")
        self.writer_evaluation = Logger(name="evaluation")
        self.writer_inference = Logger(name="inference")
        
        self.build()

    #---- LOAD TASK
    def load_task(self):
        batch_size = self.config.config_training["batch_size"]
        self.train_loader = get_loader(dataset_config=self.config.config_dataset, batch_size=batch_size, split="train")
        self.val_loader = get_loader(dataset_config=self.config.config_dataset, batch_size=batch_size, split="val")
        self.test_loader = get_loader(dataset_config=self.config.config_dataset, batch_size=16, split="test")

    #---- REGISTER
    def build_registry(self):
        # Build writer
        registry.set_module("writer", name="common", instance=self.writer)
        registry.set_module("writer", name="evaluation", instance=self.writer_evaluation)
        registry.set_module("writer", name="inference", instance=self.writer_inference)
        
        # Build args
        registry.set_module("args", name=None, instance=self.args)

        # Build config
        self.config.build_registry()

    #---- BUILD
    def build(self):
        self.writer.LOG_INFO("=== START BUILDING ===")
        self.build_registry()
        self.build_model()
        self.load_task()
        self.build_training_params()

    def build_model(self):
        self.model = TransformerSummarizer()
        # check_requires_grad(self.model)
        # raise
        self.model.to(self.device)

    def build_training_params(self):
        self.max_epochs = self.config.config_training["epochs"]
        self.batch_size = self.config.config_training["batch_size"]
        self.max_iterations = self.config.config_training["max_iterations"]
        self.snapshot_interval = self.config.config_training["snapshot_interval"]
        self.snapshot_epoch_interval = self.config.config_training["snapshot_epoch_interval"]
        self.early_stop_patience = self.config.config_training["early_stopping"]["patience"]
        self.early_stop_counter = 0
        self.current_iteration = 0
        self.current_epoch = 0

        # Training
        self.optimizer = self.build_optimizer(
            model=self.model,
            config_optimizer=self.config.config_optimizer
        )
        self.loss_fn = self.build_loss()
        self.lr_scheduler = self.build_scheduler(
            optimizer=self.optimizer,
            config_lr_scheduler=self.config.config_training["lr_scheduler"]
        )

        # Resume training
        if self.args.resume_file != None:
            self.load_model(self.args.resume_file)


    def build_scheduler(self, optimizer, config_lr_scheduler):
        if not config_lr_scheduler["status"]:
            return None
        scheduler_func = lambda x: lr_lambda_update_epoch(x, config_lr_scheduler)
            
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=scheduler_func
        )
        return lr_scheduler


    def build_loss(self):
        # pad_idx = self.model.encoder_description.get_pad_token_id()
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        return loss_fn


    def build_optimizer(self, model, config_optimizer):
        if "type" not in config_optimizer:
            raise ValueError(
                "Optimizer attributes must have a 'type' key "
                "specifying the type of optimizer. "
                "(Custom or PyTorch)"
            )
        optimizer_type = config_optimizer["type"]

        #-- Load params
        if not hasattr(config_optimizer, "params"):
            warnings.warn(
                "optimizer attributes has no params defined, defaulting to {}."
            )
        optimizer_params = config_optimizer.get("params", {})
        ic(optimizer_params)
        ic(f"=== DEBUG: Config LR = {optimizer_params.get('lr', 'NOT FOUND')} ===")
        ic(optimizer_params)
        
        #-- Load optimizer class
        if not hasattr(torch.optim, optimizer_type):
            raise ValueError(
                "No optimizer found in torch.optim"
            )
        optimizer_class = getattr(torch.optim, optimizer_type)
        # parameters = get_optimizer_parameters(
        #     model=model, 
        #     config=config_optimizer
        # )
        parameters = model.parameters()
        optimizer = optimizer_class(parameters, **optimizer_params)
        return optimizer
    

    #---- STEP
    def _forward_pass(self, batch):
        """
            Forward to model
        """
        scores, pred_inds, gt_inds = self.model(batch)
        return scores, pred_inds, gt_inds


    def _extract_loss(self, scores, targets):
        """
            scores.shape: torch.Size([B, max_length, num_vocab])
            targets.shape: torch.Size([B, max_length])
        """
        num_vocab = scores.size(-1)
        scores_reshape = scores.reshape(-1, num_vocab)
        targets_reshape = targets.reshape(-1)
        loss_output = self.loss_fn(
            scores_reshape, targets_reshape
        ) 
        return loss_output
    
    def _gradient_clipping(self):
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

    def _backward(self, loss):
        """
            Backpropagation
        """
        self.optimizer.zero_grad()
        loss.backward()
        self._gradient_clipping()
        self.optimizer.step()
    

    def _run_scheduler(self):
        """
            Learning rate scheduler
        """
        self.lr_scheduler.step()

        # debug: log LRs (every 5 epochs or so)
        lrs = [pg['lr'] for pg in self.optimizer.param_groups]
        self.writer.LOG_INFO(f"Epoch {self.current_epoch} LRs: {lrs}")

    #---- MODE
    def match_device(self, batch):
        return batch

    def preprocess_batch(self, batch):
        """
            Function:
                - Padding batch features to the same length
        """
        
        return batch


    #---- MODE
    def train(self):
        self.writer.LOG_INFO("=== Model ===")
        self.writer.LOG_INFO(self.model)

        self.writer.LOG_INFO("Starting training...")
        self.model.train()
        
        best_scores = -1
        while self.current_epoch <= self.max_epochs:
            self.current_epoch += 1
            self.writer.LOG_INFO(f"Training epoch: {self.current_epoch}")
            for batch_id, batch in tqdm(enumerate(self.train_loader), desc="Iterating through train loader"):
                self.writer.LOG_INFO(f"Training epoch: {self.current_epoch} - Batch: {batch_id}")

                batch = self.preprocess_batch(batch)
                batch = self.match_device(batch)

                self.current_iteration += 1
                #~ Loss cal - For training so it is caption_ids
                scores_output, caption_inds, target_inds = self._forward_pass(batch)
                loss = self._extract_loss(scores_output, target_inds)
                self._backward(loss)

            #-- Run scheduler
            self._run_scheduler()
            if self.current_epoch % self.snapshot_epoch_interval == 0:
                _, _, val_final_scores, loss = self.evaluate(epoch_id=self.current_epoch, split="val")
                if val_final_scores["CIDEr"] > best_scores:
                    self.early_stop_counter = 0
                    best_scores = val_final_scores["CIDEr"]
                    self.save_model(
                        model=self.model,
                        loss=loss,
                        optimizer=self.optimizer,
                        lr_scheduler=self.lr_scheduler,
                        epoch=self.current_epoch, 
                        iteration=self.current_iteration,
                        metric_score=best_scores,
                        use_name="best"
                    )
                else:
                    self.early_stop_counter += 1

                if self.early_stop_counter >= self.early_stop_patience:
                    self.writer.LOG_INFO("Early stopping triggered.")
                    break
            self.save_model(
                model=self.model,
                loss=loss,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                epoch=self.current_epoch, 
                iteration=self.current_iteration,
                metric_score=best_scores,
                use_name="last"
            )
                    
                    
    
    def evaluate(self, epoch_id=None, split="val"):
        if hasattr(self, f"{split}_loader"):
            dataloader = getattr(self, f"{split}_loader")
        else:
            self.writer_evaluation.LOG_ERROR(f"No dataloader for {split} split")
            raise ModuleNotFoundError
        
        with torch.no_grad():
            hypo: dict = {}
            ref : dict = {}
            losses = []
            self.model.eval()

            for batch_id, batch in tqdm(enumerate(dataloader), desc=f"Evaluating {split} loader"):
                self.writer_evaluation.LOG_INFO(f"Evaluate batch: {batch_id + 1}")
                list_id = batch["list_id"]
                list_captions = batch["list_captions"]
                batch = self.preprocess_batch(batch)
                batch = self.match_device(batch)
                
                #~ Calculate loss
                scores_output, pred_inds, target_inds = self._forward_pass(batch)
                loss = self._extract_loss(scores_output, target_inds)
                loss_scalar = loss.detach().cpu().item()
                losses.append(loss_scalar)
                
                #~ Metrics calculation
                if not epoch_id==None:
                    self.writer_evaluation.LOG_INFO(f"Logging at iteration: {epoch_id}")
                
                pred_caps = self.get_pred_captions(pred_inds)
                for id, pred_cap, ref_cap in zip(list_id, pred_caps, list_captions):
                    hypo[id] = [pred_cap]
                    ref[id]  = [ref_cap]
            
            # Calculate Metrics
            final_scores = metric_calculate(ref, hypo)
            avg_loss = sum(losses) / len(losses) 
            self.writer_evaluation.LOG_INFO(f"|| Metrics Calculation || {split} split || epoch: {epoch_id} || loss: {avg_loss}")
            self.writer_evaluation.LOG_INFO(f"Final scores:\n{final_scores}")
            
            # Turn on train mode to continue training
            self.model.train()
        return hypo, ref, final_scores, avg_loss


    def inference(self, mode, save_dir):
        """
            Parameters:
                mode:   Model to run "val" or "test"
        """
        hypo, ref = {}, {}
        if mode=="val":
            self.writer_inference.LOG_INFO("=== Inference Validation Split ===")
            hypo, ref, _, _ = self.evaluate(epoch_id="Inference val set", split="val")
        elif mode=="test":
            self.writer_inference.LOG_INFO("=== Inference Test Split ===")
            hypo, ref, _, _ = self.evaluate(epoch_id="Inference test set", split="test")
        else:
            self.writer_inference.LOG_ERROR(f"No mode available for {mode}")
        
        # Save Inference
        self.save_inference(
            hypo=hypo,
            ref=ref,
            save_dir=save_dir,
            name=mode
        )
        return hypo, ref

    #---- FINISH
    def save_model(self, model, loss, optimizer, lr_scheduler, epoch, iteration, metric_score, use_name=""):
        if not os.path.exists(self.args.save_dir):
            self.writer.LOG_INFO("Save dir not exist")
            os.makedirs(self.args.save_dir, exist_ok=True)
        
        model_path = os.path.join(self.args.save_dir, f"checkpoints/model_{use_name}.pth")
        self.writer.LOG_DEBUG(f"Model save at {model_path}")

        torch.save({
            'epoch': epoch,
            "iteration": iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': getattr(lr_scheduler, 'state_dict', lambda: None)(),
            'loss': loss,
            "metric_score": metric_score
        }, model_path)

    
    def load_model(self, model_path):
        # ic(self.device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        #### DEBUG
        for param_group in self.optimizer.param_groups:
            ic(param_group['lr'])
        #### DEBUG
        
        self.current_epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        if self.lr_scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
            try:
                self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception as e:
                # fallback: advance scheduler to current_iteration
                self.writer.LOG_INFO("Warning: failed to load scheduler state; advancing scheduler to iteration")
                self.lr_scheduler.step(self.current_iteration)
        else:
            # ensure scheduler is aligned with optimizer param_group's lrs
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(self.current_iteration)

        self.writer.LOG_INFO(f"=== Load model at epoch: {self.current_epoch} || loss: {loss} ===")

    #---- METRIC CALCULATION
    def get_pred_captions(self, pred_inds):
        """
            Predict batch
        """
        # Captioning
        captions_pred = self.model.encoder_summary.batch_decode(pred_inds)
        return captions_pred # BS, 
    

    def save_inference(self, hypo, ref, save_dir, name=""):
        save_path = os.path.join(save_dir, f"{name}_reference")
        inference: dict = {
            id : {
                "gt": ref[id],
                "pred": hypo[id],
            } for id in ref.keys()
        }
        save_json(save_path, content=inference)

