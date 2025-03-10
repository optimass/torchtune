# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
import sys
import time

from functools import partial
from typing import Any, Dict, Optional, Tuple
from warnings import warn
import random

import torch
from omegaconf import DictConfig, ListConfig

from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from torchtune import config, modules, rlhf, training, utils
from torchtune.data import CROSS_ENTROPY_IGNORE_IDX, padded_collate_dpo, padded_collate_traj_dpo
from torchtune.datasets import ConcatDataset
from torchtune.modules.peft import (
    disable_adapter,
    get_adapter_params,
    get_adapter_state_dict,
    get_merged_lora_ckpt,
    set_trainable_params,
    validate_missing_and_unexpected_for_lora,
)
from torchtune.recipe_interfaces import FTRecipeInterface

from torchtune.rlhf.loss import SimPOLoss
from torchtune.modules.loss import CEWithChunkedOutputLoss
from tqdm import tqdm
import torch.nn.functional as F

log = utils.get_logger("DEBUG")


class LoRADPORecipeSingleDevice(FTRecipeInterface):
    """
    LoRA DPO recipe for dense transformer-based LLMs such as Llama2 for
    single device training. This is based on HF's DPOTrainer in the
    TRL library: https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py#L65

    This recipe supports:
        - Activation checkpointing. This is enabled by default but is configurable.
        - Full bf16 training for supported HW architectures. We currently check bf16 support via
        the `torch.cuda.is_bf16_supported` API. This is disabled by default but can be enabled via
        setting `dtype=bf16` in configuration.
        - Checkpointing: of LoRA adapter parameters and their optimizer states. When resuming
            from a checkpoint, the adapter parameters are loaded from the checkpoint along
            with the base model weights. Note that intra-epoch resumption is not supported.
        - Logging to terminal, WandB, or TensorBoard.


    The following losses are supported in this recipe:
        - :class:`~torchtune.rlhf.loss.DPOLoss`: Direct Preference Optimization (DPO).
        - :class:`~torchtune.rlhf.loss.RSOPLoss`: Rejection Sampling Optimization (RSO).
        - :class:`~torchtune.rlhf.loss.SimPOLoss`: Simple Preference Optimization (SimPO).

    Assumptions:
        - Checkpoints are ONLY saved at epoch boundaries. In case of failure, work done
            in ongoing epoch is lost.
        - Datasets are Map-style and data fits in memory (not streamed).

    The following configs can be used to run this recipe:
        >>> tune ls
        RECIPE                          CONFIG
        lora_dpo_single_device          llama2/7B_lora_dpo_single_device

    Args:
        cfg (DictConfig): OmegaConf object parsed from yaml file

    Raises:
        ValueError: If ``dtype`` is set to fp16.
        RuntimeError: If ``dtype`` is set to bf16 and the hardware does not support bf16.

    """

    def __init__(self, cfg: DictConfig) -> None:

        self._device = utils.get_device(device=cfg.device)
        # Reduced precision logic
        self._dtype = training.get_dtype(cfg.dtype, device=self._device)

        # fp16 precision is explicitly disabled as it is not supported in this
        # recipe (for example, no gradient scaling).
        if self._dtype == torch.float16:
            raise ValueError(
                "fp16 precision is not supported in this recipe. Please use fp32 or bf16."
            )

        # logging attributes
        self._output_dir = cfg.output_dir
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self._log_peak_memory_stats = cfg.get("log_peak_memory_stats", False)
        self.ce_loss=CEWithChunkedOutputLoss(num_output_chunks=6)
        self.reg_lambda=cfg.reg_lambda
        if self._log_peak_memory_stats and self._device.type != "cuda":
            log.info(
                "log_peak_memory_stats was set to True, however, training does not use cuda. Setting log_peak_memory_stats=False."
            )
            self._log_peak_memory_stats = False
            # activation checkpointing/offloading
        self._enable_activation_checkpointing = cfg.get(
            "enable_activation_checkpointing", False
        )
        self._enable_activation_offloading = cfg.get(
            "enable_activation_offloading", False
        )
        if self._enable_activation_offloading:
            if self._device.type != "cuda":
                raise RuntimeError(
                    "enable_activation_offloading should only be True when training on CUDA"
                )
            if not self._enable_activation_checkpointing:
                raise RuntimeError(
                    "enable_activation_offloading should only be True when enable_activation_checkpointing is True"
                )
        elif self._enable_activation_checkpointing:
            utils.log_rank_zero(
                log,
                "Hint: enable_activation_checkpointing is True, but enable_activation_offloading isn't. "
                "Enabling activation offloading should reduce memory further.",
            )
        # These are public properties which are updated by the checkpoint loader
        # when ``resume_from_checkpoint`` is `True` or validated in tests
        self.seed = training.set_seed(seed=cfg.seed)
        self.epochs_run = 0
        self.total_epochs = cfg.epochs
        self.max_steps_per_epoch = cfg.max_steps_per_epoch
        self.global_step = 0
        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._save_adapter_weights_only = cfg.get("save_adapter_weights_only", False)
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps

        # NOTE: added by us
        self.save_checkpoints_interval = cfg.get("save_checkpoints", 1)
        self.max_seq_len = cfg.get("max_seq_len", None)
        self._max_validation_steps = int(
            cfg.get("samples_per_validation_steps") / cfg.batch_size
        )
        log.info(
            f"Setting max validation steps to {self._max_validation_steps} (samples_per_validation_steps / batch_size)"
        )
        assert self.max_steps_per_epoch is None
        effective_batch_size = cfg.batch_size * cfg.gradient_accumulation_steps
        self.max_steps_per_epoch = int(
            cfg.get("samples_per_epoch") / effective_batch_size
        )
        log.info(
            f"Setting max steps per epoch to {self.max_steps_per_epoch} (samples_per_epoch / effective_batch_size)"
        )

    def load_checkpoint(self, cfg_checkpointer: DictConfig) -> Dict[str, Any]:
        """
        Extract the checkpoint state from file and validate. This includes the
        base model weights. If resume_from_checkpoint is True, this also includes
        the adapter weights and recipe state
        """
        self._checkpointer = config.instantiate(
            cfg_checkpointer,
            resume_from_checkpoint=self._resume_from_checkpoint,
        )
        checkpoint_dict = self._checkpointer.load_checkpoint()

        if self._resume_from_checkpoint:
            if training.ADAPTER_KEY not in checkpoint_dict:
                raise ValueError(
                    "Adapter weights not found. Please ensure a valid adapter checkpoint is provided."
                )
            # _update_recipe_state will throw an exception if the recipe state is not correctly loaded
            # no need to check here
            self._update_recipe_state(checkpoint_dict)
        return checkpoint_dict

    def _update_recipe_state(self, ckpt_dict: Dict[str, Any]) -> None:
        """
        Updates the recipe state from checkpoint.
        """
        try:
            self.epochs_run = ckpt_dict[training.EPOCHS_KEY]

            # on mismatch, warn the user and prevent the override
            if self.seed != ckpt_dict[training.SEED_KEY]:
                warn(
                    message=(
                        "Config value for seed does not match the checkpoint value, "
                        f"using the checkpoint value: {ckpt_dict[training.SEED_KEY]}"
                    )
                )
                self.seed = ckpt_dict[training.SEED_KEY]
            if self.max_steps_per_epoch != ckpt_dict[training.MAX_STEPS_KEY]:
                warn(
                    message=(
                        "Config value for max_steps_per_epoch does not match the checkpoint value, "
                        f"using the checkpoint value: {ckpt_dict[training.MAX_STEPS_KEY]}"
                    )
                )
                self.max_steps_per_epoch = ckpt_dict[training.MAX_STEPS_KEY]

            # on mismatch, warn the user but allow the override
            if self.total_epochs != ckpt_dict[training.TOTAL_EPOCHS_KEY]:
                warn(
                    message=(
                        "Config value for total_epochs does not match the checkpoint value, "
                        f"using the config value: {self.total_epochs}"
                    )
                )

        except KeyError as e:
            raise KeyError(
                "Checkpoint does not contain the required keys needed for updating recipe state. "
                "Are you sure you passed in the right recipe checkpoint?"
            ) from e

    def setup(self, cfg: DictConfig) -> None:
        """
        Setup the recipe state. This includes recipe state (if resume_from_checkpoint is True),
        model, tokenizer, loss, optimizer, learning rate scheduler, sampler, and dataloader.
        """
        self._metric_logger = config.instantiate(cfg.metric_logger)

        # log config with parameter override
        self._metric_logger.log_config(cfg)

        self._model_compile = cfg.compile
        checkpoint_dict = self.load_checkpoint(cfg_checkpointer=cfg.checkpointer)

        self._model = self._setup_model(
            cfg_model=cfg.model,
            enable_activation_checkpointing=cfg.enable_activation_checkpointing,
            enable_activation_offloading=self._enable_activation_offloading,
            compile_model=cfg.compile,
            base_model_state_dict=checkpoint_dict[training.MODEL_KEY],
            lora_weights_state_dict=(
                checkpoint_dict[training.ADAPTER_KEY]
                if self._resume_from_checkpoint
                else None
            ),
        )

        self._tokenizer = config.instantiate(cfg.tokenizer)
        log.info("Tokenizer is initialized from file.")

        self._optimizer = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            opt_state_dict=(
                checkpoint_dict[training.OPT_KEY]
                if self._resume_from_checkpoint
                else None
            ),
        )

        self._loss_fn = config.instantiate(cfg.loss)
        log.info("Loss function is initialized.")

        # NOTE: no collate_func in this recipe

        # Dataloader depends on the tokenizer and loss_fn and should be
        # setup after all of these are setup
        cfg.dataset["split"] = "train"  # NOTE: added by us
        self._sampler, self._dataloader = self._setup_data(
            cfg_dataset=cfg.dataset,
            shuffle=cfg.shuffle,
            batch_size=cfg.batch_size,
        )

        # NOTE: added by us
        # validation dataloader
        cfg["validation_dataset"] = deepcopy(cfg.dataset)
        cfg["validation_dataset"]["split"] = "validation"
        self._sampler_validation, self._dataloader_validation = self._setup_data(
            cfg_dataset=cfg["validation_dataset"],
            shuffle=cfg.shuffle,
            batch_size=cfg.batch_size,  # TODO: have a separate batch size for validation
        )

        # Finally update the recipe state which can only be correctly set after all of the
        # other components have been initialized and updated.

        # Number of training steps in each epoch depends on the number of batches produced
        # by the dataloader and the max_steps_per_epoch param set by the user and is used
        # for logging and tracking training state. This should be computed after the dataloader
        # has been setup
        self._steps_per_epoch = (
            len(self._dataloader) // self._gradient_accumulation_steps
        )
        if (
            self.max_steps_per_epoch is not None
            and self.max_steps_per_epoch < self._steps_per_epoch
        ):
            self._steps_per_epoch = self.max_steps_per_epoch
            self.global_step = self.epochs_run * self._steps_per_epoch

        # Learning rate scheduler can only be set up after number of steps
        # has been computed
        self._lr_scheduler = self._setup_lr_scheduler(
            cfg_lr_scheduler=cfg.get("lr_scheduler", None),
            num_training_steps=self.total_epochs * self._steps_per_epoch,
            last_epoch=self.global_step - 1,
        )

    def _setup_model(
        self,
        cfg_model: DictConfig,
        enable_activation_checkpointing: bool,
        enable_activation_offloading: bool,
        compile_model: bool,
        base_model_state_dict: Dict[str, Any],
        lora_weights_state_dict: Optional[Dict[str, Any]] = None,
    ) -> nn.Module:
        with training.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(cfg_model)
        self._lora_rank = cfg_model.lora_rank
        self._lora_alpha = cfg_model.lora_alpha
        self._lora_attn_modules = list(cfg_model.lora_attn_modules)
        self._apply_lora_to_mlp = cfg_model.apply_lora_to_mlp
        self._apply_lora_to_output = getattr(cfg_model, "apply_lora_to_output", False)
        self.adapter_params = get_adapter_params(model)
        set_trainable_params(model, self.adapter_params)

        if enable_activation_checkpointing:
            training.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerSelfAttentionLayer}
            )

        base_missing, base_unexpected = model.load_state_dict(
            base_model_state_dict, strict=False
        )
        if lora_weights_state_dict:
            lora_missing, lora_unexpected = model.load_state_dict(
                lora_weights_state_dict, strict=False
            )
        else:
            lora_missing, lora_unexpected = None, None
        validate_missing_and_unexpected_for_lora(
            lora_attn_modules=self._lora_attn_modules,
            apply_lora_to_mlp=self._apply_lora_to_mlp,
            apply_lora_to_output=self._apply_lora_to_output,
            base_missing=base_missing,
            base_unexpected=base_unexpected,
            lora_missing=lora_missing,
            lora_unexpected=lora_unexpected,
        )
        # activation offloading
        self.activations_handling_ctx = training.get_act_offloading_ctx_manager(
            model, enable_activation_offloading
        )

        log.info(f"Model is initialized with precision {self._dtype}.")

        # Compile model, if enabled.
        if compile_model:
            training.compile_model(model)
        if self._device == torch.device("cuda"):
            memory_stats = training.get_memory_stats(device=self._device)
            training.log_memory_stats(memory_stats)
        return model

    def _setup_optimizer(
        self, cfg_optimizer: DictConfig, opt_state_dict: Optional[Dict[str, Any]] = None
    ) -> Optimizer:
        optimizer = config.instantiate(cfg_optimizer, self._model.parameters())
        if opt_state_dict:
            optimizer.load_state_dict(opt_state_dict)

        log.info("Optimizer and loss are initialized.")
        return optimizer

    def _setup_lr_scheduler(
        self,
        cfg_lr_scheduler: Optional[DictConfig],
        num_training_steps: int,
        last_epoch: int,
    ) -> Optional[Optimizer]:
        if cfg_lr_scheduler is None:
            log.info(
                "No learning rate scheduler configured. Using constant learning rate."
            )
            return None

        lr_scheduler = config.instantiate(
            cfg_lr_scheduler,
            self._optimizer,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )

        log.info("Learning rate scheduler is initialized.")
        return lr_scheduler

    def _setup_data(
        self,
        cfg_dataset: DictConfig,
        shuffle: bool,
        batch_size: int,
    ) -> Tuple[DistributedSampler, DataLoader]:
        """
        All data related setup happens here. Currently this recipe only supports
        Map-style Datasets which fit into memory and an option for random shuffling.
        Samplers, iterable datasets, and streaming datasets are not supported.
        """
        if isinstance(cfg_dataset, ListConfig):
            datasets = [
                config.instantiate(single_cfg_dataset, tokenizer=self._tokenizer)
                for single_cfg_dataset in cfg_dataset
            ]
            ds = ConcatDataset(datasets=datasets)
        else:
            ds = config.instantiate(cfg_dataset, tokenizer=self._tokenizer)

        sampler = DistributedSampler(
            ds,
            num_replicas=1,
            rank=0,
            shuffle=shuffle,
            seed=0,
        )
        dataloader = DataLoader(
            dataset=ds,
            sampler=sampler,
            batch_size=batch_size,
            # dropping last avoids shape issues with compile + flex attention
            drop_last=True,
            collate_fn=partial(
                padded_collate_traj_dpo,
                padding_idx=self._tokenizer.pad_id,
                ignore_idx=CROSS_ENTROPY_IGNORE_IDX,
            ),
        )
        log.info("Dataset and Sampler are initialized.")

        return sampler, dataloader

    def save_checkpoint(self, epoch: int) -> None:
        """
        Checkpoint the state of the recipe. The constructed checkpoint state dict
        contains the following information:
        - Merged weights with key MODEL_KEY
        - Adapter weights with key ADAPTER_KEY
        - Relevant recipe state if training is not complete
        - If the `self._save_adapter_weights_only` option is True, the checkpointer will save only the adapter weights

        To correctly resume from training, the adapter weights and recipe state must be provided along with the base model weights.
        """
        # NOTE: added by us
        if self.save_checkpoints_interval:
            if epoch + 1 == self.total_epochs:
                pass
            elif (
                self.save_checkpoints_interval > 0
                and epoch % self.save_checkpoints_interval == 0
            ):
                pass
            else:
                return
        else:
            return
        ckpt_dict = {}

        intermediate_checkpoint = epoch + 1 < self.total_epochs
        # if training is in-progress, checkpoint the optimizer state as well
        if intermediate_checkpoint:
            ckpt_dict.update(
                {
                    training.OPT_KEY: self._optimizer.state_dict(),
                    training.SEED_KEY: self.seed,
                    training.EPOCHS_KEY: self.epochs_run,
                    training.TOTAL_EPOCHS_KEY: self.total_epochs,
                    training.MAX_STEPS_KEY: self.max_steps_per_epoch,
                }
            )

        adapter_state_dict = get_adapter_state_dict(self._model.state_dict())
        ckpt_dict.update({training.ADAPTER_KEY: adapter_state_dict})
        if not self._save_adapter_weights_only:
            # Construct the full state dict with LoRA weights merged into base LLM weights

            # Move to CPU to avoid a copy on GPU
            state_dict = {k: v.cpu() for k, v in self._model.state_dict().items()}

            merged_state_dict = get_merged_lora_ckpt(
                state_dict,
                rank=self._lora_rank,
                alpha=self._lora_alpha,
            )

            ckpt_dict.update({training.MODEL_KEY: merged_state_dict})

        self._checkpointer.save_checkpoint(
            ckpt_dict,
            epoch=epoch,
            intermediate_checkpoint=intermediate_checkpoint,
            adapter_only=self._save_adapter_weights_only,
        )

    def concatenated_forward(
        self, 
        model: nn.Module, 
        input_ids, 
        labels
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
    Run forward pass of the model with chosen and rejected samples concatenated.
    
    Args:
        model (nn.Module): The model to be used for the forward pass.
        input_ids: Input token IDs
        labels: Corresponding labels
    
    Returns:
        Tuple of log probs and logits
    """
        concatenated_input_ids = input_ids.to(self._device).unsqueeze(0)
        concatenated_labels = labels.to(self._device).unsqueeze(0)
    
        
    
        with self.activations_handling_ctx:
            all_logits = model(concatenated_input_ids)

        all_log_probs = rlhf.get_batch_log_probs(logits=all_logits, labels=concatenated_labels, return_average_logprobs=True)
    
        return (all_log_probs, all_logits)

    # NOTE: added by us
    def _skip_max_seq_len_samples(self, input_ids):
        max_inp=0
        for inp in input_ids:
            if len(inp)>max_inp:
                max_inp=len(inp)
        
        if max_inp>5000:
            return True
        else:
            return False

    def train(self) -> None:
        """
        The core training loop.
        """
        if self._model_compile:
            log.info(
                "NOTE: torch.compile is enabled and model is compiled in first forward. Expect a relatively slow first iteration."
            )

        # Initialize tokens count and running loss (for grad accumulation)
        t0 = time.perf_counter()

        # self.epochs_run should be non-zero when we're resuming from a checkpoint
        for curr_epoch in range(self.epochs_run, self.total_epochs):
            # Update the sampler to ensure data is correctly shuffled across epochs
            # in case shuffle is True
            self._sampler.set_epoch(curr_epoch)
            self._sampler_validation.set_epoch(curr_epoch)  # NOTE: added by us

            # NOTE: added by us
            # ------ Validation Step ------ #
            self._model.eval()

            with torch.no_grad():
                running_val_loss = 0
                running_reward_accuracy = 0

                num_eval_steps = (
                    min(self._max_validation_steps, len(self._dataloader_validation))
                    if self._max_validation_steps is not None
                    else len(self._dataloader_validation)
                )
                # NOTE: added by us
                # start a counter for samples that are too long
                max_len_samples = 0

                pbar_val = tqdm(total=num_eval_steps, desc="Validation")
                # NOTE: added by us - counter to account for samples that are too long
                idx = 0

                policy_chosen_sum = torch.zeros(1, device=self._device)
                policy_rejected_sum = torch.zeros(1, device=self._device)
                reference_chosen_sum = torch.zeros(1, device=self._device)
                reference_rejected_sum = torch.zeros(1, device=self._device)

                for _, batch in enumerate(self._dataloader_validation):
                    if self._max_validation_steps is not None and idx == self._max_validation_steps:
                        break

                    input_ids, labels, ratio = batch
                    if self._skip_max_seq_len_samples(input_ids):
                        max_len_samples += 1
                        continue

                    policy_chosen_sum.zero_()
                    policy_rejected_sum.zero_()
                    reference_chosen_sum.zero_()
                    reference_rejected_sum.zero_()

                    reg_index=random.randint(0,len(input_ids)-1)

                    for index in range(len(input_ids)):
                        log_policy_probs, policy_logits = self.concatenated_forward(
                            self._model, input_ids[index], labels[index]
                        )
                        if index==reg_index:
                            sft_policy_logits=policy_logits
                            sft_policy_labels=labels[index]
                        del policy_logits

                        with torch.no_grad(), disable_adapter(self._model):
                            reference_log_probs, reference_logits = self.concatenated_forward(
                                self._model, input_ids[index], labels[index]
                            )

                            del reference_logits

                        if index < ratio[0]:
                            policy_chosen_sum += log_policy_probs
                            reference_chosen_sum += reference_log_probs
                        else:
                            policy_rejected_sum += log_policy_probs
                            reference_rejected_sum += reference_log_probs

                    loss, chosen_rewards, rejected_rewards = self._loss_fn(
                            policy_chosen_sum,
                            policy_rejected_sum,
                            reference_chosen_sum,
                            reference_rejected_sum,
                    )

                    

                    logits_chunks = sft_policy_logits.chunk(6, dim=1)
                    labels_=torch.hstack((sft_policy_labels[1:], torch.tensor([-100], device=sft_policy_labels.device)))
                    labels_=labels_.unsqueeze(0).to(self._device)

                    ce_loss = self.ce_loss(logits_chunks, labels_)
 
                    del sft_policy_labels, sft_policy_logits, logits_chunks, labels_
                    ce_loss=ce_loss.mean()
                    loss = loss.mean()
                    reward_accuracy = (chosen_rewards > rejected_rewards).float().mean().cpu()
 
                    running_val_loss += (loss + self.reg_lambda*ce_loss) 
                    running_reward_accuracy += reward_accuracy

                    pbar_val.update(1)
                    pbar_val.set_description(
                        f"{self.epochs_run+1}|{self.global_step}|Validation Loss: {running_val_loss / (idx + 1)}"
                    )
                    idx += 1

                mean_val_loss = running_val_loss / (idx + 1)
                mean_reward_accuracy = running_reward_accuracy / (idx + 1)

                self._metric_logger.log_dict(
                    {
                        "val_loss": mean_val_loss,
                        "val_reward_accuracies": mean_reward_accuracy,
                    },
                    step=self.global_step,
                )

                pbar_val.close()
                print("Number of samples that were too long:", max_len_samples)


            # ------ Training Epoch ------ #
            # Initialize tokens count and running loss (for grad accumulation)
            t0 = time.perf_counter()
            running_loss = 0
            positive_num_tokens = 0
            negative_num_tokens=0
            max_len_samples = 0
            self._model.train()  # NOTE: added by us

            pbar = tqdm(total=self._steps_per_epoch, desc="Training")
            # NOTE: added by us - counter to account for samples that are too long
            positive_trajectory_length = 0
            negative_trajectory_length = 0
            idx = 0
            for _, batch in enumerate(self._dataloader):
                if (
                    self.max_steps_per_epoch is not None
                    and (idx // self._gradient_accumulation_steps)
                    == self.max_steps_per_epoch
                ):
                    break

                input_ids, labels, ratio = batch

                if self._skip_max_seq_len_samples(input_ids):
                    max_len_samples += 1
                    continue
                policy_chosen_sum = torch.zeros(1, device=self._device)
                policy_rejected_sum = torch.zeros(1, device=self._device)
                reference_chosen_sum = torch.zeros(1, device=self._device)
                reference_rejected_sum = torch.zeros(1, device=self._device)

                
                positive_trajectory_length += ratio[0]
                negative_trajectory_length += ratio[1]


                # batch is input_ids, labels
                reg_index=random.randint(0,len(input_ids)-1)

                for index in range(len(input_ids)):

                    log_policy_probs, policy_logits = self.concatenated_forward(
                        self._model, input_ids[index], labels[index]
                    )
                    if index==reg_index:
                        sft_policy_logits=policy_logits
                        sft_policy_labels=labels[index]
                    del policy_logits

                    with torch.no_grad(), disable_adapter(self._model):
                        reference_log_probs, reference_logits = self.concatenated_forward(
                            self._model, input_ids[index], labels[index]
                        )
                        del reference_logits

                    if index < ratio[0]:
                        positive_num_tokens += input_ids[index].numel()
                        policy_chosen_sum += log_policy_probs
                        reference_chosen_sum += reference_log_probs
                    else:
                        negative_num_tokens += input_ids[index].numel()
                        policy_rejected_sum += log_policy_probs
                        reference_rejected_sum += reference_log_probs

                loss, chosen_rewards, rejected_rewards = self._loss_fn(
                    policy_chosen_sum,
                    policy_rejected_sum,
                    reference_chosen_sum,
                    reference_rejected_sum,
                )

                logits_chunks = sft_policy_logits.chunk(6, dim=1)
                labels_=torch.hstack((sft_policy_labels[1:], torch.tensor([-100], device=sft_policy_labels.device)))
                labels_=labels_.unsqueeze(0).to(self._device)
                torch.cuda.empty_cache()

                ce_loss = self.ce_loss(logits_chunks, labels_)
 
                del sft_policy_labels, sft_policy_logits, logits_chunks, labels_
                torch.cuda.empty_cache()

                loss = loss.mean()
                ce_loss=ce_loss.mean()
                reward_accuracies = (chosen_rewards > rejected_rewards).float()

                loss = loss / self._gradient_accumulation_steps
                ce_loss=ce_loss / self._gradient_accumulation_steps
                loss=loss+self.reg_lambda*ce_loss
                running_loss += loss
                loss.backward()

                # Step with optimizer
                if (idx + 1) % self._gradient_accumulation_steps == 0:
                    self._optimizer.step()
                    self._optimizer.zero_grad(set_to_none=True)

                    if self._lr_scheduler is not None:
                        self._lr_scheduler.step()
                    # Update the number of steps when the weights are updated
                    self.global_step += 1

                    loss_to_log = running_loss.item()
                    pbar.update(1)
                    pbar.set_description(
                        f"{curr_epoch + 1}|{self.global_step}|Loss: {loss_to_log}"
                    )

                    avg_positive_length = positive_trajectory_length / self._gradient_accumulation_steps
                    avg_negative_length = negative_trajectory_length / self._gradient_accumulation_steps

                    # Log per-step metrics
                    if self.global_step % self._log_every_n_steps == 0:
                        time_per_step = time.perf_counter() - t0
                        log_dict = {
                            "loss": loss_to_log,
                            "idx": index,
                            "positive_trajectory_length": avg_positive_length,
                            "negative_trajectory_length": avg_negative_length,
                            "lr": self._optimizer.param_groups[0]["lr"],
                            "positive_tokens_per_second_per_gpu": positive_num_tokens / time_per_step,
                            "negative_tokens_per_second_per_gpu": negative_num_tokens / time_per_step,
                            "rewards/chosen": chosen_rewards.mean().cpu(),
                            "rewards/rejected": rejected_rewards.mean().cpu(),
                            "rewards/accuracies": reward_accuracies.mean().cpu(),
                            "rewards/margins": (chosen_rewards - rejected_rewards).mean().cpu(),
                            # "log_probs/rejected": policy_rejected_sum.detach().mean().cpu(),
                            # "log_probs/chosen": policy_chosen_sum.detach().mean().cpu(),
                        }
                        if self._log_peak_memory_stats:
                            log_dict.update(
                                training.get_memory_stats(device=self._device)
                            )
                        self._metric_logger.log_dict(
                            log_dict,
                            step=self.global_step,
                        )

                    # Reset running stats for the next step
                    running_loss = 0
                    positive_num_tokens = 0
                    negative_num_tokens=0
                    t0 = time.perf_counter()

                idx += 1  # NOTE: added by us

            self.epochs_run += 1
            self.save_checkpoint(epoch=curr_epoch)

    def cleanup(self) -> None:
        self._metric_logger.close()


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """
    Entry point for the recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in config (see available configs through ``tune ls``)
        - Overwritten by arguments from the command-line
    """
    config.log_config(recipe_name="LoRADPORecipeSingleDevice", cfg=cfg)
    recipe = LoRADPORecipeSingleDevice(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())