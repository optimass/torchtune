# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
import sys
import time
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union
from warnings import warn
import re
import torch
import numpy as np
from omegaconf import DictConfig, ListConfig

from torch import nn
from torch.distributed import destroy_process_group, init_process_group

from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler, Subset
from full_finetune_distributed import FullFinetuneRecipeDistributed

from torchtune import config, modules, training, utils
from torchtune.config._utils import _get_component_from_path
from torchtune.data import padded_collate_packed
from torchtune.datasets import ConcatDataset
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.training import DummyProfiler, PROFILER_KEY
from torchtune.training.activations import apply_selective_activation_checkpointing
from torchtune.training.lr_schedulers import get_lr
from omegaconf import OmegaConf
import pprint
import os
import re
import os
import debugpy
import random
from tqdm import tqdm
import gc
from collections import defaultdict

# Add debugpy support for remote debugging

# Check if debugging should be enabled via environment variable
debug_enabled = os.environ.get("ENABLE_DEBUGPY", "0").lower() in ("1", "true", "yes")

debug_port = 5678


# if debug_enabled:
#     # Allow remote connections
#     debugpy.listen(("0.0.0.0", debug_port))
#     print(f"🐞 Debugpy listening on port {debug_port}")

#     print(f"⏳ Waiting for debugger to attach on port {debug_port}...")
#     debugpy.wait_for_client()
#     print("🔗 Debugger attached!")


log = utils.get_logger("DEBUG")


class FullFinetuneRecipeDistributedPrivalaged(FullFinetuneRecipeDistributed):
    """
    Full finetuning recipe for dense transformer-based LLMs such as Llama2. This recipe supports
    distributed training and can be run on a single node (1 to 8 GPUs).

    Features:
        - FSDP. Supported using PyTorch's FSDP APIs. CPU offload of parameters, gradients, and optimizer states
            is supported via ``fsdp_cpu_offload``. Resharding of parameters after the forward pass is
            done by default (corresponding to FULL_SHARD sharding strategy), but can be disabled by setting the config
            ``fsdp_reshard_after_forward`` to False (this corresponds to SHARD_GRAD_OP sharding strategy).
            DDP is currently not supported. Training on CPU is not supported.

        - Activation Checkpointing. This can be controlled using the ``enable_activation_checkpointing``
            flag. Activation checkpointing helps reduce the memory footprint since we no longer keep
            activations in memory and instead recompute them during the backward pass. This is especially
            helpful for larger batch sizes when you're memory constrained. But these savings in memory
            come at the cost of training performance. In most cases training can slow-down quite a bit as
            a result of this activation recomputation.

        - Activation Offloading. This can be controlled using the ``enable_activation_offloading``
            flag. Activation offloading is a technique similar to activations checkpointing that helps
            reduce the memory footprint to prevent OOMs on CUDA and enable bigger batches. Where activations
            checkpointing drops the activation in the forward to recompute it later in the backward,
            activations offloading will drop the activation in the forward to the CPU and bring it
            back during the backward pass. As always, there is a tradeoff--these savings in memory can
            come at the cost of training performance and CPU resources. To recover some runtime cost,
            we've added an option to enable offloading on a different stream to permit overlapping with
            the computation. This option is currently only available on PyTorch 2.5 or later and will
            be enabled by default if an acceptable torch version is found. Activation offloading can be
            used in conjunction with activation checkpointing.

        - Precision. Full fp32 and bf16 training are supported. Precision is controlled using the ``dtype``
            flag. When ``dtype=bf16``, all activations, gradients and optimizer states are in bfloat16. In
            most cases this should halve the memory footprint of full precision (fp32) training, without
            loss in model quality (will depend on the model, training data and other settings). For
            GPUs which do not support bfloat16, we fall back to fp32. Mixed precision training and fp16
            precision are currently not supported.

        - Gradient Accumulation. You can simulate larger batch sizes by accumulating gradients. This is
            controlled using the ``gradient_accumulation_steps`` flag.

                Total Batch Size = batch_size * number of GPUs * gradient accumulation steps.

            For example: with batch_size=1, nproc_per_node=2 and gradient_accumulation_steps=32 we get a
            total batch size of 64.

            Gradient accumulation is especially useful when you are memory constrained. In this case,
            accumulating gradients might give you better training speed than enabling activation
            checkpointing.

        - Checkpointing. Model weights are checkpointed both at the end of each epoch and at the end of
            training. Optimizer state and recipe state (seed, total_epochs, number of epochs run etc) are
            only saved at the end of a given epoch and used in case of resuming training.

            Resuming training is controlled by the ``resume_from_checkpoint`` flag. Mid-epoch checkpointing is
            currently not supported.

            For more details on the checkpointer, please take a look at
            our checkpointer deepdive (https://pytorch.org/torchtune/main/deep_dives/checkpointer.html).

        - Logging. Terminal, Disk, WandB and TensorBoard are all supported.

        - Gradient Clipping. Gradient clipping is supported using the ``clip_grad_norm`` flag. By default,
            ``clip_grad_norm`` is set to ``None``. If you only want to log the grad norm, you can set
            ``clip_grad_norm='inf'``.

    For a full list of example configs for this recipe, run ``tune ls`` on the command line. Each config
    has example commands for how to kick-off training.

    Args:
        cfg (DictConfig): OmegaConf object parsed from yaml file

    Raises:
        ValueError: If ``dtype`` is set to fp16.
        RuntimeError: If ``dtype`` is set to bf16 and the hardware does not support bf16.
        RuntimeError: If ``left_pad_sequence`` is set as the data collator.
        RuntimeError: If ``enable_activation_offloading`` is True and device is not CUDA.
        RuntimeError: If ``enable_activation_offloading`` is True and ``enable_activation_checkpointing`` is False.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg=cfg)
        self.to_train_p = cfg.get("train_p", False)
        self.to_train_q = cfg.get("train_q", False)
        self.to_train_quiet_p = cfg.get("train_quiet_p", 0)
        # Store reference logprobs for importance sampling
        self.reference_logprobs_cache = {}
        self.reference_logprobs_cache_p = {}
        self.batch_info_cache = {}
        # Online training annealing parameters
        self.online_training_coeff = cfg.get("online_training_coeff", 0.0)

    def _get_sample_log_probs(
        self,
        logits: Union[torch.Tensor, List[torch.Tensor]],
        labels: torch.Tensor,
        start_pos: Optional[torch.Tensor] = None,
        end_pos: Optional[torch.Tensor] = None,
        compute_entropy: bool = False,
        return_per_token: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Computes statistics of token log probabilities for each sample, handling chunked logits.

        Modes:
        - Default (return_per_token=False, compute_entropy=False): returns mean log-prob per sample over the (optional) range mask.
        - Entropy (compute_entropy=True): returns mean token-wise entropy per sample over the (optional) range mask.
        - Per-token (return_per_token=True): returns a list of per-chunk tensors of shape [B, T_chunk] with gathered
          log-probabilities for the provided target labels (labels may be pre-shifted by caller). If start/end are
          provided, positions outside the range are zeroed in the returned tensors.
        """
        if isinstance(logits, torch.Tensor):
            logits = [logits]

        batch_size = labels.shape[0]

        # Create a mask to select specific token ranges if start and end positions are provided
        range_mask = torch.ones_like(labels, dtype=torch.bool)
        if start_pos is not None and end_pos is not None:
            range_mask = torch.zeros_like(labels, dtype=torch.bool)
            for i in range(batch_size):
                s, e = start_pos[i].item(), end_pos[i].item()
                if s < e:
                    range_mask[i, s:e] = True

        # If caller requests per-token outputs, return per-chunk gathered logprobs
        if return_per_token:
            per_token_chunks: List[torch.Tensor] = []
            num_chunks = len(logits)
            labels_chunks = labels.chunk(num_chunks, dim=1)
            mask_chunks = range_mask.chunk(num_chunks, dim=1)

            for logit_chunk, label_chunk, mask_chunk in zip(
                logits, labels_chunks, mask_chunks
            ):
                log_probs_full = torch.nn.functional.log_softmax(
                    logit_chunk.float(), dim=-1
                )
                vocab_size = log_probs_full.shape[-1]
                # Clamp labels to prevent gather crash on ignore_index; loss will mask later
                valid_indices = label_chunk.clamp(0, vocab_size - 1)
                gathered = torch.gather(
                    log_probs_full, dim=-1, index=valid_indices.unsqueeze(-1)
                ).squeeze(-1)
                # If a range was specified, zero out positions outside it
                if start_pos is not None and end_pos is not None:
                    gathered = gathered * mask_chunk
                per_token_chunks.append(gathered)
            return per_token_chunks

        chunk_values = []
        chunk_token_counts = []

        num_chunks = len(logits)
        labels_chunks = labels.chunk(num_chunks, dim=1)
        mask_chunks = range_mask.chunk(num_chunks, dim=1)

        for logit_chunk, label_chunk, mask_chunk in zip(
            logits, labels_chunks, mask_chunks
        ):
            # If there are no tokens in the specified range for this chunk, skip it
            if not mask_chunk.any():
                chunk_values.append(
                    torch.zeros(
                        batch_size, device=logits[0].device, dtype=logits[0].dtype
                    )
                )
                chunk_token_counts.append(
                    torch.zeros(batch_size, device=logits[0].device)
                )
                continue

            if compute_entropy:
                probs = torch.nn.functional.softmax(logit_chunk, dim=-1)
                log_probs = torch.nn.functional.log_softmax(logit_chunk, dim=-1)
                # Entropy H(p) = - Σ p(x) * log p(x)
                value_per_token_2d = -torch.sum(probs * log_probs, dim=-1)
                # Apply the range mask
                masked_values = value_per_token_2d * mask_chunk
                chunk_values.append(masked_values.sum(dim=1))
            else:

                log_probs_full = torch.nn.functional.log_softmax(logit_chunk, dim=-1)

                vocab_size = log_probs_full.shape[-1]

                # Mask for all invalid tokens, including padding and any other out-of-bounds indices.
                # A token is valid if its ID is within the vocabulary range [0, vocab_size - 1].
                # The loss function's ignore_index (e.g. -100) is also handled by this check.
                padding_mask = (label_chunk >= 0) & (label_chunk < vocab_size)

                # Replace invalid indices with a dummy index (0) to prevent gather from crashing.
                # The results for these indices will be ignored anyway due to the mask.
                labels_for_gather = torch.where(
                    padding_mask, label_chunk, torch.zeros_like(label_chunk)
                )

                gathered_log_probs = torch.gather(
                    log_probs_full, -1, labels_for_gather.unsqueeze(-1)
                ).squeeze(-1)

                # Combine with the range mask
                final_mask = padding_mask & mask_chunk

                masked_log_probs = gathered_log_probs * final_mask
                chunk_values.append(masked_log_probs.sum(dim=1))

            chunk_token_counts.append(mask_chunk.sum(dim=1))

        total_value_per_sample = torch.stack(chunk_values).sum(dim=0)
        num_tokens_per_sample = torch.stack(chunk_token_counts).sum(dim=0)

        # Calculate the mean value per token for each sample
        mean_value = torch.where(
            num_tokens_per_sample > 0,
            total_value_per_sample / num_tokens_per_sample,
            torch.zeros_like(total_value_per_sample),
        )

        return mean_value

    def _get_sample_log_probs_from_chunks(
        self, logprobs_chunks: List[torch.Tensor], labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Helper method to compute sample log probabilities from precomputed logprob chunks.
        This reuses the logic from _get_sample_log_probs but for already computed logprobs.
        """
        batch_size = 1

        # Shift labels to align with logits for next-token prediction
        shifted_labels = torch.hstack(
            (labels[..., 1:], self.ignore_labels_cache[:batch_size].squeeze(0))
        ).unsqueeze(0)

        num_chunks = len(logprobs_chunks)
        labels_chunks = shifted_labels.chunk(num_chunks, dim=1)

        total_log_probs_per_sample = torch.zeros(batch_size, device=labels.device)
        num_valid_tokens_per_sample = torch.zeros(batch_size, device=labels.device)

        for logprob_chunk, label_chunk in zip(logprobs_chunks, labels_chunks):
            # Optimization: skip chunks where all labels are the ignore index
            if (label_chunk == self._loss_fn.ignore_index).all():
                continue

            chunk_seq_len, vocab_size = logprob_chunk.shape

            logprobs_flat = logprob_chunk.reshape(-1, vocab_size)
            labels_flat = label_chunk.reshape(-1)

            ignore_mask = labels_flat == self._loss_fn.ignore_index

            # Clamp labels to be valid indices for gather
            valid_indices = labels_flat.clamp(0, vocab_size - 1)

            # Gather the log probabilities of the target tokens
            gathered_log_probs = torch.gather(
                logprob_chunk, dim=-1, index=valid_indices.unsqueeze(-1)
            ).squeeze(-1)

            # Apply mask to zero out ignored tokens
            gathered_log_probs = torch.where(
                ignore_mask,
                torch.zeros_like(gathered_log_probs),
                gathered_log_probs,
            )

            # Sum log probabilities for the current chunk and add to total
            chunk_log_probs = gathered_log_probs.reshape(1, chunk_seq_len).sum(dim=1)
            total_log_probs_per_sample += chunk_log_probs

            # Count valid (non-ignored) tokens in the chunk
            chunk_valid_tokens = (label_chunk != self._loss_fn.ignore_index).sum(dim=1)
            num_valid_tokens_per_sample += chunk_valid_tokens

        # Calculate the mean log probability per valid token for each sample
        # Avoid division by zero for samples with no valid tokens
        mean_log_probs = torch.where(
            num_valid_tokens_per_sample > 0,
            total_log_probs_per_sample / num_valid_tokens_per_sample,
            torch.zeros_like(total_log_probs_per_sample),
        )

        return mean_log_probs

    def _compute_kl_divergence_rao_blackwellized_masked(
        self,
        logits_p: Union[torch.Tensor, List[torch.Tensor]],
        logits_q: Union[torch.Tensor, List[torch.Tensor]],
        labels_without_priv: torch.Tensor,
        labels_with_priv: torch.Tensor,
        end_of_prompt_with_priv: Optional[torch.Tensor] = None,
        action_start_pos_with_priv: Optional[torch.Tensor] = None,
        end_of_prompt_without_priv: Optional[torch.Tensor] = None,
        action_start_pos_without_priv: Optional[torch.Tensor] = None,
        return_logprobs: bool = False,
        action_end_pos_with_priv: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Computes the Rao-Blackwellized KL divergence estimate between two token distributions,
        focusing on the tokens between the end of the prompt and the start of the action.

        KL(p||q) = E_p[log p(x) - log q(x)] = Σ exp(log p(x)) * (log p(x) - log q(x))

        Args:
            logits_p: Logits from the first model (p distribution)
            logits_q: Logits from the second model (q distribution)
            labels_without_priv: Target labels for the without privilege model
            labels_with_priv: Target labels for the with privilege model
            end_of_prompt_with_priv: End position of prompt tokens in with privilege scenario
            action_start_pos_with_priv: Start position of action tokens in with privilege scenario
            end_of_prompt_without_priv: End position of prompt tokens in without privilege scenario
            action_start_pos_without_priv: Start position of action tokens in without privilege scenario
            return_logprobs: Whether to return the computed logprobs for caching

        Returns:
            KL divergence for each sample in the batch (for thought tokens)
            Optionally: tuple of (KL divergence, logprobs_q) if return_logprobs=True
        """
        device = logits_p[0].device

        # Store original logits for reference logprobs computation if needed
        if return_logprobs:
            if isinstance(logits_q, torch.Tensor):
                original_logits_q = [logits_q]
            else:
                original_logits_q = logits_q.copy()

        if isinstance(logits_p, torch.Tensor):
            num_chunks = logits_p.shape[0]
            logits_p = torch.concat(logits_p)
        else:
            num_chunks = len(logits_p)
            logits_p = torch.concat(logits_p, dim=1)

        if isinstance(logits_q, torch.Tensor):
            logits_q = torch.concat(logits_q)
        else:
            logits_q = torch.concat(logits_q, dim=1)

        batch_size = labels_with_priv.shape[0]
        seq_len = labels_with_priv.shape[1]
        seq_without_priv = labels_without_priv.shape[1]

        # Build masks for KL region; if positions are not provided, use full sequence
        if end_of_prompt_with_priv is None or action_start_pos_with_priv is None:
            thought_mask_with_priv = torch.ones(
                (batch_size, seq_len), dtype=torch.bool, device=device
            )
        else:
            thought_mask_with_priv = torch.zeros(
                (batch_size, seq_len), dtype=torch.bool, device=device
            )
            for i in range(batch_size):
                start_with = end_of_prompt_with_priv[i].item()
                end_with = action_start_pos_with_priv[i].item()
                if start_with < end_with:
                    thought_mask_with_priv[i, start_with:end_with] = True

        if end_of_prompt_without_priv is None or action_start_pos_without_priv is None:
            thought_mask_without_priv = torch.ones(
                (batch_size, seq_without_priv), dtype=torch.bool, device=device
            )
        else:
            thought_mask_without_priv = torch.zeros(
                (batch_size, seq_without_priv), dtype=torch.bool, device=device
            )
            for i in range(batch_size):
                start_without = end_of_prompt_without_priv[i].item()
                end_without = action_start_pos_without_priv[i].item()
                if start_without < end_without:
                    thought_mask_without_priv[i, start_without:end_without] = True

        # Apply thought mask to validity masks
        valid_mask_priv = (
            labels_with_priv != self._loss_fn.ignore_index
        ) & thought_mask_with_priv
        valid_mask_without_priv = (
            labels_without_priv != self._loss_fn.ignore_index
        ) & thought_mask_without_priv

        # Select only thought tokens
        unchunked_logits_q = logits_q[valid_mask_priv]
        logits_q = unchunked_logits_q.chunk(num_chunks, dim=0)

        unchunked_logits_p = logits_p[valid_mask_without_priv]
        logits_p = unchunked_logits_p.chunk(num_chunks, dim=0)

        num_chunks = len(logits_p)
        total_kl_per_sample = torch.zeros(batch_size, device=device)
        num_valid_tokens_per_sample = torch.zeros(batch_size, device=device)

        # Store logprobs for importance sampling if needed
        all_logprobs_q = [] if return_logprobs else None

        for logit_p_chunk, logit_q_chunk in zip(logits_p, logits_q):
            if logit_p_chunk.shape[0] == 0 or logit_q_chunk.shape[0] == 0:
                # Skip empty chunks
                continue

            chunk_seq_len, vocab_size = logit_p_chunk.shape

            # Compute KL divergence using the adapted function
            logp = torch.log_softmax(logit_p_chunk, dim=-1)
            logq = torch.log_softmax(logit_q_chunk, dim=-1)

            # Store logprobs for importance sampling if needed
            if return_logprobs:
                all_logprobs_q.append(logq)

            # KL(p||q) = Σ exp(log p(x)) * (log p(x) - log q(x))
            kl_per_token = torch.sum(torch.exp(logq) * (logq - logp), dim=-1)

            # For batched processing, we need to aggregate per sample
            # Since we've flattened valid tokens, we need to track which sample each token belongs to
            # This is a simplified approach - in practice, you might need more sophisticated batching
            chunk_kl = kl_per_token.sum()
            total_kl_per_sample[
                0
            ] += chunk_kl  # Simplified for single sample processing

            # Count valid tokens in the chunk (thought tokens)
            chunk_valid_tokens = logit_p_chunk.shape[0]
            num_valid_tokens_per_sample[0] += chunk_valid_tokens

        # Calculate the mean KL divergence per valid token for each sample
        mean_kl = torch.where(
            num_valid_tokens_per_sample > 0,
            total_kl_per_sample / num_valid_tokens_per_sample,
            torch.zeros_like(total_kl_per_sample),
        )

        if return_logprobs:
            # Compute reference logprobs using original logits_q and focusing on thought tokens
            reference_logprobs = self._get_sample_log_probs(
                original_logits_q,
                labels_with_priv,
                start_pos=end_of_prompt_with_priv,
                end_pos=action_end_pos_with_priv,
            )

            return mean_kl, reference_logprobs
        else:
            return mean_kl

    def _apply_control_variate_flip(self, rewards: List[float]) -> List[float]:
        """
        Apply control variate to flip the reward sign by subtracting rewards from negative min reward,
        then min-max scale them to [-1, 1] (or [0, 1] if all positive).

        Args:
            rewards: List of original rewards

        Returns:
            List of flipped and scaled rewards using control variate
        """
        if not rewards:
            return rewards

        min_reward = min(rewards)
        control_variate_baseline = -min_reward
        flipped_rewards = [control_variate_baseline + reward for reward in rewards]

        min_flipped = min(flipped_rewards)
        max_flipped = max(flipped_rewards)

        # If all rewards are positive, scale to [0, 1]
        if min_flipped >= 0:
            if max_flipped == min_flipped:
                scaled_rewards = [0.0 for _ in flipped_rewards]
            else:
                scaled_rewards = [
                    (r - min_flipped) / (max_flipped - min_flipped)
                    for r in flipped_rewards
                ]
        else:
            # Scale to [-1, 1]
            if max_flipped == min_flipped:
                scaled_rewards = [0.0 for _ in flipped_rewards]
            else:
                scaled_rewards = [
                    2 * (r - min_flipped) / (max_flipped - min_flipped) - 1
                    for r in flipped_rewards
                ]

        return scaled_rewards

    def compute_rewards(self, action_log_ps_as_reward: bool = False):
        if self.train_q_online:
            return self.compute_rewards_q_online(action_log_ps_as_reward)
        else:
            return self.compute_rewards_standard(action_log_ps_as_reward)

    def compute_rewards_q_online(
        self,
        action_log_ps_as_reward: bool = False,
    ) -> Tuple[
        List[float],
        List[float],
        float,
        float,
        float,
        Dict[str, List[float]],
        List[float],
    ]:
        """
        Computes rewards and advantages for each sample in the dataloader.
        The reward combines the original reward with KL divergence penalty:
        reward = original_reward - gamma * KL(thought_tokens)

        KL divergence is computed on "thought" tokens (between prompt and action):
        KL(p(thought | prompt_without_secret) || p(thought | prompt_with_secret))

        This uses the Rao-Blackwellized estimator for KL divergence on thought tokens.
        Advantages are calculated as reward - mean(rewards for the same goal).

        Returns:
            A tuple containing:
            - A list of rewards for each sample (original + KL penalty).
            - A list of advantages for each sample.
            - Mean log probability of action tokens without privilege (p_y_g_zx_mean).
            - Mean log probability of action tokens with privilege (q_y_g_xz_mean).
            - Mean entropy of thought tokens with privilege (q_z_g_xy_mean).
            - Dictionary of rewards grouped by goal.
            - A list of KL divergences for each sample (thought tokens only).
        """
        self._model.eval()
        all_rewards = []
        all_goals = []
        p_y_g_zx = []
        q_z_g_xy = []  # This will now store entropy of thought tokens
        q_y_g_xz = []  # This will store action log-probs with privilege
        kls = []
        dataloader = self._dataloader

        # Clear previous reference logprobs cache and store detailed logprobs for importance sampling
        self.reference_logprobs_cache = {}
        self.reference_logprobs_cache_p = {}
        # Also store batch info for matching during training
        self.batch_info_cache = {} if self.use_importance_sampling else None
        batch_idx = 0

        # Track trajectory indices and steps for GRPO grouping
        all_trajectory_indices: List[int] = []
        all_steps: List[int] = []
        for batch in tqdm(dataloader, desc="Computing Rewards"):
            all_steps.append(batch.get("step", 0))
            all_trajectory_indices.append(batch.get("trajectory_id", 0))
            goals = batch.pop("goal", None)
            _ = batch.pop("privileged_found", None)
            reward = torch.tensor([batch.pop("reward")], device=self._device)

            utils.batch_to_device(batch, self._device)

            # Record trajectory indices and steps for grouping (move to CPU for list)

            # with privilege
            batch_with_priv = batch["with_privilege"]
            labels_with_priv = batch_with_priv["labels"]
            model_inputs_with_priv = {
                k: v
                for k, v in batch_with_priv.items()
                if k
                not in [
                    "labels",
                    "action_start_pos",
                    "action_end_pos",
                    "end_of_prompt",
                    "mask",
                    "attention_mask",
                ]
            }
            with torch.no_grad():
                logits_with_priv = self._model(**model_inputs_with_priv)
                logits_with_priv = [
                    logit / self.sampling_temperature for logit in logits_with_priv
                ]

            # without privilege
            batch_without_priv = batch["without_privilege"]
            labels_without_priv = batch_without_priv["labels"]
            model_inputs_without_priv = {
                k: v
                for k, v in batch_without_priv.items()
                if k
                not in [
                    "labels",
                    "action_start_pos",
                    "action_end_pos",
                    "end_of_prompt",
                    "mask",
                    "attention_mask",
                ]
            }
            with torch.no_grad():
                logits_without_priv = self._ref_model(**model_inputs_without_priv)
                logits_without_priv = [
                    logit / self.sampling_temperature for logit in logits_without_priv
                ]
            labels_shifted_with_priv = torch.hstack(
                (
                    labels_with_priv[..., 1:],
                    self.ignore_labels_cache[: labels_with_priv.shape[0]],
                )
            )
            labels_shifted_without_priv = torch.hstack(
                (
                    labels_without_priv[..., 1:],
                    self.ignore_labels_cache[: labels_without_priv.shape[0]],
                )
            )

            # Compute KL divergence (positions optional; defaults to all non-ignored)
            if self.use_importance_sampling:

                ref_logprobs_chunks_q: List[torch.Tensor] = self._get_sample_log_probs(
                    logits_with_priv,
                    labels_shifted_with_priv,
                    return_per_token=True,
                )
                self.reference_logprobs_cache[batch_idx] = ref_logprobs_chunks_q
            # if (end_of_prompt_without_priv - action_start_pos_without_priv).item() == (end_of_prompt_without_priv - action_start_pos_without_priv).item():
            try:
                # Positions are optional: compute KL over all non-ignored labels if not provided
                kl_divergence = self._compute_kl_divergence_rao_blackwellized_masked(
                    logits_without_priv,
                    logits_with_priv,
                    labels_without_priv,
                    labels_with_priv,
                    None,
                    None,
                    None,
                    None,
                    return_logprobs=False,
                )
            except Exception as e:
                log.info(f"Error computing KL divergence for batch {batch_idx}: {e}")
                kl_divergence = torch.tensor([1.0], device=self._device)

            rewards = reward - self.gamma * kl_divergence

            # q_y_g_xz.extend(action_log_prob_with_privilege.detach().cpu().tolist())
            # q_z_g_xy.extend(thought_entropy_with_privilege.detach().cpu().tolist())
            # p_y_g_zx.extend(action_log_prob_without_privilege.detach().cpu().tolist())
            all_rewards.extend(rewards.cpu().tolist())
            all_goals.extend(goals)
            # Store KL divergence (thought tokens) for logging
            if not action_log_ps_as_reward:
                kls.extend(kl_divergence.cpu().tolist())

            batch_idx += 1
        rewards = rescale_rewards(all_rewards)
        # Group rewards by goal
        rewards_by_goal = {}
        for goal, reward in zip(all_goals, all_rewards):
            if goal not in rewards_by_goal:
                rewards_by_goal[goal] = []
            rewards_by_goal[goal].append(reward)

        # GRPO: group-relative advantages by (goal, step) aggregating (trajectory_id, reward)
        rewards_by_goal_step: Dict[str, Dict[int, List[Tuple[int, float]]]] = (
            defaultdict(lambda: defaultdict(list))
        )
        for g_, tid, stp, r in zip(
            all_goals, all_trajectory_indices, all_steps, all_rewards
        ):
            rewards_by_goal_step[g_][int(stp)].append((int(tid), float(r)))

        # Calculate advantages per sample relative to its (goal, step) group
        advantages: List[float] = []
        if not all_rewards:
            advantages = []
        else:
            for g_, tid, stp, r in zip(
                all_goals, all_trajectory_indices, all_steps, all_rewards
            ):
                grp = rewards_by_goal_step[g_][int(stp)]
                if len(grp) > 1:
                    mean_r = float(np.mean([rv for _, rv in grp]))
                    advantages.append(r - mean_r)
                else:
                    # Single sample in group: use reward directly to avoid zero advantage
                    advantages.append(r)

        # Apply control variate flip if enabled
        if self.flip_reward_control_variate:
            log.info("Applying control variate to flip reward signs")
            original_rewards = all_rewards.copy()  # Keep original for logging
            all_rewards = self._apply_control_variate_flip(all_rewards)

            # Recalculate rewards_by_goal_step with flipped rewards
            rewards_by_goal_step = defaultdict(lambda: defaultdict(list))
            for g_, tid, stp, reward in zip(
                all_goals, all_trajectory_indices, all_steps, all_rewards
            ):
                rewards_by_goal_step[g_][int(stp)].append((int(tid), float(reward)))

            # Recalculate advantages with flipped rewards using (goal, step) grouping
            advantages = []
            for g_, tid, stp, reward in zip(
                all_goals, all_trajectory_indices, all_steps, all_rewards
            ):
                grp = rewards_by_goal_step[g_][int(stp)]
                if len(grp) > 1:
                    mean_r = float(np.mean([rv for _, rv in grp]))
                    advantages.append(reward - mean_r)
                else:
                    advantages.append(reward)

            log.info(
                f"Original reward range: [{min(original_rewards):.4f}, {max(original_rewards):.4f}]"
            )
            log.info(
                f"Flipped reward range: [{min(all_rewards):.4f}, {max(all_rewards):.4f}]"
            )
            # Explicitly delete large tensors and clear cache
        del logits_with_priv
        del logits_without_priv
        del labels_with_priv
        del labels_without_priv
        del labels_shifted_with_priv
        del labels_shifted_without_priv
        del reward
        del kl_divergence
        if self.use_importance_sampling:
            del ref_logprobs_chunks_q
        p_y_g_xz_mean = np.mean(p_y_g_zx)
        q_y_g_xz_mean = np.mean(q_y_g_xz)
        q_z_g_xy_mean = np.mean(q_z_g_xy)

        self._model.train()
        if self.reference_set:
            del self._ref_model

        return (
            all_rewards,
            advantages,
            p_y_g_xz_mean,
            q_y_g_xz_mean,
            q_z_g_xy_mean,
            rewards_by_goal,
            kls,
            p_y_g_zx,
        )

    def compute_rewards_offline(
        self,
        action_log_ps_as_reward: bool = False,
    ) -> Tuple[
        List[float],
        List[float],
        float,
        float,
        float,
        Dict[str, List[float]],
        List[float],
    ]:
        """
        Computes rewards and advantages for each sample in the dataloader.
        The reward combines the original reward with KL divergence penalty:
        reward = original_reward - gamma * KL(thought_tokens)

        KL divergence is computed on "thought" tokens (between prompt and action):
        KL(p(thought | prompt_without_secret) || p(thought | prompt_with_secret))

        This uses the Rao-Blackwellized estimator for KL divergence on thought tokens.
        Advantages are calculated as reward - mean(rewards for the same goal).

        Returns:
            A tuple containing:
            - A list of rewards for each sample (original + KL penalty).
            - A list of advantages for each sample.
            - Mean log probability of action tokens without privilege (p_y_g_zx_mean).
            - Mean log probability of action tokens with privilege (q_y_g_xz_mean).
            - Mean entropy of thought tokens with privilege (q_z_g_xy_mean).
            - Dictionary of rewards grouped by goal.
            - A list of KL divergences for each sample (thought tokens only).
        """
        self._model.eval()
        all_rewards = []
        all_goals = []
        p_y_g_zx = []
        q_z_g_xy = []  # This will now store entropy of thought tokens
        q_y_g_xz = []  # This will store action log-probs with privilege
        kls = []
        dataloader = self._dataloader

        # Clear previous reference logprobs cache and store detailed logprobs for importance sampling
        self.reference_logprobs_cache = {}
        self.reference_logprobs_cache_p = {}
        # Also store batch info for matching during training
        self.batch_info_cache = {} if self.use_importance_sampling else None
        batch_idx = 0

        # Track trajectory indices and steps for GRPO grouping
        all_trajectory_indices: List[int] = []
        all_steps: List[int] = []
        for batch in tqdm(dataloader, desc="Computing Rewards"):
            if self._skip_max_seq_len_samples(
                batch["with_privilege"]
            ) or self._skip_max_seq_len_samples(batch["without_privilege"]):
                continue

            goals = batch.pop("goal", None)
            _ = batch.pop("privileged_found", None)
            reward = torch.tensor([batch.pop("reward")], device=self._device)

            # Extract positions from batch
            end_of_prompt_with_priv = batch["with_privilege"]["end_of_prompt"]
            action_start_pos_with_priv = batch["with_privilege"]["action_start_pos"]
            action_end_pos_with_priv = batch["with_privilege"]["action_end_pos"]

            end_of_prompt_without_priv = batch["without_privilege_target_action"][
                "end_of_prompt"
            ]
            action_start_pos_without_priv = batch["without_privilege_target_action"][
                "action_start_pos"
            ]
            action_end_pos_without_priv = batch["without_privilege_target_action"][
                "action_end_pos"
            ]

            trajectory_index = batch["trajectory_index"]
            step_ids = batch.get("step", None)

            utils.batch_to_device(batch, self._device)

            # Record trajectory indices and steps for grouping (move to CPU for list)
            all_trajectory_indices.extend(trajectory_index.cpu().tolist())
            if step_ids is not None:
                all_steps.extend(step_ids.cpu().tolist())
            else:
                # If step not provided, default to 0 for all items in this batch
                all_steps.extend([0] * trajectory_index.shape[0])

            # with privilege
            batch_with_priv = batch["with_privilege"]
            labels_with_priv = batch_with_priv["labels"]
            model_inputs_with_priv = {
                k: v
                for k, v in batch_with_priv.items()
                if k
                not in [
                    "labels",
                    "action_start_pos",
                    "action_end_pos",
                    "end_of_prompt",
                    "mask",
                    "attention_mask",
                ]
            }
            with torch.no_grad():
                logits_with_priv = self._model(**model_inputs_with_priv)
                logits_with_priv = [
                    logit / self.sampling_temperature for logit in logits_with_priv
                ]

            # without privilege
            batch_without_priv = batch["without_privilege_target_action"]
            labels_without_priv = batch_without_priv["labels"]
            model_inputs_without_priv = {
                k: v
                for k, v in batch_without_priv.items()
                if k
                not in [
                    "labels",
                    "action_start_pos",
                    "action_end_pos",
                    "end_of_prompt",
                    "mask",
                    "attention_mask",
                ]
            }
            with torch.no_grad():
                logits_without_priv = self._ref_model(**model_inputs_without_priv)
                logits_without_priv = [
                    logit / self.sampling_temperature for logit in logits_without_priv
                ]
            labels_shifted_with_priv = torch.hstack(
                (
                    labels_with_priv[..., 1:],
                    self.ignore_labels_cache[: labels_with_priv.shape[0]],
                )
            )
            labels_shifted_without_priv = torch.hstack(
                (
                    labels_without_priv[..., 1:],
                    self.ignore_labels_cache[: labels_without_priv.shape[0]],
                )
            )

            # Compute log probabilities of action tokens
            action_log_prob_with_privilege = self._get_sample_log_probs(
                logits_with_priv,
                labels_shifted_with_priv,
                start_pos=action_start_pos_with_priv,
                end_pos=action_end_pos_with_priv,
            )

            action_log_prob_without_privilege = self._get_sample_log_probs(
                logits_without_priv,
                labels_shifted_without_priv,
                start_pos=action_start_pos_without_priv,
                end_pos=action_end_pos_without_priv,
            )

            # Compute KL divergence on thought tokens
            if self.use_importance_sampling:

                ref_logprobs_chunks_q: List[torch.Tensor] = self._get_sample_log_probs(
                    logits_with_priv,
                    labels_shifted_with_priv,
                    return_per_token=True,
                )
                self.reference_logprobs_cache[batch_idx] = ref_logprobs_chunks_q
            # if (end_of_prompt_without_priv - action_start_pos_without_priv).item() == (end_of_prompt_without_priv - action_start_pos_without_priv).item():
            try:
                kl_divergence = self._compute_kl_divergence_rao_blackwellized_masked(
                    logits_without_priv,
                    logits_with_priv,
                    labels_without_priv,
                    labels_with_priv,
                    end_of_prompt_with_priv,
                    action_start_pos_with_priv,
                    end_of_prompt_without_priv,
                    action_start_pos_without_priv,
                    return_logprobs=False,
                )
            except Exception as e:
                kl_divergence = torch.tensor([1.0], device=self._device)

            # Compute entropy of thought tokens for the privileged model
            thought_entropy_with_privilege = self._get_sample_log_probs(
                logits_with_priv,
                labels_shifted_with_priv,
                start_pos=end_of_prompt_with_priv,
                end_pos=action_start_pos_with_priv,
                compute_entropy=True,
            )

            # Use KL divergence as rewards with gamma weighting
            if action_log_ps_as_reward:
                rewards = reward + action_log_prob_without_privilege
                # rewards = action_log_prob_without_privilege
            else:
                rewards = reward + (
                    torch.clamp(action_log_prob_without_privilege, min=-1.0)
                    - self.gamma * kl_divergence
                )
            # if action_log_prob_with_privilege != action_log_prob_without_privilege else torch.tensor([-1.0], device=self._device)

            # clip rewards
            # rewards = torch.clamp(rewards, -1.0 , 1.0)

            # Apply unsuccessful negative reward transformation if enabled
            if self.unsuccessful_negative_reward:
                # Convert reward=0 to reward=-1 for unsuccessful traces
                rewards = torch.where(
                    reward == 0, torch.tensor(-1.0, device=self._device), rewards
                )

            q_y_g_xz.extend(action_log_prob_with_privilege.cpu().tolist())
            q_z_g_xy.extend(thought_entropy_with_privilege.cpu().tolist())
            p_y_g_zx.extend(action_log_prob_without_privilege.cpu().tolist())
            all_rewards.extend(rewards.cpu().tolist())
            all_goals.extend(goals)
            # Store KL divergence (thought tokens) for logging
            if not action_log_ps_as_reward:
                kls.extend(kl_divergence.cpu().tolist())

            batch_idx += 1
        rewards = rescale_rewards(all_rewards)
        # Group rewards by goal
        rewards_by_goal = {}
        for goal, reward in zip(all_goals, all_rewards):
            if goal not in rewards_by_goal:
                rewards_by_goal[goal] = []
            rewards_by_goal[goal].append(reward)

        # GRPO: group-relative advantages by (goal, step) aggregating (trajectory_id, reward)
        rewards_by_goal_step: Dict[str, Dict[int, List[Tuple[int, float]]]] = (
            defaultdict(lambda: defaultdict(list))
        )
        for g_, tid, stp, r in zip(
            all_goals, all_trajectory_indices, all_steps, all_rewards
        ):
            rewards_by_goal_step[g_][int(stp)].append((int(tid), float(r)))

        # Calculate advantages per sample relative to its (goal, step) group
        advantages: List[float] = []
        if not all_rewards:
            advantages = []
        else:
            for g_, tid, stp, r in zip(
                all_goals, all_trajectory_indices, all_steps, all_rewards
            ):
                grp = rewards_by_goal_step[g_][int(stp)]
                if len(grp) > 1:
                    mean_r = float(np.mean([rv for _, rv in grp]))
                    advantages.append(r - mean_r)
                else:
                    # Single sample in group: use reward directly to avoid zero advantage
                    advantages.append(r)

        # Apply control variate flip if enabled
        if self.flip_reward_control_variate:
            log.info("Applying control variate to flip reward signs")
            original_rewards = all_rewards.copy()  # Keep original for logging
            all_rewards = self._apply_control_variate_flip(all_rewards)

            # Recalculate rewards_by_goal_step with flipped rewards
            rewards_by_goal_step = defaultdict(lambda: defaultdict(list))
            for g_, tid, stp, reward in zip(
                all_goals, all_trajectory_indices, all_steps, all_rewards
            ):
                rewards_by_goal_step[g_][int(stp)].append((int(tid), float(reward)))

            # Recalculate advantages with flipped rewards using (goal, step) grouping
            advantages = []
            for g_, tid, stp, reward in zip(
                all_goals, all_trajectory_indices, all_steps, all_rewards
            ):
                grp = rewards_by_goal_step[g_][int(stp)]
                if len(grp) > 1:
                    mean_r = float(np.mean([rv for _, rv in grp]))
                    advantages.append(reward - mean_r)
                else:
                    advantages.append(reward)

            log.info(
                f"Original reward range: [{min(original_rewards):.4f}, {max(original_rewards):.4f}]"
            )
            log.info(
                f"Flipped reward range: [{min(all_rewards):.4f}, {max(all_rewards):.4f}]"
            )

        p_y_g_xz_mean = np.mean(p_y_g_zx)
        q_y_g_xz_mean = np.mean(q_y_g_xz)
        q_z_g_xy_mean = np.mean(q_z_g_xy)

        self._model.train()
        if self.reference_set:
            del self._ref_model

        return (
            all_rewards,
            advantages,
            p_y_g_xz_mean,
            q_y_g_xz_mean,
            q_z_g_xy_mean,
            rewards_by_goal,
            kls,
            p_y_g_zx,
        )

    def train(self) -> None:
        if self.train_q_online:
            log.info("=" * 50)
            log.info("=" * 50)
            log.info("=" * 50)
            log.info("Training Q model online with and also P model")
            log.info("=" * 50)
            log.info("=" * 50)
            log.info("=" * 50)
            self.train_q_online_run(
                train_q=self.train_q_online_with_p, train_p=self.train_p_online_with_q
            )

        elif self.to_train_quiet_p:
            log.info("=" * 50)
            log.info("=" * 50)
            log.info("=" * 50)
            log.info("Training P model in quiet mode")
            log.info("=" * 50)
            log.info("=" * 50)
            log.info("=" * 50)
            self.epochs_run = 0
            self.train_quiet_p()
        else:
            if self.to_train_q:
                log.info("=" * 50)
                log.info("=" * 50)
                log.info("=" * 50)
                log.info("Training Q model")
                log.info("=" * 50)
                log.info("=" * 50)
                log.info("=" * 50)
                self.train_q()

            if self.to_train_p:
                log.info("=" * 50)
                log.info("=" * 50)
                log.info("=" * 50)
                log.info("Training P model")
                log.info("=" * 50)
                log.info("=" * 50)
                log.info("=" * 50)
                self.epochs_run = 0
                self.train_p()

    def train_p(self) -> None:
        """
        The core training loop.
        """
        # clean up before training begins
        training.cleanup_before_training()

        world_size, rank = training.get_world_size_and_rank()

        # zero out the gradients before starting training
        if not self._optimizer_in_bwd:
            self._optimizer.zero_grad()
        else:
            for opt in self._optim_ckpt_wrapper.optim_map.values():
                opt.zero_grad()

        # Initialize tokens count and running loss (for grad accumulation)
        t0 = time.perf_counter()
        running_loss = 0
        num_tokens = 0

        # NOTE: added by us - sample just once at the beginning of the epoch loop
        self._sampler.set_epoch(0)

        self._profiler.start()
        # self.epochs_run should be non-zero when we're resuming from a checkpoint
        for curr_epoch in range(self.epochs_run, self.total_epochs):
            # Update the sampler to ensure data is correctly shuffled across epochs
            # in case shuffle is True
            # NOTE: removing it from here and putting it before the epoch loop
            # because our epochs are not the same as the dataloader epochs
            for _sampler_validation in self._sampler_validation_list:
                _sampler_validation.set_epoch(curr_epoch)  # NOTE: added by us

            # NOTE: added by us
            # ------ Validation Step ------ #
            self._model.eval()

            with torch.no_grad():
                for i, dataloader_validation in enumerate(
                    self._dataloader_validation_list
                ):
                    for _, batch in enumerate(dataloader_validation):
                        batch.pop("goal", None)
                        batch.pop("privileged_found", None)
                        utils.batch_to_device(batch, self._device)
                        val_loss = torch.tensor(0.0, device=self._device)
                        if self._is_rank_zero:
                            self._metric_logger.log_dict(
                                {f"val_loss_{i}": val_loss.item()},
                                step=self.global_step,
                            )
            del val_loss

            # Precompute ref logprobs cache for P if using importance sampling
            if self.use_importance_sampling:
                self.reference_logprobs_cache_p = {}
                pre_idx = 0
                with torch.no_grad():
                    # keep model in eval for precompute
                    self._model.eval()
                    for pre_batch in tqdm(
                        self._dataloader,
                        desc="Precomputing P ref logprobs",
                        disable=not (rank == 0),
                    ):
                        if self._skip_max_seq_len_samples(
                            pre_batch["with_privilege"]
                        ) or self._skip_max_seq_len_samples(
                            pre_batch["without_privilege"]
                        ):
                            continue
                        # move to device
                        pre_batch.pop("goal", None)
                        pre_batch.pop("privileged_found", None)
                        utils.batch_to_device(pre_batch, self._device)
                        batch_without_priv = pre_batch["without_privilege"]
                        model_inputs_without_priv = {
                            k: v
                            for k, v in batch_without_priv.items()
                            if k
                            not in [
                                "labels",
                                "action_start_pos",
                                "action_end_pos",
                                "end_of_prompt",
                                "mask",
                                "attention_mask",
                            ]
                        }
                        with self.activations_handling_ctx and torch.no_grad():
                            ref_logits = self._model(**model_inputs_without_priv)

                        # shifted labels from without_priv side (align to next-token)
                        labels_without_priv = pre_batch["without_privilege"]["labels"]
                        labels_shifted_without_priv = torch.hstack(
                            (
                                labels_without_priv[..., 1:],
                                self.ignore_labels_cache[
                                    : labels_without_priv.shape[0]
                                ],
                            )
                        )
                        # gather per-token ref logprobs chunks
                        ref_chunks_p: List[torch.Tensor] = self._get_sample_log_probs(
                            ref_logits,
                            labels_shifted_without_priv,
                            return_per_token=True,
                        )
                        self.reference_logprobs_cache_p[pre_idx] = ref_chunks_p
                        pre_idx += 1
                # restore train mode for training below
                self._model.train()

            # ------ Training Epoch ------ #
            # Initialize tokens count and running loss (for grad accumulation)
            t0 = time.perf_counter()
            running_loss = torch.tensor(0.0, device=self._device)
            num_tokens = 0
            real_num_tokens = 0
            max_len_samples = 0
            # Update entropy tracking variables to include sum and mean metrics
            running_per_token_ent_sum = 0
            running_full_token_ent_sum = 0
            running_per_token_ent_mean = 0
            running_full_token_ent_mean = 0
            # Track per-step and running stats for tokens
            running_bprop_tokens_sum = 0.0
            running_bprop_tokens_min = float("inf")
            running_bprop_tokens_max = 0.0
            running_bprop_tokens_count = 0
            running_total_tokens_sum = 0.0
            running_total_tokens_min = float("inf")
            running_total_tokens_max = 0.0
            running_total_tokens_count = 0
            self._model.train()  # NOTE: added by us

            pbar = tqdm(
                total=self._steps_per_epoch, disable=not (rank == 0), desc="Training"
            )

            # NOTE: added by us - counter to account for samples that are too long
            idx = 0
            processed_samples = 0
            n_samples = len(self._dataloader)
            n_gpus = torch.distributed.get_world_size()
            number_leftover_samples = (
                n_samples * n_gpus
            ) % self._gradient_accumulation_steps

            for j, batch in enumerate(self._dataloader):
                if ((idx // self._gradient_accumulation_steps)) >= (
                    self._steps_per_epoch
                ) and not self.max_bsize:
                    break

                train_batch = batch["without_privilege"]
                if self._skip_max_seq_len_samples(
                    train_batch
                ) or self._skip_max_seq_len_samples(batch["without_privilege"]):
                    max_len_samples += 1
                    continue

                # Start tracking CUDA memory for active steps for just the first epoch
                if (
                    self._is_rank_zero
                    and curr_epoch == 0
                    and self.profiler_profile_memory
                    and idx == self.profiler_wait_steps + self.profiler_warmup_steps
                ):
                    torch.cuda.memory._record_memory_history()

                # Keep privileged_found for branching; still drop goal
                batch.pop("goal", None)
                privileged_found_flags = batch.get("privileged_found", None)
                utils.batch_to_device(batch, self._device)

                # Calculate the number of unmasked tokens in the current batch
                # and increment the total number of tokens seen in the step
                current_num_tokens = (
                    train_batch["labels"] != self._loss_fn.ignore_index
                ).sum()
                num_tokens += current_num_tokens
                # NOTE: added by us
                # let's monitor the total number of tokens
                # Accumulate total tokens across microbatches in this step
                real_num_tokens += train_batch["labels"].numel()

                # Shape [b, s], needed for the loss not the model
                labels = train_batch.pop("labels")
                train_batch.pop("action_start_pos", None)
                train_batch.pop("action_end_pos", None)
                train_batch.pop("end_of_prompt", None)
                train_batch.pop("mask", None)
                train_batch.pop("attention_mask", None)

                batch_size = labels.shape[0]

                processed_samples += batch_size

                with self.activations_handling_ctx:
                    logits = self._model(**train_batch)
                # Shift once for all paths
                labels_shifted = torch.hstack(
                    (labels[..., 1:], self.ignore_labels_cache[: labels.shape[0]])
                )

                # Branch: SFT for privileged samples, RL for non-privileged using reward weights
                # Simplified for batch size 1: directly choose SFT or RL based on privileged_found
                # Prepare logits/labels for loss
                if isinstance(logits, list):
                    loss_logits = logits
                    loss_labels = labels_shifted
                else:
                    loss_logits = logits.reshape(-1, logits.size(-1))
                    loss_labels = labels_shifted.reshape(-1)

                # Determine path: RL when privileged not found (0), else SFT
                use_rl = False
                if privileged_found_flags is not None:
                    flag0 = int(privileged_found_flags.view(-1)[0].item())
                    use_rl = flag0 == 0

                if use_rl:
                    # Importance sampling: use cached per-token ref logprobs for P if available
                    ref_logprobs_for_loss = None
                    if self.use_importance_sampling:

                        ref_cached = self.reference_logprobs_cache_p.get(j)
                        if ref_cached is not None:
                            ref_logprobs_for_loss = ref_cached

                    rewards_vec = batch.get("og_reward", None)
                    if rewards_vec is None:
                        reward_val = torch.tensor([1.0], device=self._device)
                    else:
                        reward_val = rewards_vec.to(self._device).view(-1)[:1]
                    combined_loss = (
                        self._loss_fn(
                            logits=loss_logits,
                            labels=labels_shifted,
                            reward=reward_val,
                            ref_logprobs=ref_logprobs_for_loss,
                            epsilon_low=self.epsilon_low_neg,
                            epsilon_high=self.epsilon_high_pos,
                        ).squeeze(0)
                        * current_num_tokens
                    )
                else:
                    combined_loss = (
                        self._loss_fn(logits=loss_logits, labels=labels_shifted)
                        * current_num_tokens
                    )
                if use_rl:
                    log.info(f"Using RL loss at idx {idx} for sample {j}")

                running_loss += combined_loss.detach()
                # For optimizer in backward, we need to normalize before calling backward
                # This case and gradient accumulation are mutually exclusive
                if self._optimizer_in_bwd:
                    torch.distributed.all_reduce(num_tokens)
                    torch.distributed.all_reduce(running_loss)
                    combined_loss = combined_loss / num_tokens
                combined_loss.backward()
                del combined_loss
                # Step with optimizer
                if (idx + 1) % self._gradient_accumulation_steps == 0 or (
                    (idx + 1) == n_samples
                ):

                    if not self._optimizer_in_bwd:
                        # Get total number of tokens across all ranks to normalize gradients
                        torch.distributed.all_reduce(num_tokens)
                        # This will ensure that the logged loss matches what we're optimizing
                        torch.distributed.all_reduce(running_loss)
                        # All-reduce all entropy metrics

                        # Manually scale the gradients from unnormalized loss by total # of tokens
                        training.scale_grads(self._model, 1 / num_tokens)
                        # scale grads by max_batchsize and real_batchsize
                        if self.max_bsize and (idx + 1) == n_samples:
                            if number_leftover_samples == 1:
                                number_leftover_samples = n_samples
                            scaler = torch.tensor(
                                number_leftover_samples / self.max_bsize
                                if number_leftover_samples > 0
                                else n_samples / self.max_bsize
                            )

                            training.scale_grads(
                                self._model,
                                scaler,
                            )
                            log.info(
                                f"Scaling gradients by {scaler} Original bsize = {number_leftover_samples}"
                            )

                        # Calculate gradient norms before clipping (efficient way)
                        total_norm = torch.nn.utils.clip_grad_norm_(
                            self._model.parameters(), max_norm=float("inf")
                        )

                        grad_norm_stats = {"grad_norm_total": total_norm.item()}

                        if self._clip_grad_norm is not None:
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                self._model.parameters(),
                                max_norm=float(self._clip_grad_norm),
                            )
                        self._optimizer.step()
                        log.info(f"optimizer step")
                        self._optimizer.zero_grad(set_to_none=True)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                            gc.collect()
                            torch.cuda.synchronize()

                    # Update the number of steps when the weights are updated
                    self.global_step += 1

                    # Step the learning rate scheduler
                    if self._lr_scheduler is not None:
                        self._lr_scheduler.step()

                    loss_to_log = running_loss.item() / num_tokens
                    # Update running stats for tokens using this step's totals
                    bprop_tokens_this_step = float(num_tokens.item())
                    total_tokens_this_step = float(real_num_tokens)
                    running_bprop_tokens_sum += bprop_tokens_this_step
                    running_bprop_tokens_min = min(
                        running_bprop_tokens_min, bprop_tokens_this_step
                    )
                    running_bprop_tokens_max = max(
                        running_bprop_tokens_max, bprop_tokens_this_step
                    )
                    running_bprop_tokens_count += 1
                    running_total_tokens_sum += total_tokens_this_step
                    running_total_tokens_min = min(
                        running_total_tokens_min, total_tokens_this_step
                    )
                    running_total_tokens_max = max(
                        running_total_tokens_max, total_tokens_this_step
                    )
                    running_total_tokens_count += 1
                    # Compute running averages
                    bprop_tokens_avg = (
                        running_bprop_tokens_sum / running_bprop_tokens_count
                        if running_bprop_tokens_count > 0
                        else 0.0
                    )
                    total_tokens_avg = (
                        running_total_tokens_sum / running_total_tokens_count
                        if running_total_tokens_count > 0
                        else 0.0
                    )
                    pbar.update(1)
                    pbar.set_description(
                        f"{curr_epoch + 1}|{self.global_step}|Loss: {loss_to_log}"
                    )
                    n_samples
                    # Log per-step metrics
                    if self._is_rank_zero:
                        time_per_step = time.perf_counter() - t0
                        log_dict = {
                            "loss_p": loss_to_log.cpu().item(),
                            "lr": get_lr(
                                (
                                    self._optimizer
                                    if not self._optimizer_in_bwd
                                    else self._optim_ckpt_wrapper
                                ),
                            ),
                            "tokens_per_second_per_gpu": real_num_tokens
                            / (time_per_step * world_size),
                            # Token stats (per local step)
                            "bprop_tokens_avg": bprop_tokens_avg,
                            "bprop_tokens_min": running_bprop_tokens_min,
                            "bprop_tokens_max": running_bprop_tokens_max,
                            "total_tokens_avg": total_tokens_avg,
                            "total_tokens_min": running_total_tokens_min,
                            "total_tokens_max": running_total_tokens_max,
                        }
                        # Add gradient norm stats to logging
                        log_dict.update(grad_norm_stats)

                        # Log clip stats from loss if available (P training)
                        clip_stats = getattr(self._loss_fn, "last_clip_stats", None)
                        if clip_stats is not None:
                            log_dict.update(
                                {
                                    "ppo_masked_tokens": clip_stats.get(
                                        "masked_tokens", 0
                                    ),
                                    "ppo_total_tokens": clip_stats.get(
                                        "total_tokens", 0
                                    ),
                                    "ppo_affected_samples": clip_stats.get(
                                        "affected_samples", 0
                                    ),
                                    "ppo_total_samples": clip_stats.get(
                                        "total_samples", 0
                                    ),
                                }
                            )
                        self._metric_logger.log_dict(
                            log_dict,
                            step=self.global_step,
                        )

                    # Reset running stats for the next step
                    running_loss = torch.tensor(0.0, device=self._device)
                    combined_loss = 0
                    num_tokens = 0
                    real_num_tokens = 0
                    running_per_token_ent_sum = 0
                    running_full_token_ent_sum = 0
                    running_per_token_ent_mean = 0
                    running_full_token_ent_mean = 0
                    t0 = time.perf_counter()

                    # Stop tracking CUDA memory now that active steps are complete
                    if (
                        self._is_rank_zero
                        and curr_epoch == 0
                        and self.profiler_profile_memory
                        and idx
                        == self.profiler_wait_steps
                        + self.profiler_warmup_steps
                        + self.profiler_active_steps
                    ):
                        torch.cuda.memory._record_memory_history(enabled=None)

                    # Step profiler
                    # Note that this is called within gradient accumulation block, hence
                    # will include multiple forward / backward passes if gradient accumulation > 1
                    self._profiler.step()

                idx += 1  # NOTE: added by us

            self.epochs_run += 1
            self.save_checkpoint(epoch=curr_epoch)
            # Add after each epoch completes
            if self._is_rank_zero and self.profiler_profile_memory:
                torch.cuda.memory._dump_snapshot(
                    f"memory_snapshot_epoch_{curr_epoch}.pickle"
                )
                torch.cuda.memory._record_memory_history(enabled=None)

        self._profiler.stop()

    def train_q(self) -> None:
        """
        The core training loop.
        """
        # clean up before training begins
        training.cleanup_before_training()

        world_size, rank = training.get_world_size_and_rank()

        # zero out the gradients before starting training
        if not self._optimizer_in_bwd:
            self._optimizer.zero_grad()
        else:
            for opt in self._optim_ckpt_wrapper.optim_map.values():
                opt.zero_grad()

        # Initialize tokens count and running loss (for grad accumulation)
        t0 = time.perf_counter()
        running_loss = 0
        num_tokens = 0

        # NOTE: added by us - sample just once at the beginning of the epoch loop
        self._sampler.set_epoch(0)

        self._profiler.start()
        # self.epochs_run should be non-zero when we're resuming from a checkpoint
        for curr_epoch in range(self.epochs_run, self.total_epochs):
            # Update the sampler to ensure data is correctly shuffled across epochs
            # in case shuffle is True
            # NOTE: removing it from here and putting it before the epoch loop
            # because our epochs are not the same as the dataloader epochs
            for _sampler_validation in self._sampler_validation_list:
                _sampler_validation.set_epoch(curr_epoch)  # NOTE: added by us

            # NOTE: added by us
            # ------ Validation Step ------ #
            self._model.eval()
            self._ref_model.eval()

            with torch.no_grad():
                for i, dataloader_validation in enumerate(
                    self._dataloader_validation_list
                ):
                    for _, batch in enumerate(dataloader_validation):
                        batch.pop("goal", None)
                        batch.pop("privileged_found", None)
                        utils.batch_to_device(batch, self._device)
                        val_loss = torch.tensor(0.0, device=self._device)
                        if self._is_rank_zero:
                            self._metric_logger.log_dict(
                                {f"val_loss_{i}": val_loss.item()},
                                step=self.global_step,
                            )
            del val_loss
            # ------ Reward and Advantage Computation ------ #
            (
                all_rewards,
                all_advantages,
                p_y_g_zx_mean,
                q_y_g_xz_mean,
                q_z_g_xy_mean,
                rewards_by_goal,
                kls,
                p_y_g_zx_all,
            ) = self.compute_rewards()

            # TODO: This is hacky need a better way to do this but this will work for now
            # I think the issue is that the advantages are
            # Use advantages if apply_advantage_in_tune is True, otherwise use rewards
            if self.apply_advantage_in_tune:
                all_advantages = all_advantages  # Use computed advantages
            else:
                all_advantages = all_rewards  # Use rewards directly
            # Log KL divergence statistics
            if self._is_rank_zero:
                kl_mean = np.mean(kls)
                kl_std = np.std(kls)

                log.info(
                    f"KL Divergence Stats - Mean: {kl_mean:.4f}, Std: {kl_std:.4f}"
                )

                self._metric_logger.log_dict(
                    {
                        "kl_divergence_mean": kl_mean,
                        "kl_divergence_std": kl_std,
                        "reward_mean": np.mean(all_rewards),
                        "reward_std": np.std(all_rewards),
                    },
                    step=self.global_step,
                )

            # ------ Training Epoch ------ #
            # Initialize tokens count and running loss (for grad accumulation)
            t0 = time.perf_counter()
            running_loss = 0
            num_tokens = 0
            real_num_tokens = 0
            max_len_samples = 0
            # Update entropy tracking variables to include sum and mean metrics
            running_per_token_ent_sum = 0
            running_full_token_ent_sum = 0
            running_per_token_ent_mean = 0
            running_full_token_ent_mean = 0
            # Track per-step and running stats for tokens
            running_bprop_tokens_sum = 0.0
            running_bprop_tokens_min = float("inf")
            running_bprop_tokens_max = 0.0
            running_bprop_tokens_count = 0
            running_total_tokens_sum = 0.0
            running_total_tokens_min = float("inf")
            running_total_tokens_max = 0.0
            running_total_tokens_count = 0
            self._model.train()  # NOTE: added by us

            pbar = tqdm(
                total=self._steps_per_epoch, disable=not (rank == 0), desc="Training"
            )

            # NOTE: added by us - counter to account for samples that are too long
            idx = 0
            processed_samples = 0
            n_samples = len(self._dataloader)
            n_gpus = torch.distributed.get_world_size()
            number_leftover_samples = (
                n_samples * n_gpus
            ) % self._gradient_accumulation_steps
            for j, batch in enumerate(self._dataloader):
                if ((idx // self._gradient_accumulation_steps)) >= (
                    self._steps_per_epoch
                ) and not self.max_bsize:
                    break
                if j != processed_samples:
                    log.warning(
                        f"Skipping batch {j} as it does not match processed_samples {processed_samples}"
                    )

                train_batch = batch["with_privilege"]
                if self._skip_max_seq_len_samples(
                    train_batch
                ) or self._skip_max_seq_len_samples(batch["without_privilege"]):
                    max_len_samples += 1
                    continue

                # Start tracking CUDA memory for active steps for just the first epoch
                if (
                    self._is_rank_zero
                    and curr_epoch == 0
                    and self.profiler_profile_memory
                    and idx == self.profiler_wait_steps + self.profiler_warmup_steps
                ):
                    torch.cuda.memory._record_memory_history()

                batch.pop("goal", None)
                batch.pop("privileged_found", None)
                utils.batch_to_device(batch, self._device)

                # Calculate the number of unmasked tokens in the current batch
                # and increment the total number of tokens seen in the step
                current_num_tokens = (
                    train_batch["labels"] != self._loss_fn.ignore_index
                ).sum()
                num_tokens += current_num_tokens
                # NOTE: added by us
                # let's monitor the total number of tokens
                # Accumulate total tokens across microbatches in this step
                real_num_tokens += train_batch["labels"].numel()

                # Shape [b, s], needed for the loss not the model
                labels = train_batch.pop("labels")
                action_start_pos = train_batch.pop("action_start_pos", None)
                train_batch.pop("action_end_pos", None)
                train_batch.pop("end_of_prompt", None)
                train_batch.pop("mask", None)
                train_batch.pop("attention_mask", None)

                batch_size = labels.shape[0]
                advantages = all_advantages[j]
                advantages = torch.tensor(advantages, device=self._device)

                # Check if we should skip forward and backward pass for zero rewards
                skip_computation = torch.all(advantages == 0)

                # Build shifted labels once
                labels_shifted = torch.hstack(
                    (
                        labels[..., 1:],
                        self.ignore_labels_cache[: labels.shape[0]],
                    )
                )

                # Importance sampling: use cached per-token ref logprobs from compute_rewards
                ref_logprobs_for_loss = None

                if self.use_importance_sampling:

                    ref_cached = self.reference_logprobs_cache.get(j)
                    if ref_cached is not None:
                        n_chunks = len(ref_cached)
                        # ref_logprobs_for_loss = torch.cat(ref_cached,dim=1)[:,:action_start_pos].chunk(n_chunks, dim=1)
                        ref_logprobs_for_loss = ref_cached

                with self.activations_handling_ctx:
                    logits = self._model(**train_batch)

                    # logits = [logit / self.sampling_temperature for logit in logits]
                # Use shifted labels for loss
                labels = labels_shifted
                # Ensure logits are chunked to match loss expectations                if not isinstance(logits, list):                if not isinstance(logits, list):
                if not isinstance(logits, list):
                    labels = labels.reshape(-1)
                    logits = logits.reshape(-1, logits.size(-1))

                # index for labels of the cot only

                # logits = index_chunkced(logits,  action_start_pos)
                # labels = labels[:, :action_start_pos]

                combined_loss = (
                    self._loss_fn(
                        logits=logits,
                        labels=labels,
                        reward=advantages,
                        ref_logprobs=ref_logprobs_for_loss,
                        # Provide PPO bounds (low<1, high>1) to enable masking inside the loss
                        epsilon_low=self.epsilon_low_neg,
                        epsilon_high=self.epsilon_high_pos,
                    )
                    * current_num_tokens
                )
                running_loss += combined_loss.detach() * batch_size

                # For optimizer in backward, we need to normalize before calling backward
                # This case and gradient accumulation are mutually exclusive
                if self._optimizer_in_bwd:
                    torch.distributed.all_reduce(num_tokens)
                    torch.distributed.all_reduce(running_loss)
                    combined_loss = combined_loss / num_tokens
                combined_loss.backward()
                del combined_loss

                processed_samples += batch_size
                # Step with optimizer
                if (idx + 1) % self._gradient_accumulation_steps == 0 or (
                    (idx + 1) == n_samples
                ):

                    if not self._optimizer_in_bwd:
                        # Get total number of tokens across all ranks to normalize gradients
                        torch.distributed.all_reduce(num_tokens)
                        # This will ensure that the logged loss matches what we're optimizing
                        torch.distributed.all_reduce(running_loss)
                        # All-reduce all entropy metrics

                        # Manually scale the gradients from unnormalized loss by total # of tokens
                        training.scale_grads(self._model, 1 / num_tokens)
                        # scale grads by max_batchsize and real_batchsize
                        if self.max_bsize and (idx + 1) == n_samples:
                            if number_leftover_samples == 1:
                                number_leftover_samples = n_samples
                            scaler = torch.tensor(
                                number_leftover_samples / self.max_bsize
                                if number_leftover_samples > 0
                                else n_samples / self.max_bsize
                            )

                            training.scale_grads(
                                self._model,
                                scaler,
                            )
                            log.info(
                                f"Scaling gradients by {scaler} Original bsize = {number_leftover_samples}"
                            )

                        # Calculate gradient norms before clipping (efficient way)
                        total_norm = torch.nn.utils.clip_grad_norm_(
                            self._model.parameters(), max_norm=float("inf")
                        )

                        grad_norm_stats = {"grad_norm_total": total_norm.item()}

                        if self._clip_grad_norm is not None:
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                self._model.parameters(),
                                max_norm=float(self._clip_grad_norm),
                            )
                        self._optimizer.step()
                        log.info(f"optimizer step")
                        self._optimizer.zero_grad(set_to_none=True)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                            gc.collect()
                            torch.cuda.synchronize()

                    # Update the number of steps when the weights are updated
                    self.global_step += 1

                    # Step the learning rate scheduler
                    if self._lr_scheduler is not None:
                        self._lr_scheduler.step()

                    loss_to_log = running_loss.item() / num_tokens
                    pbar.update(1)
                    pbar.set_description(
                        f"{curr_epoch + 1}|{self.global_step}|Loss: {loss_to_log}"
                    )
                    n_samples
                    # Log per-step metrics

                    if self._is_rank_zero:
                        time_per_step = time.perf_counter() - t0
                        log_dict = {
                            "loss": loss_to_log.cpu().item(),
                            "lr": get_lr(
                                (
                                    self._optimizer
                                    if not self._optimizer_in_bwd
                                    else self._optim_ckpt_wrapper
                                ),
                            ),
                            "tokens_per_second_per_gpu": real_num_tokens  # NOTE: added by us
                            / (time_per_step * world_size),
                            "q_y_g_xz_mean": q_y_g_xz_mean,
                            "q_z_g_xy_mean": q_z_g_xy_mean,
                            "p_y_g_zx_mean": p_y_g_zx_mean,
                        }
                        # Add gradient norm stats to logging
                        log_dict.update(grad_norm_stats)
                        if self._log_peak_memory_stats:
                            log_dict.update(
                                training.get_memory_stats(device=self._device)
                            )
                        if self._clip_grad_norm is not None:
                            log_dict.update({"grad_norm": grad_norm})
                        self._metric_logger.log_dict(
                            log_dict,
                            step=self.global_step,
                        )

                    # Reset running stats for the next step
                    running_loss = 0
                    combined_loss = 0
                    num_tokens = 0
                    real_num_tokens = 0
                    running_per_token_ent_sum = 0
                    running_full_token_ent_sum = 0
                    running_per_token_ent_mean = 0
                    running_full_token_ent_mean = 0
                    t0 = time.perf_counter()

                    # Stop tracking CUDA memory now that active steps are complete
                    if (
                        self._is_rank_zero
                        and curr_epoch == 0
                        and self.profiler_profile_memory
                        and idx
                        == self.profiler_wait_steps
                        + self.profiler_warmup_steps
                        + self.profiler_active_steps
                    ):
                        torch.cuda.memory._record_memory_history(enabled=None)

                    # Step profiler
                    # Note that this is called within gradient accumulation block, hence
                    # will include multiple forward / backward passes if gradient accumulation > 1
                    self._profiler.step()

                idx += 1  # NOTE: added by us

            self.epochs_run += 1
            self.save_checkpoint(epoch=curr_epoch)
            # Add after each epoch completes
            if self._is_rank_zero and self.profiler_profile_memory:
                torch.cuda.memory._dump_snapshot(
                    f"memory_snapshot_epoch_{curr_epoch}.pickle"
                )
                torch.cuda.memory._record_memory_history(enabled=None)

        self._profiler.stop()

    def process_logits_memory_efficient(self, logits, labels, ignore_index):
        """Memory-efficient logits processing that minimizes copies"""
        if isinstance(logits, list):
            # Process chunks sequentially to minimize peak memory
            all_valid_logits = []
            all_valid_labels = []

            labels_chunks = labels.chunk(len(logits), dim=1)

            for i, (logit_chunk, label_chunk) in enumerate(zip(logits, labels_chunks)):
                # Process one chunk at a time
                logit_2d = logit_chunk.view(-1, logit_chunk.size(-1))
                label_1d = label_chunk.view(-1)

                # Get valid mask
                valid_mask = label_1d != ignore_index

                if valid_mask.any():
                    # Use the most memory-efficient selection
                    all_valid_logits.append(logit_2d[valid_mask])
                    all_valid_labels.append(label_1d[valid_mask])

                # Free intermediate tensors immediately
                del logit_2d, label_1d, valid_mask

            if all_valid_logits:
                concatenated_logits = torch.cat(all_valid_logits, dim=0)
                concatenated_labels = torch.cat(all_valid_labels, dim=0)

                # Re-chunk if originally was a list
                original_chunks = len(logits)
                chunk_size = (
                    concatenated_logits.size(0) + original_chunks - 1
                ) // original_chunks
                rechunked_logits = list(concatenated_logits.split(chunk_size, dim=0))

                return rechunked_logits, concatenated_labels.unsqueeze(0)
            else:
                return [], torch.empty(0, dtype=labels.dtype, device=labels.device)
        else:
            # Handle tensor case
            logit_2d = logits.view(-1, logits.size(-1))
            label_1d = labels.view(-1)
            valid_mask = label_1d != ignore_index

            if valid_mask.any():
                return logit_2d[valid_mask], label_1d[valid_mask]
            else:
                return torch.empty(
                    (0, logits.size(-1)), dtype=logits.dtype, device=logits.device
                ), torch.empty(0, dtype=labels.dtype, device=labels.device)

    def calculate_ppo_clipped_importance_sampling(
        self,
        wp_priv_logits: List[torch.Tensor],
        wop_logits: List[torch.Tensor],
        labels: torch.Tensor,
        labels_np: torch.Tensor,
        advantages: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate PPO-style clipped importance sampling ratios.
        This version handles logits of different lengths by finding the intersection
        of valid (non-ignored) tokens. This implementation is optimized for memory
        by processing chunk-by-chunk.

        Args:
            wp_priv_logits (List[torch.Tensor]): Logits from the model with privilege (list of chunks).
            wop_logits (List[torch.Tensor]): Logits from the model without privilege (list of chunks).
            labels (torch.Tensor): Ground truth labels for the privileged model.
            labels_np (torch.Tensor): Ground truth labels for the non-privileged model.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the unclipped and clipped importance ratios (summed).
        """
        import torch.nn.functional as F

        wp_selected_log_ps_list = []
        wop_selected_log_ps_list = []

        # Process each chunk individually to save memory
        labels_chunks = labels.view(1, -1).chunk(len(wp_priv_logits), dim=1)
        labels_np_chunks = labels_np.view(1, -1).chunk(len(wop_logits), dim=1)

        for i, (wp_chunk, wop_chunk) in enumerate(zip(wp_priv_logits, wop_logits)):
            # For each chunk, we expect labels to be 1D
            labels_flat = labels_chunks[i].view(-1)
            labels_np_flat = labels_np_chunks[i].view(-1)

            # Find valid indices within the current chunk's scope
            valid_mask_priv = labels_flat != self._loss_fn.ignore_index
            valid_mask_wop = labels_np_flat != self._loss_fn.ignore_index

            # Ensure we are comparing the same tokens
            assert (
                valid_mask_priv.sum() == valid_mask_wop.sum()
            ), "Valid token counts do not match between privileged and non-privileged"

            # Filter logits and labels for valid tokens
            wp_logits_valid = wp_chunk[valid_mask_priv]
            wop_logits_valid = wop_chunk[valid_mask_wop]
            labels_valid = labels_flat[valid_mask_priv]

            if labels_valid.numel() == 0:
                continue

            # Compute log softmax
            wp_log_ps = F.log_softmax(wp_logits_valid, dim=-1)
            wop_log_ps = F.log_softmax(wop_logits_valid, dim=-1)

            # Gather log-probabilities for the correct tokens
            vocab_size = wp_log_ps.size(-1)
            valid_indices = labels_valid.clamp(0, vocab_size - 1).unsqueeze(-1)

            wp_selected_log_ps = torch.gather(
                wp_log_ps, dim=-1, index=valid_indices
            ).squeeze(-1)
            wop_selected_log_ps = torch.gather(
                wop_log_ps, dim=-1, index=valid_indices
            ).squeeze(-1)

            wp_selected_log_ps_list.append(wp_selected_log_ps)
            wop_selected_log_ps_list.append(wop_selected_log_ps)

        # Concatenate results from all chunks
        if not wp_selected_log_ps_list:
            # Handle case with no valid tokens
            return torch.tensor(0.0, device=wp_priv_logits[0].device), torch.tensor(
                0.0, device=wp_priv_logits[0].device
            )

        wp_selected_log_ps_final = torch.cat(wp_selected_log_ps_list)
        wop_selected_log_ps_final = torch.cat(wop_selected_log_ps_list)

        # Calculate the importance ratio
        ratio = torch.exp(wop_selected_log_ps_final - wp_selected_log_ps_final)

        # Clip the ratio
        clipped_ratio = torch.clamp(ratio, self.epsilon_low_neg, self.epsilon_high_pos)

        # Track clipping statistics
        if not hasattr(self, "ppo_clip_stats_tracker"):
            self.ppo_clip_stats_tracker = []

        total_tokens = ratio.numel()
        clipped_tokens = (
            ((ratio < self.epsilon_low_neg) | (ratio > self.epsilon_high_pos))
            .sum()
            .item()
        )

        log.info(
            f"Total tokens: {total_tokens}, Clipped tokens: {clipped_tokens} Percentage: {clipped_tokens/total_tokens*100:.2f}%"
        )

        self.ppo_clip_stats_tracker.append(
            {
                "total_tokens": total_tokens,
                "clipped_tokens": clipped_tokens,
                "clipped_low": (ratio < self.epsilon_low_neg).sum().item(),
                "clipped_high": (ratio > self.epsilon_high_pos).sum().item(),
                "mean_ratio": ratio.mean().detach().cpu().item(),
            }
        )

        # PPO policy loss with advantages (standard PPO objective)
        # L^CLIP = -E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages)

        del (
            wp_selected_log_ps_final,
            wop_selected_log_ps_final,
            wp_selected_log_ps_list,
            wop_selected_log_ps_list,
            wp_logits_valid,
            wop_logits_valid,
            ratio,
            clipped_ratio,
        )
        # Return summed policy loss
        return policy_loss.sum()

    def calculate_topr_loss(
        self,
        logits: List[torch.Tensor],
        labels: torch.Tensor,
        advantages: torch.Tensor,
        ref_logits: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Calculate TOPR (Tapered Off-Policy Reinforcement Learning) loss.

        For positive advantages (A > 0): SFT loss weighted by advantage
        For negative advantages (A ≤ 0): Truncated Importance Sampling (TIS)

        From the paper: https://arxiv.org/pdf/2503.14286
        ∇J_topr(π) = Σ_{τ∈T+} μ(τ)R(τ)∇log π(τ) + Σ_{τ∈T-} μ(τ) * clip(π/μ, ε_low, ε_high) * R(τ) * ∇log π(τ)

        Args:
            logits: Current policy logits (list of chunks)
            labels: Ground truth labels
            advantages: Advantage values per sample
            ref_logits: Optional reference policy logits for TIS (used for negative advantages)

        Returns:
            Combined TORPO loss (summed over valid tokens)
        """
        import torch.nn.functional as F

        # Process chunks
        labels_chunks = labels.view(1, -1).chunk(len(logits), dim=1)

        policy_log_ps_list = []
        ref_log_ps_list = []

        # Gather log-probs chunk by chunk
        for i, logit_chunk in enumerate(logits):
            labels_flat = labels_chunks[i].view(-1)
            valid_mask = labels_flat != self._loss_fn.ignore_index

            if not valid_mask.any():
                continue

            # Filter to valid tokens
            logit_valid = logit_chunk[valid_mask]
            labels_valid = labels_flat[valid_mask]

            # Current policy log-probs
            log_ps = F.log_softmax(logit_valid, dim=-1)
            vocab_size = log_ps.size(-1)
            valid_indices = labels_valid.clamp(0, vocab_size - 1).unsqueeze(-1)
            selected_log_ps = torch.gather(log_ps, dim=-1, index=valid_indices).squeeze(
                -1
            )
            policy_log_ps_list.append(selected_log_ps)

            # Reference policy log-probs (for negative advantages)
            if ref_logits is not None:
                ref_logit_chunk = ref_logits[i]
                ref_logit_valid = ref_logit_chunk[valid_mask]
                ref_log_ps = F.log_softmax(ref_logit_valid, dim=-1)
                ref_selected_log_ps = torch.gather(
                    ref_log_ps, dim=-1, index=valid_indices
                ).squeeze(-1)
                ref_log_ps_list.append(ref_selected_log_ps)

        # Handle empty case
        if not policy_log_ps_list:
            return torch.tensor(0.0, device=logits[0].device)

        # Concatenate all valid tokens
        policy_log_ps = torch.cat(policy_log_ps_list)

        # Separate positive and negative advantages
        positive_mask = advantages > 0
        negative_mask = advantages <= 0

        total_loss = torch.tensor(0.0, device=logits[0].device, requires_grad=True)

        # ===== Positive samples: SFT loss weighted by advantage =====
        # L_positive = -Σ A * log π(τ) for A > 0
        if positive_mask.any():
            positive_loss = -(advantages * policy_log_ps).sum()
            total_loss = total_loss + positive_loss
            log.info(f"TORPO positive loss: {positive_loss.item():.4f}")

        # ===== Negative samples: Truncated Importance Sampling =====
        # L_negative = -Σ clip(π/μ, ε_low, ε_high) * A * log π(τ) for A ≤ 0
        if negative_mask.any() and ref_logits is not None:
            ref_log_ps = torch.cat(ref_log_ps_list)

            # Importance weight: π(τ) / μ(τ)
            # CRITICAL: Detach policy_log_ps for unbiased gradient estimation
            importance_weight = torch.exp(policy_log_ps.detach() - ref_log_ps)

            # Clip importance weight for negative samples
            clipped_weight = torch.clamp(
                importance_weight, min=self.epsilon_low_neg, max=self.epsilon_high_neg
            )

            # TIS loss: -clipped_weight * advantage * log π(τ)
            # The gradient flows only through policy_log_ps, not through clipped_weight
            negative_loss = -(clipped_weight * advantages * policy_log_ps).sum()
            total_loss = total_loss + negative_loss

            # Log statistics
            clipped_tokens = (
                (
                    (importance_weight < self.epsilon_low_neg)
                    | (importance_weight > self.epsilon_high_neg)
                )
                .sum()
                .item()
            )
            log.info(
                f"TORPO negative loss: {negative_loss.item():.4f}, "
                f"Mean importance weight: {importance_weight.mean().item():.4f}, "
                f"Clipped tokens: {clipped_tokens}/{importance_weight.numel()}"
            )

        elif negative_mask.any():
            # Fallback: treat as SFT if no reference available
            negative_loss = -(advantages * policy_log_ps).sum()
            total_loss = total_loss + negative_loss
            log.info(f"TORPO negative loss (no ref): {negative_loss.item():.4f}")

        return total_loss

    def train_q_online_run(self, train_q=True, train_p=False) -> None:
        """
        The core training loop.
        """
        # clean up before training begins
        training.cleanup_before_training()

        world_size, rank = training.get_world_size_and_rank()

        # zero out the gradients before starting training
        if not self._optimizer_in_bwd:
            self._optimizer.zero_grad()
        else:
            for opt in self._optim_ckpt_wrapper.optim_map.values():
                opt.zero_grad()

        # Initialize tokens count and running loss (for grad accumulation)
        t0 = time.perf_counter()
        running_loss = 0
        num_tokens = 0

        # NOTE: added by us - sample just once at the beginning of the epoch loop
        self._sampler.set_epoch(0)

        self._profiler.start()
        # self.epochs_run should be non-zero when we're resuming from a checkpoint
        for curr_epoch in range(self.epochs_run, self.total_epochs):
            # Log the current online training coefficient for this epoch
            if self._is_rank_zero and train_q and train_p:
                log.info("=" * 80)
                log.info(
                    f"Epoch {curr_epoch}: Online training coefficient = {self.online_training_coeff:.4f}"
                )
                log.info(f"  Q loss weight: {1.0 - self.online_training_coeff:.4f}")
                log.info(f"  P loss weight: {self.online_training_coeff:.4f}")
                log.info("=" * 80)

            # Update the sampler to ensure data is correctly shuffled across epochs
            # in case shuffle is True
            # NOTE: removing it from here and putting it before the epoch loop
            # because our epochs are not the same as the dataloader epochs
            for _sampler_validation in self._sampler_validation_list:
                _sampler_validation.set_epoch(curr_epoch)  # NOTE: added by us

            # NOTE: added by us
            # ------ Validation Step ------ #
            self._model.eval()
            self._ref_model.eval()

            with torch.no_grad():
                for i, dataloader_validation in enumerate(
                    self._dataloader_validation_list
                ):
                    for _, batch in enumerate(dataloader_validation):
                        batch.pop("goal", None)
                        batch.pop("privileged_found", None)
                        utils.batch_to_device(batch, self._device)
                        val_loss = torch.tensor(0.0, device=self._device)
                        if self._is_rank_zero:
                            self._metric_logger.log_dict(
                                {f"val_loss_{i}": val_loss.item()},
                                step=self.global_step,
                            )
            del val_loss
            # ------ Reward and Advantage Computation ------ #
            (
                all_rewards,
                all_advantages,
                p_y_g_zx_mean,
                q_y_g_xz_mean,
                q_z_g_xy_mean,
                rewards_by_goal,
                kls,
                p_y_g_zx_all,
            ) = self.compute_rewards()

            # TODO: This is hacky need a better way to do this but this will work for now
            # I think the issue is that the advantages are
            # Use advantages if apply_advantage_in_tune is True, otherwise use rewards
            if self.apply_advantage_in_tune:
                all_advantages = all_advantages  # Use computed advantages
            else:
                all_advantages = all_rewards  # Use rewards directly
            # Log KL divergence statistics
            if self._is_rank_zero:
                kl_mean = np.mean(kls)
                kl_std = np.std(kls)

                log.info(
                    f"KL Divergence Stats - Mean: {kl_mean:.4f}, Std: {kl_std:.4f}"
                )

                self._metric_logger.log_dict(
                    {
                        "kl_divergence_mean": kl_mean,
                        "kl_divergence_std": kl_std,
                        "reward_mean": np.mean(all_rewards),
                        "reward_std": np.std(all_rewards),
                    },
                    step=self.global_step,
                )

            # ------ Training Epoch ------ #
            # Initialize tokens count and running loss (for grad accumulation)
            t0 = time.perf_counter()
            running_loss = 0
            num_tokens = 0
            real_num_tokens = 0
            max_len_samples = 0
            # Update entropy tracking variables to include sum and mean metrics
            running_per_token_ent_sum = 0
            running_full_token_ent_sum = 0
            running_per_token_ent_mean = 0
            running_full_token_ent_mean = 0
            # Track per-step and running stats for tokens
            running_bprop_tokens_sum = 0.0
            running_bprop_tokens_min = float("inf")
            running_bprop_tokens_max = 0.0
            running_bprop_tokens_count = 0
            running_total_tokens_sum = 0.0
            running_total_tokens_min = float("inf")
            running_total_tokens_max = 0.0
            running_total_tokens_count = 0
            self._model.train()  # NOTE: added by us

            pbar = tqdm(
                total=self._steps_per_epoch, disable=not (rank == 0), desc="Training"
            )

            # NOTE: added by us - counter to account for samples that are too long
            idx = 0
            processed_samples = 0
            n_samples = len(self._dataloader)
            n_gpus = torch.distributed.get_world_size()
            number_leftover_samples = (
                n_samples * n_gpus
            ) % self._gradient_accumulation_steps
            agent_tokens_tracker = []
            total_tokens_avg_tracker = []
            pbar = tqdm(
                total=self._steps_per_epoch, disable=not (rank == 0), desc="Training"
            )
            for j, batch in enumerate(self._dataloader):
                if ((idx // self._gradient_accumulation_steps)) >= (
                    self._steps_per_epoch
                ) and not self.max_bsize:
                    break
                if j != processed_samples:
                    log.warning(
                        f"Skipping batch {j} as it does not match processed_samples {processed_samples}"
                    )

                train_batch = batch["with_privilege"]
                train_batch_np = batch["without_privilege"]
                # if self._skip_max_seq_len_samples(
                #     train_batch
                # ) or self._skip_max_seq_len_samples(batch["without_privilege"]):
                #     max_len_samples += 1
                #     continue

                # Start tracking CUDA memory for active steps for just the first epoch
                if (
                    self._is_rank_zero
                    and curr_epoch == 0
                    and self.profiler_profile_memory
                    and idx == self.profiler_wait_steps + self.profiler_warmup_steps
                ):
                    torch.cuda.memory._record_memory_history()

                batch.pop("goal", None)
                batch.pop("privileged_found", None)
                og_reward = batch.pop("og_reward", None)
                utils.batch_to_device(batch, self._device)

                # Calculate the number of unmasked tokens in the current batch
                # and increment the total number of tokens seen in the step
                current_num_tokens = (
                    train_batch["labels"] != self._loss_fn.ignore_index
                ).sum()
                current_num_tokens_track = current_num_tokens.item()
                # NOTE: added by us
                # let's monitor the total number of tokens
                # Accumulate total tokens across microbatches in this step
                real_num_tokens += train_batch["labels"].numel()

                # Shape [b, s], needed for the loss not the model
                labels = train_batch.pop("labels")
                action_start_pos = train_batch.pop("action_start_pos", None)
                train_batch.pop("action_end_pos", None)
                train_batch.pop("end_of_prompt", None)
                train_batch.pop("mask", None)
                train_batch.pop("attention_mask", None)

                labels_np = train_batch_np.pop("labels")
                action_start_pos_np = train_batch_np.pop("action_start_pos", None)
                train_batch_np.pop("action_end_pos", None)
                train_batch_np.pop("end_of_prompt", None)
                train_batch_np.pop("mask", None)
                train_batch_np.pop("attention_mask", None)
                utils.batch_to_device(train_batch_np, self._device)

                batch_size = labels.shape[0]
                total_tokens_avg_tracker.append(real_num_tokens / batch_size)
                agent_tokens_tracker.append(current_num_tokens_track / batch_size)
                advantages = all_advantages[j]
                advantages = torch.tensor(advantages, device=self._device)
                rank = torch.distributed.get_rank()
                # Check if we should skip forward and backward pass for zero rewards
                skip_computation = og_reward == 0
                log.info(f"Current number of tokens {current_num_tokens}")
                num_tokens += current_num_tokens

                # Build shifted labels once
                log.info(f"{current_num_tokens=}")
                labels_shifted = torch.hstack(
                    (
                        labels[..., 1:],
                        self.ignore_labels_cache[: labels.shape[0]],
                    )
                )
                labels_shifted_np = torch.hstack(
                    (
                        labels_np[..., 1:],
                        self.ignore_labels_cache[: labels_np.shape[0]],
                    )
                )

                # Importance sampling: use cached per-token ref logprobs from compute_rewards
                ref_logprobs_for_loss = None
                if self.use_importance_sampling:
                    ref_cached = self.reference_logprobs_cache.get(j)
                    if ref_cached is not None:
                        n_chunks = len(ref_cached)
                        ref_logprobs_for_loss = ref_cached

                total_loss = torch.tensor(0.0, device=self._device, requires_grad=True)

                # Always compute forward with gradients for distributed sync
                with self.activations_handling_ctx:
                    logits = self._model(**train_batch)

                # Prepare labels for first loss
                labels = labels_shifted

                # Use memory-efficient processing
                logits, labels = self.process_logits_memory_efficient(
                    logits, labels, self._loss_fn.ignore_index
                )

                # First loss computation and backward pass
                # Skip if there are no valid tokens to compute loss on

                combined_loss = (
                    self._loss_fn(
                        logits=logits,
                        labels=labels,
                        reward=advantages,
                        ref_logprobs=ref_logprobs_for_loss,
                        # Provide PPO bounds (low<1, high>1) to enable masking inside the loss
                        epsilon_low=self.epsilon_low_neg,
                        epsilon_high=self.epsilon_high_pos,
                    )
                    * current_num_tokens
                )
                log.info(f"Calculated combined loss {combined_loss.item()}")

                if train_p:

                    logits_with_priv = (
                        [l.clone().detach() for l in logits]
                        if isinstance(logits, list)
                        else logits.clone().detach()
                    )
                else:
                    logits_with_priv = None

                # Handle loss normalization based on whether we're training Q
                if not train_q:
                    # Detach so backward won't affect Q model parameters
                    combined_loss_detached = combined_loss.detach()
                    if self._optimizer_in_bwd:
                        torch.distributed.all_reduce(num_tokens)
                        first_loss_normalized = combined_loss_detached / num_tokens
                    else:
                        first_loss_normalized = combined_loss_detached

                    # Manually clear the activation offloading tracker
                    # since we're not doing a real backward pass
                    if hasattr(self.activations_handling_ctx, "__enter__"):
                        # Clear the tracker if using activation offloading
                        try:
                            from torchtune.training._activation_offloading import (
                                OffloadActivations,
                            )

                            if isinstance(
                                self.activations_handling_ctx, OffloadActivations
                            ):
                                self.activations_handling_ctx.tracker.clear()
                        except (ImportError, AttributeError):
                            pass
                else:
                    if self._optimizer_in_bwd:
                        torch.distributed.all_reduce(num_tokens)
                        first_loss_normalized = combined_loss / num_tokens
                    else:
                        first_loss_normalized = combined_loss

                # Backward pass for Q loss to clear activation offloading tracker
                # Apply online training coefficient weighting: (1 - coeff) * Q_loss
                if train_q and train_p:
                    q_weight = 1.0 - self.online_training_coeff
                    weighted_first_loss = first_loss_normalized * q_weight
                    weighted_first_loss.backward()
                    if self._is_rank_zero and j == 0:
                        log.info(
                            f"Q loss weighted by {q_weight:.4f}: original={first_loss_normalized.item():.4f}, weighted={weighted_first_loss.item():.4f}"
                        )
                elif train_q:
                    first_loss_normalized.backward()
                # If not train_q, don't run backward at all - tracker already cleared above

                total_loss = total_loss + combined_loss.detach()

                del logits, combined_loss, first_loss_normalized
                # if torch.cuda.is_available():
                #     torch.cuda.empty_cache()
                # Basic memory usage
                # allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                # reserved = torch.cuda.memory_reserved() / 1024**3  # GB
                # log.info(
                #     f"CUDA Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
                # )
                if train_p:
                    # log.info("training_p")
                    # Second forward pass: without privilege (always execute for distributed sync)
                    with self.activations_handling_ctx:
                        logits_np = self._model(**train_batch_np)

                    # we always use batchsize of 1 extract logits where the labels are not -100

                    # Prepare labels for second loss
                    labels_np = labels_shifted_np

                    # Use memory-efficient processing
                    logits_np, labels_np = self.process_logits_memory_efficient(
                        logits_np, labels_np, self._loss_fn.ignore_index
                    )

                    # Choose loss function based on configuration
                    if self.train_p_loss == "trpo":
                        log.info("Using TORPO loss for P model training")
                        # For TORPO, we need reference logits for negative advantages
                        ref_logits_np = None
                        if advantages <= 0 and logits_with_priv is not None:
                            # Use privileged logits as reference for negative advantages
                            ref_logits_np = (
                                logits_with_priv
                                if isinstance(logits_with_priv, list)
                                else [logits_with_priv]
                            )

                        combined_loss_np = self.calculate_topr_loss(
                            logits=(
                                logits_with_priv
                                if isinstance(logits_np, list)
                                else [logits_np]
                            ),
                            labels=labels_np,
                            advantages=advantages,
                            ref_logits=ref_logits_np,
                        )
                    else:  # default to PPO
                        log.info("Using PPO loss for P model training")
                        combined_loss_np = self.calculate_ppo_clipped_importance_sampling(  # noqa: E501
                            wp_priv_logits=(
                                logits_with_priv
                                if isinstance(logits_with_priv, list)
                                else [logits_with_priv]
                            ),  # noqa: E501
                            wop_logits=(
                                logits_np
                                if isinstance(logits_np, list)
                                else [logits_np]
                            ),
                            labels=labels,
                            labels_np=labels_np,
                            advantages=advantages,
                        )
                    log.info(
                        f"Calculated combined loss with np model ({self.train_p_loss}): {combined_loss_np.item()}"
                    )
                    del logits_with_priv, labels

                    # Normalize and backward for second pass
                    if self._optimizer_in_bwd:
                        second_loss_normalized = combined_loss_np / num_tokens
                    else:
                        second_loss_normalized = combined_loss_np

                    # Apply online training coefficient weighting: coeff * P_loss
                    if train_q and train_p:
                        p_weight = self.online_training_coeff
                        weighted_second_loss = second_loss_normalized * p_weight
                        weighted_second_loss.backward()
                        if self._is_rank_zero and j == 0:
                            log.info(
                                f"P loss weighted by {p_weight:.4f}: original={second_loss_normalized.item():.4f}, weighted={weighted_second_loss.item():.4f}"
                            )
                    else:
                        second_loss_normalized.backward()

                    total_loss = total_loss + combined_loss_np.detach()
                    del (
                        logits_np,
                        combined_loss_np,
                        labels_np,
                        second_loss_normalized,
                    )
                else:
                    del logits_with_priv

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                running_loss += total_loss * batch_size

                # Handle distributed reduction for running_loss only (gradients already accumulated)
                if self._optimizer_in_bwd:
                    torch.distributed.all_reduce(running_loss)
                # Final memory cleanup

                del total_loss

                processed_samples += batch_size
                # Step with optimizer
                if (idx + 1) % self._gradient_accumulation_steps == 0 or (
                    (idx + 1) == n_samples
                ):

                    if not self._optimizer_in_bwd:
                        # Get total number of tokens across all ranks to normalize gradients
                        torch.distributed.all_reduce(num_tokens)
                        # This will ensure that the logged loss matches what we're optimizing
                        torch.distributed.all_reduce(running_loss)
                        # All-reduce all entropy metrics

                        # Manually scale the gradients from unnormalized loss by total # of tokens
                        training.scale_grads(self._model, 1 / num_tokens)
                        # scale grads by max_batchsize and real_batchsize
                        if self.max_bsize and (idx + 1) == n_samples:
                            if number_leftover_samples == 1:
                                number_leftover_samples = n_samples
                            # scaler = torch.tensor(
                            #     number_leftover_samples / self.max_bsize
                            #     if number_leftover_samples > 0
                            #     else n_samples / self.max_bsize
                            # )
                            scaler = torch.tensor(
                                1, device=self._device, dtype=torch.float32
                            )

                            training.scale_grads(
                                self._model,
                                scaler,
                            )
                            log.info(
                                f"Scaling gradients by {scaler} Original bsize = {number_leftover_samples}"
                            )

                        # Calculate gradient norms before clipping (efficient way)
                        total_norm = torch.nn.utils.clip_grad_norm_(
                            self._model.parameters(), max_norm=float("inf")
                        )

                        grad_norm_stats = {"grad_norm_total": total_norm.item()}

                        if self._clip_grad_norm is not None:
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                self._model.parameters(),
                                max_norm=float(self._clip_grad_norm),
                            )
                        self._optimizer.step()
                        log.info(f"optimizer step")
                        self._optimizer.zero_grad(set_to_none=True)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                            gc.collect()
                            torch.cuda.synchronize()

                    # Update the number of steps when the weights are updated
                    self.global_step += 1

                    # Step the learning rate scheduler
                    if self._lr_scheduler is not None:
                        self._lr_scheduler.step()

                    loss_to_log = running_loss.item() / num_tokens
                    pbar.update(1)
                    pbar.set_description(
                        f"{curr_epoch + 1}|{self.global_step}|Loss: {loss_to_log}"
                    )
                    # Log per-step metrics

                    if self._is_rank_zero:
                        agent_num_tokens_avg = (
                            sum(agent_tokens_tracker) / len(agent_tokens_tracker)
                            if len(agent_tokens_tracker) > 0
                            else 0.0
                        )
                        total_num_tokens_avg = (
                            sum(total_tokens_avg_tracker)
                            / len(total_tokens_avg_tracker)
                            if len(total_tokens_avg_tracker) > 0
                            else 0.0
                        )
                        time_per_step = time.perf_counter() - t0
                        log_dict = {
                            "loss": loss_to_log.cpu().item(),
                            "lr": get_lr(
                                (
                                    self._optimizer
                                    if not self._optimizer_in_bwd
                                    else self._optim_ckpt_wrapper
                                ),
                            ),
                            "tokens_per_second_per_gpu": real_num_tokens  # NOTE: added by us
                            / (time_per_step * world_size),
                            "agent_num_tokens_avg": agent_num_tokens_avg,
                            "total_num_tokens_avg": total_num_tokens_avg,
                            # Token stats (per local step)
                            "q_y_g_xz_mean": q_y_g_xz_mean,
                            "q_z_g_xy_mean": q_z_g_xy_mean,
                            "p_y_g_zx_mean": p_y_g_zx_mean,
                        }

                        # Aggregate PPO clipping statistics
                        if (
                            hasattr(self, "ppo_clip_stats_tracker")
                            and self.ppo_clip_stats_tracker
                        ):
                            total_tokens_sum = sum(
                                s["total_tokens"] for s in self.ppo_clip_stats_tracker
                            )
                            clipped_tokens_sum = sum(
                                s["clipped_tokens"] for s in self.ppo_clip_stats_tracker
                            )
                            clipped_low_sum = sum(
                                s["clipped_low"] for s in self.ppo_clip_stats_tracker
                            )
                            clipped_high_sum = sum(
                                s["clipped_high"] for s in self.ppo_clip_stats_tracker
                            )
                            mean_ratio_avg = (
                                sum(
                                    s["mean_ratio"] * s["total_tokens"]
                                    for s in self.ppo_clip_stats_tracker
                                )
                                / total_tokens_sum
                                if total_tokens_sum > 0
                                else 0.0
                            )

                            log_dict.update(
                                {
                                    "ppo_clipped_tokens": clipped_tokens_sum,
                                    "ppo_total_tokens": total_tokens_sum,
                                    "ppo_clipped_low": clipped_low_sum,
                                    "ppo_clipped_high": clipped_high_sum,
                                    "ppo_clip_percentage": (
                                        100.0 * clipped_tokens_sum / total_tokens_sum
                                        if total_tokens_sum > 0
                                        else 0.0
                                    ),
                                    "ppo_mean_ratio": mean_ratio_avg,
                                }
                            )

                            log.info(
                                f"PPO Clip - Total: {total_tokens_sum}, Clipped: {clipped_tokens_sum} "
                                f"({100.0 * clipped_tokens_sum / total_tokens_sum:.2f}%), "
                                f"Low: {clipped_low_sum}, High: {clipped_high_sum}"
                            )

                            self.ppo_clip_stats_tracker = []

                        # Add gradient norm stats to logging
                        log_dict.update(grad_norm_stats)
                        if self._log_peak_memory_stats:
                            log_dict.update(
                                training.get_memory_stats(device=self._device)
                            )
                        if self._clip_grad_norm is not None:
                            log_dict.update({"grad_norm": grad_norm})
                        self._metric_logger.log_dict(
                            log_dict,
                            step=self.global_step,
                        )

                    # Reset running stats for the next step
                    running_loss = 0
                    combined_loss = 0
                    num_tokens = 0
                    real_num_tokens = 0
                    running_per_token_ent_sum = 0
                    running_full_token_ent_sum = 0
                    running_per_token_ent_mean = 0
                    running_full_token_ent_mean = 0
                    t0 = time.perf_counter()

                    # Stop tracking CUDA memory now that active steps are complete
                    if (
                        self._is_rank_zero
                        and curr_epoch == 0
                        and self.profiler_profile_memory
                        and idx
                        == self.profiler_wait_steps
                        + self.profiler_warmup_steps
                        + self.profiler_active_steps
                    ):
                        torch.cuda.memory._record_memory_history(enabled=None)

                    # Step profiler
                    # Note that this is called within gradient accumulation block, hence
                    # will include multiple forward / backward passes if gradient accumulation > 1
                    self._profiler.step()

                idx += 1  # NOTE: added by us

            self.epochs_run += 1
            self.save_checkpoint(epoch=curr_epoch)
            # Add after each epoch completes
            if self._is_rank_zero and self.profiler_profile_memory:
                torch.cuda.memory._dump_snapshot(
                    f"memory_snapshot_epoch_{curr_epoch}.pickle"
                )
                torch.cuda.memory._record_memory_history(enabled=None)

        self._profiler.stop()

    def cleanup(self) -> None:
        if self._is_rank_zero:
            self._metric_logger.close()
        destroy_process_group()

    def train_quiet_p(self) -> None:
        """
        The core training loop.
        """
        # clean up before training begins
        training.cleanup_before_training()

        world_size, rank = training.get_world_size_and_rank()

        # zero out the gradients before starting training
        if not self._optimizer_in_bwd:
            self._optimizer.zero_grad()
        else:
            for opt in self._optim_ckpt_wrapper.optim_map.values():
                opt.zero_grad()

        # Initialize tokens count and running loss (for grad accumulation)
        t0 = time.perf_counter()
        running_loss = 0
        num_tokens = 0

        # NOTE: added by us - sample just once at the beginning of the epoch loop
        self._sampler.set_epoch(0)

        self._profiler.start()
        # self.epochs_run should be non-zero when we're resuming from a checkpoint
        for curr_epoch in range(self.epochs_run, self.total_epochs):
            # Update the sampler to ensure data is correctly shuffled across epochs
            # in case shuffle is True
            # NOTE: removing it from here and putting it before the epoch loop
            # because our epochs are not the same as the dataloader epochs
            for _sampler_validation in self._sampler_validation_list:
                _sampler_validation.set_epoch(curr_epoch)  # NOTE: added by us

            # NOTE: added by us
            # ------ Validation Step ------ #
            self._model.eval()
            self._ref_model.eval()

            with torch.no_grad():
                for i, dataloader_validation in enumerate(
                    self._dataloader_validation_list
                ):
                    for _, batch in enumerate(dataloader_validation):
                        batch.pop("goal", None)
                        batch.pop("privileged_found", None)

                        utils.batch_to_device(batch, self._device)
                        val_loss = torch.tensor(0.0, device=self._device)
                        if self._is_rank_zero:
                            self._metric_logger.log_dict(
                                {f"val_loss_{i}": val_loss.item()},
                                step=self.global_step,
                            )
            del val_loss
            # ------ Reward and Advantage Computation ------ #
            (
                all_rewards,
                all_advantages,
                p_y_g_zx_mean,
                q_y_g_xz_mean,
                q_z_g_xy_mean,
                rewards_by_goal,
                kls,
                p_y_g_zx_all,
            ) = self.compute_rewards(action_log_ps_as_reward=True)

            # TODO: This is hacky need a better way to do this but this will work for now
            # I think the issue is that the advantages are
            # Use advantages if apply_advantage_in_tune is True, otherwise use rewards
            if self.apply_advantage_in_tune:
                all_advantages = all_advantages  # Use computed advantages
            else:
                all_advantages = all_rewards  # Use rewards directly
            # Log KL divergence statistics
            if self._is_rank_zero:
                kl_mean = np.mean(kls)
                kl_std = np.std(kls)

                log.info(
                    f"KL Divergence Stats - Mean: {kl_mean:.4f}, Std: {kl_std:.4f}"
                )

                self._metric_logger.log_dict(
                    {
                        "kl_divergence_mean": kl_mean,
                        "kl_divergence_std": kl_std,
                        "reward_mean": np.mean(all_rewards),
                        "reward_std": np.std(all_rewards),
                    },
                    step=self.global_step,
                )

            # ------ Training Epoch ------ #
            # Initialize tokens count and running loss (for grad accumulation)
            t0 = time.perf_counter()
            running_loss = 0
            num_tokens = 0
            real_num_tokens = 0
            max_len_samples = 0
            # Update entropy tracking variables to include sum and mean metrics
            running_per_token_ent_sum = 0
            running_full_token_ent_sum = 0
            running_per_token_ent_mean = 0
            running_full_token_ent_mean = 0
            self._model.train()  # NOTE: added by us

            pbar = tqdm(
                total=self._steps_per_epoch, disable=not (rank == 0), desc="Training"
            )

            # NOTE: added by us - counter to account for samples that are too long
            idx = 0
            processed_samples = 0
            n_samples = len(self._dataloader)
            n_gpus = torch.distributed.get_world_size()
            number_leftover_samples = (
                n_samples * n_gpus
            ) % self._gradient_accumulation_steps
            for j, batch in enumerate(self._dataloader):
                if ((idx // self._gradient_accumulation_steps)) >= (
                    self._steps_per_epoch
                ) and not self.max_bsize:
                    break
                if j != processed_samples:
                    log.warning(
                        f"Skipping batch {j} as it does not match processed_samples {processed_samples}"
                    )

                train_batch = batch["with_privilege"]
                if self._skip_max_seq_len_samples(
                    train_batch
                ) or self._skip_max_seq_len_samples(batch["without_privilege"]):
                    max_len_samples += 1
                    continue

                # Start tracking CUDA memory for active steps for just the first epoch
                if (
                    self._is_rank_zero
                    and curr_epoch == 0
                    and self.profiler_profile_memory
                    and idx == self.profiler_wait_steps + self.profiler_warmup_steps
                ):
                    torch.cuda.memory._record_memory_history()

                batch.pop("goal", None)
                batch.pop("privileged_found", None)
                utils.batch_to_device(batch, self._device)

                # Calculate the number of unmasked tokens in the current batch
                # and increment the total number of tokens seen in the step
                current_num_tokens = (
                    train_batch["labels"] != self._loss_fn.ignore_index
                ).sum()
                num_tokens += current_num_tokens
                # NOTE: added by us
                # let's monitor the total number of tokens
                real_num_tokens = train_batch["labels"].numel()

                # Shape [b, s], needed for the loss not the model
                labels = train_batch.pop("labels")
                action_start_pos = train_batch.pop("action_start_pos", None)
                train_batch.pop("action_end_pos", None)
                train_batch.pop("end_of_prompt", None)
                train_batch.pop("mask", None)
                train_batch.pop("attention_mask", None)
                batch_size = labels.shape[0]
                advantages = all_advantages[j]
                advantages = torch.tensor(advantages, device=self._device)

                # Build shifted labels once
                labels_shifted = torch.hstack(
                    (
                        labels[..., 1:],
                        self.ignore_labels_cache[: labels.shape[0]],
                    )
                )

                # Importance sampling: use cached per-token ref logprobs from compute_rewards
                ref_logprobs_for_loss = None

                if self.use_importance_sampling:

                    ref_cached = self.reference_logprobs_cache.get(j)
                    if ref_cached is not None:
                        n_chunks = len(ref_cached)
                        # ref_logprobs_for_loss = torch.cat(ref_cached,dim=1)[:,:action_start_pos].chunk(n_chunks, dim=1)
                        ref_logprobs_for_loss = ref_cached
                processed_samples += batch_size

                with self.activations_handling_ctx:
                    logits = self._model(**train_batch)

                    # logits = [logit / self.sampling_temperature for logit in logits]
                # Use shifted labels for loss
                labels = labels_shifted
                # Ensure logits are chunked to match loss expectations                if not isinstance(logits, list):                if not isinstance(logits, list):
                if not isinstance(logits, list):
                    labels = labels.reshape(-1)
                    logits = logits.reshape(-1, logits.size(-1))

                # index for labels of the cot only

                # logits = index_chunkced(logits,  action_start_pos)
                # labels = labels[:, :action_start_pos]

                combined_loss = self._loss_fn(
                    logits=logits,
                    labels=labels,
                    reward=advantages,
                    ref_logprobs=ref_logprobs_for_loss,
                    # Provide PPO bounds (low<1, high>1) to enable masking inside the loss
                    epsilon_low=self.epsilon_low_neg,
                    epsilon_high=self.epsilon_high_pos,
                )
                running_loss += combined_loss.detach() * batch_size

                # For optimizer in backward, we need to normalize before calling backward
                # This case and gradient accumulation are mutually exclusive
                if self._optimizer_in_bwd:
                    torch.distributed.all_reduce(num_tokens)
                    torch.distributed.all_reduce(running_loss)
                    combined_loss = combined_loss / num_tokens
                combined_loss.backward()
                del combined_loss
                # Step with optimizer
                if (idx + 1) % self._gradient_accumulation_steps == 0 or (
                    (idx + 1) == n_samples
                ):

                    if not self._optimizer_in_bwd:
                        # Get total number of tokens across all ranks to normalize gradients
                        torch.distributed.all_reduce(num_tokens)
                        # This will ensure that the logged loss matches what we're optimizing
                        torch.distributed.all_reduce(running_loss)
                        # All-reduce all entropy metrics

                        # Manually scale the gradients from unnormalized loss by total # of tokens
                        training.scale_grads(self._model, 1 / num_tokens)
                        # scale grads by max_batchsize and real_batchsize
                        if self.max_bsize and (idx + 1) == n_samples:
                            if number_leftover_samples == 1:
                                number_leftover_samples = n_samples
                            # scaler = torch.tensor(
                            #     number_leftover_samples / self.max_bsize
                            #     if number_leftover_samples > 0
                            #     else n_samples / self.max_bsize
                            # )
                            scaler = torch.tensor(1, device=self._device)

                            training.scale_grads(
                                self._model,
                                scaler,
                            )
                            log.info(
                                f"Scaling gradients by {scaler} Original bsize = {number_leftover_samples}"
                            )

                        # Calculate gradient norms before clipping (efficient way)
                        total_norm = torch.nn.utils.clip_grad_norm_(
                            self._model.parameters(), max_norm=float("inf")
                        )

                        grad_norm_stats = {"grad_norm_total": total_norm.item()}

                        if self._clip_grad_norm is not None:
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                self._model.parameters(),
                                max_norm=float(self._clip_grad_norm),
                            )
                        self._optimizer.step()
                        log.info(f"optimizer step")
                        self._optimizer.zero_grad(set_to_none=True)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                            gc.collect()
                            torch.cuda.synchronize()

                    # Update the number of steps when the weights are updated
                    self.global_step += 1

                    # Step the learning rate scheduler
                    if self._lr_scheduler is not None:
                        self._lr_scheduler.step()

                    loss_to_log = running_loss.item() / num_tokens
                    pbar.update(1)
                    pbar.set_description(
                        f"{curr_epoch + 1}|{self.global_step}|Loss: {loss_to_log}"
                    )
                    n_samples
                    # Log per-step metrics

                    if self._is_rank_zero:
                        time_per_step = time.perf_counter() - t0
                        log_dict = {
                            "loss": loss_to_log.cpu().item(),
                            "lr": get_lr(
                                (
                                    self._optimizer
                                    if not self._optimizer_in_bwd
                                    else self._optim_ckpt_wrapper
                                ),
                            ),
                            "tokens_per_second_per_gpu": real_num_tokens  # NOTE: added by us
                            / (time_per_step * world_size),
                            "q_y_g_xz_mean": q_y_g_xz_mean,
                            "q_z_g_xy_mean": q_z_g_xy_mean,
                            "p_y_g_zx_mean": p_y_g_zx_mean,
                        }
                        # Add gradient norm stats to logging
                        log_dict.update(grad_norm_stats)
                        if self._log_peak_memory_stats:
                            log_dict.update(
                                training.get_memory_stats(device=self._device)
                            )
                        if self._clip_grad_norm is not None:
                            log_dict.update({"grad_norm": grad_norm})
                        self._metric_logger.log_dict(
                            log_dict,
                            step=self.global_step,
                        )

                    # Reset running stats for the next step
                    running_loss = 0
                    combined_loss = 0
                    num_tokens = 0
                    real_num_tokens = 0
                    running_per_token_ent_sum = 0
                    running_full_token_ent_sum = 0
                    running_per_token_ent_mean = 0
                    running_full_token_ent_mean = 0
                    t0 = time.perf_counter()

                    # Stop tracking CUDA memory now that active steps are complete
                    if (
                        self._is_rank_zero
                        and curr_epoch == 0
                        and self.profiler_profile_memory
                        and idx
                        == self.profiler_wait_steps
                        + self.profiler_warmup_steps
                        + self.profiler_active_steps
                    ):
                        torch.cuda.memory._record_memory_history(enabled=None)

                    # Step profiler
                    # Note that this is called within gradient accumulation block, hence
                    # will include multiple forward / backward passes if gradient accumulation > 1
                    self._profiler.step()

                idx += 1  # NOTE: added by us

            self.epochs_run += 1
            self.save_checkpoint(epoch=curr_epoch)
            # Add after each epoch completes
            if self._is_rank_zero and self.profiler_profile_memory:
                torch.cuda.memory._dump_snapshot(
                    f"memory_snapshot_epoch_{curr_epoch}.pickle"
                )
                torch.cuda.memory._record_memory_history(enabled=None)

        self._profiler.stop()


def rescale_rewards(all_rewards):
    """
    Min-max scale a list of rewards to the range [-1, 1].
    - Empty list -> []
    - All rewards identical -> all zeros
    """
    if not all_rewards:
        return []
    try:
        min_r = float(min(all_rewards))
        max_r = float(max(all_rewards))
    except TypeError:
        # If values are not comparable/numeric, return as-is
        return all_rewards
    if max_r == min_r:
        return [0.0 for _ in all_rewards]
    denom = max_r - min_r
    scaled = [2.0 * ((float(r) - min_r) / denom) - 1.0 for r in all_rewards]
    # Clamp for numerical safety
    return [max(-1.0, min(1.0, s)) for s in scaled]


def index_chunkced(chunked, index):
    """Concatenate chunked [B, T_i, D] tensors, slice up to index along seq dim, then re-chunk.

    Args:
        chunked: List of tensors shaped [B, T_i, D] with consistent B and D.
        index:   Integer sequence cut-off; keep tokens in [0, index) along seq dim.

    Returns:
        List[Tensor]: Re-chunked tensors up to the provided index, preserving original
        chunk sizes where possible (last chunk may be truncated). If index <= 0, returns [].
    """
    # Concatenate along sequence dimension [B, S_total, D]
    num_chunks = len(chunked)
    unchunked = torch.cat(chunked, dim=1)
    unchunked = unchunked[..., :index, :]
    rechunked = unchunked.chunk(num_chunks, dim=1)

    return rechunked


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """
    Entry point for the recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in config (see available configs through ``tune ls``)
        - Overwritten by arguments from the command-line
    """
    if not training.is_distributed():
        raise RuntimeError(
            "Distributed finetune recipe should be run via a distributed launcher."
            "If using tune CLI, please specify --nnodes 1 and --nproc_per_node [num_gpus]"
        )
    init_process_group(backend="gloo" if cfg.device == "cpu" else "nccl")
    if cfg.get("fsdp_cpu_offload", False):
        # Utilize all available CPU cores for intra-op parallelism. This provides ~2x
        # speed up when benchmarking fused AdamW on CPU
        training.set_torch_num_threads()

    config.log_config(recipe_name="FullFinetuneRecipeDistributedPrivalaged", cfg=cfg)

    recipe = FullFinetuneRecipeDistributedPrivalaged(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())
