# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from torchtune import config, modules, training, utils

log = utils.get_logger("DEBUG")


class CEWithChunkedOutputLoss(torch.nn.Module):
    """
    Cross-entropy with chunked outputs that saves memory by only upcasting one chunk at a time.

    Whenever the model is trained with bf16, before running CE, we have to upcast
    it to fp32 for better accuracy and stability. When upcasting happens, the memory usage doubles.
    Models like llama3 have large vocabulary size and, therefore, have a large output
    tensor of shape ``(bsz, num_tokens, vocab_size)``. If we chunk on the token level, you can still compute
    the cross entropy normally, but upcasting only one chunk at a time saves considerable memory.

    The CE and upcasting have to be compiled together for better performance.
    When using this class, we recommend using :func:`torch.compile` only on the method ``compute_cross_entropy``.
    The gains from chunking won't be realized if you compile the entire class.

    For more details, please refer to: https://github.com/pytorch/torchtune/pull/1390
    """

    def __init__(self, num_output_chunks: int = 8, ignore_index: int = -100, **kwargs):
        super().__init__()
        self.num_output_chunks = num_output_chunks
        self.ignore_index = ignore_index

    def compute_cross_entropy(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        ratio: torch.Tensor = None,
        normalize: bool = True,
        epsilon=1e-10,
    ) -> torch.Tensor:
        """
        Upcast logits to fp32 and compute cross entropy loss.

        Args:
            logits: Input logits tensor
            labels: Ground truth labels
            ratio: Optional per-token importance weights or advantages
            normalize: Whether to normalize the loss
        """
        # Standard cross entropy loss

        loss = F.cross_entropy(
            logits.float(), labels, ignore_index=self.ignore_index, reduction="none"
        )
        self.epsilon = epsilon
        # Apply ratio if provided
        if ratio is not None:
            # Multiply element-wise by provided per-token weight
            loss = loss * ratio

        # Sum the losses
        return loss.sum()

    def _calculate_importance_ratio(
        self,
        new_log_ps: torch.Tensor,
        old_log_ps: torch.Tensor,  # This is now the pre-gathered value
        labels: torch.Tensor,
        epsilon_low: float = 0.0,
        epsilon_high: float = float("inf"),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the importance ratio for the new and reference log probabilities.

        Args:
            new_log_ps: Log probabilities from the current model
            old_log_ps: Precomputed log probabilities from the reference model
            labels: Token indices for selecting the correct log probabilities
            epsilon_low: Lower bound for importance ratio clipping
            epsilon_high: Upper bound for importance ratio clipping

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of (unclipped_ratio, clipped_ratio)
        """
        with torch.no_grad():
            # Create a mask for tokens that are the ignore index
            ignore_mask = labels == self.ignore_index

            # For valid calculation, clamp indices to be within vocab range
            vocab_size = new_log_ps.size(-1)
            valid_indices = labels.clamp(0, vocab_size - 1)

            # Use the valid indices for gathering only current model log probs
            new_selected = torch.gather(
                new_log_ps, dim=-1, index=valid_indices.unsqueeze(-1)
            ).squeeze(-1)

            # old_log_ps is already the selected value
            old_selected = old_log_ps

            # Calculate the importance ratio (unclipped)
            importance_ratio = torch.exp(new_selected - old_selected)

            # Set importance ratio to 1.0 for ignored tokens (won't affect loss)
            importance_ratio = torch.where(
                ignore_mask, torch.ones_like(importance_ratio), importance_ratio
            )

            # Get the clipped version of importance ratio
            clipped_importance_ratio = torch.clamp(
                importance_ratio, min=epsilon_low, max=epsilon_high
            )

        return importance_ratio, clipped_importance_ratio

    def _build_ratio_chunks_bs1(
        self,
        logits_chunks: List[torch.Tensor],
        labels: torch.Tensor,
        ref_logprob_chunks: List[torch.Tensor],
        reward: Optional[torch.Tensor],
        epsilon_low: float,
        epsilon_high: float,
    ) -> List[torch.Tensor]:
        """Build per-chunk importance weights with PPO-style masking for bs=1 and set last_clip_stats.

        Returns a list of flattened per-chunk weights aligned with labels/logits chunks.
        """
        batch_size = labels.shape[0]  # assumed 1
        labels_chunks_2d = [c for c in labels.chunk(self.num_output_chunks, dim=1)]

        # Build a per-token reward grid of shape [1, T]
        if (
            reward is not None
            and not isinstance(reward, (int, float))
            and reward.numel() > 1
            and reward.shape == labels.shape
        ):
            reward_grid = reward  # [1, T]
        else:
            if reward is None:
                r_val = 1.0
            elif isinstance(reward, torch.Tensor) and reward.numel() == batch_size:
                r_val = float(reward.view(-1)[0].item())
            else:
                r_val = float(reward)
            reward_grid = torch.full_like(labels.long(), fill_value=r_val, dtype=torch.float32)

        reward_chunks_2d = [
            rc for rc in reward_grid.chunk(self.num_output_chunks, dim=1)
        ]

        ratio_chunks: List[torch.Tensor] = []
        masked_tokens_total = 0
        total_tokens_total = 0
        affected_any = torch.tensor(False, dtype=torch.bool, device=labels.device)

        for i, (logits_chunk, ref_logprob_chunk, labels_chunk_flat) in enumerate(
            zip(
                logits_chunks,
                ref_logprob_chunks,
                [lc.reshape(-1) for lc in labels_chunks_2d],
            )
        ):
            # Compute current log-probs and importance ratio (no grad)
            curr_log_ps = F.log_softmax(logits_chunk.float(), dim=-1)
            unclipped_ratio, _ = self._calculate_importance_ratio(
                curr_log_ps,
                ref_logprob_chunk,
                labels_chunk_flat,
                epsilon_low,
                epsilon_high,
            )

            Tchunk = labels_chunks_2d[i].shape[1]
            unclipped_ratio_2d = unclipped_ratio.view(batch_size, Tchunk)
            valid_mask_2d = labels_chunks_2d[i] != self.ignore_index
            reward_2d = reward_chunks_2d[i]

            # PPO-style mask based on sign of reward
            is_pos = reward_2d >= 0
            mask_pos = unclipped_ratio_2d <= epsilon_high
            mask_neg = unclipped_ratio_2d >= epsilon_low
            mask_2d = torch.where(is_pos, mask_pos, mask_neg) & valid_mask_2d

            # Stats (bs=1)
            masked_tokens_total += ((~mask_2d) & valid_mask_2d).sum()
            total_tokens_total += valid_mask_2d.sum()
            affected_any |= ((~mask_2d) & valid_mask_2d).any()

            # Effective weight inside mask, zero outside
            effective_weight_2d = (unclipped_ratio_2d * reward_2d) * mask_2d.to(
                unclipped_ratio_2d.dtype
            )
            ratio_chunks.append(effective_weight_2d.reshape(-1))

        # Update simple last_clip_stats for bs=1


        return ratio_chunks

    def forward(
        self,
        logits: Union[torch.Tensor, List[torch.Tensor]],
        labels: torch.Tensor,
        ref_logprobs: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        reward: Optional[torch.Tensor] = None,
        epsilon_low: float = 0.0,
        epsilon_high: float = float("inf"),
    ) -> torch.Tensor:
        """
        Args:
            logits (Union[torch.Tensor, List[torch.Tensor]]): Either a single logits tensor or list of chunked logits,
                where each chunk has shape ``(batch_size, num_tokens / num_output_chunks, vocab_size)``.
            labels (torch.Tensor): Ground truth labels of shape ``(batch_size, num_tokens)``.
            ref_logits (Optional[Union[torch.Tensor, List[torch.Tensor]]]): Reference model logits for importance sampling.
            reward (Optional[torch.Tensor]): Reward tensor for scaling importance ratio.
            epsilon_low (float): Lower bound for importance ratio clipping.
            epsilon_high (float): Upper bound for importance ratio clipping.

        Returns:
            torch.Tensor: Cross entropy loss of shape (1,).
        """
        # Normalization factor
        total_elements = (labels != self.ignore_index).sum()

        # Chunk and reshape labels
        labels_chunks = [
            target_chunk.reshape(-1)
            for target_chunk in labels.chunk(self.num_output_chunks, dim=1)
        ]

        # Reshape logits chunks
        logits_chunks = [
            logit_chunk.reshape(-1, logit_chunk.size(-1)) for logit_chunk in logits
        ]

        # Preprocess reward
        has_per_token_reward = (
            reward is not None
            and not isinstance(reward, (int, float))
            and reward.numel() > 1
            and reward.shape == labels.shape
        )

        reward_chunks = None
        scalar_reward = None

        if has_per_token_reward:
            # Process per-token rewards
            reward_chunks = [
                r_chunk.reshape(-1)
                for r_chunk in reward.chunk(self.num_output_chunks, dim=1)
            ]
        elif reward is not None:
            # Convert to scalar reward
            scalar_reward = reward

        # Process reference logprobs for importance sampling
        ratio_chunks = None
        if ref_logprobs is not None:
            # Ensure we have a list of flattened ref logprobs chunks
            if isinstance(ref_logprobs, torch.Tensor):
                ref_chunks_2d = [
                    c for c in ref_logprobs.chunk(self.num_output_chunks, dim=1)
                ]
                ref_logprob_chunks = [c.reshape(-1) for c in ref_chunks_2d]
            else:
                ref_logprob_chunks = [r_chunk.reshape(-1) for r_chunk in ref_logprobs]

            # Build bs=1 ratio chunks with masking and stats
            ratio_chunks = self._build_ratio_chunks_bs1(
                logits_chunks=logits_chunks,
                labels=labels,
                ref_logprob_chunks=ref_logprob_chunks,
                reward=reward,
                epsilon_low=epsilon_low,
                epsilon_high=epsilon_high,
            )

        # Compute loss chunk by chunk
        total_loss = 0.0
        for i, (logits_chunk, labels_chunk) in enumerate(
            zip(logits_chunks, labels_chunks)
        ):
            if ratio_chunks is not None:
                # Case 1: Using importance sampling (with PPO-style masking)
                chunk_loss = self.compute_cross_entropy(
                    logits_chunk, labels_chunk, ratio_chunks[i]
                )
            elif reward is not None:
                # Case 2: No importance sampling but we have reward
                base_loss = self.compute_cross_entropy(logits_chunk, labels_chunk)

                if has_per_token_reward:
                    chunk_loss = base_loss * reward_chunks[i].mean()
                else:
                    chunk_loss = base_loss * scalar_reward
            else:
                # Case 3: Standard cross-entropy
                chunk_loss = self.compute_cross_entropy(logits_chunk, labels_chunk)

            total_loss += chunk_loss

        # Normalize the loss
        return total_loss / total_elements

    # Helper to retrieve and optionally reset accumulated clip stats
    def get_and_reset_clip_stats(self, reset: bool = True) -> Optional[dict]:
        stats = getattr(self, "_clip_stats_total", None)
        if reset and stats is not None:
            self._clip_stats_total = None
        return stats

    def compute_entropy(
        self, logits: List[torch.Tensor], labels: torch.Tensor, ent_weight: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Memory-optimized entropy calculation."""
        logits_chunks = [
            logit_chunk.reshape(-1, logit_chunk.size(-1)) for logit_chunk in logits
        ]
        labels_chunks = [
            label_chunk.reshape(-1)
            for label_chunk in labels.chunk(self.num_output_chunks, dim=1)
        ]

        # Initialize empty tensors for more efficient operation
        full_token_entropy_mean = 0
        per_token_entropy_sum = 0
        full_token_entropy_sum = 0
        per_token_entropy_mean = 0
        chunk_count = 0
        valid_token_count = 0  # Track valid tokens for proper normalization

        # Process chunks one by one
        for i, (logits_chunk, labels_chunk) in enumerate(
            zip(logits_chunks, labels_chunks)
        ):
            # Create mask for valid tokens (not ignore_index)
            # valid_mask = torch.ones_like(labels_chunk, dtype=torch.bool)
            valid_mask = labels_chunk != self.ignore_index
            valid_tokens_in_chunk = valid_mask.sum()

            # Skip chunk if no valid tokens
            if valid_tokens_in_chunk == 0:
                continue

            chunk_count += 1
            valid_token_count += valid_tokens_in_chunk

            # Calculate entropy components
            log_probs = F.log_softmax(logits_chunk, dim=-1)
            vocab_size = log_probs.size(-1)
            valid_indices = labels_chunk.clamp(0, vocab_size - 1)
            gathered_log_probs = torch.gather(log_probs, dim=-1, index=valid_indices.unsqueeze(-1)).squeeze(-1)

            gathered_probs = gathered_log_probs.exp() + self.epsilon

            # Per-token metrics (always detached) - apply mask
            with torch.no_grad():
                per_token_ent = -gathered_probs * gathered_log_probs
                # Apply mask and sum only valid tokens
                per_token_ent_masked = per_token_ent * valid_mask
                per_token_entropy_sum += per_token_ent_masked.sum().detach()
                # Mean only over valid tokens in this chunk
                per_token_entropy_mean += (
                    per_token_ent_masked.sum() / valid_tokens_in_chunk
                ).detach()

            # Full token entropy calculation
            if ent_weight > 0:
                ungathered_probs = log_probs.exp() + self.epsilon
                full_token_ent = -ungathered_probs * log_probs
                # Apply mask across vocab dimension by expanding valid_mask
                valid_mask_expanded = valid_mask.unsqueeze(-1)  # Shape: [seq_len, 1]
                full_token_ent_masked = full_token_ent * valid_mask_expanded
                # Only keep gradients for mean computation which is used in loss
                full_token_entropy_mean += (
                    full_token_ent_masked.sum() / valid_tokens_in_chunk
                ).detach() / len(logits_chunks)
            else:
                with torch.no_grad():
                    ungathered_probs = log_probs.exp() + self.epsilon
                    full_token_ent = -ungathered_probs * log_probs
                    # Apply mask
                    valid_mask_expanded = valid_mask.unsqueeze(-1)
                    full_token_ent_masked = full_token_ent * valid_mask_expanded
                    full_token_entropy_mean += (
                        full_token_ent_masked.sum() / valid_tokens_in_chunk
                    ) / len(logits_chunks)

            with torch.no_grad():
                full_token_entropy_sum += full_token_ent_masked.sum().detach()

            # Clean up this chunk's intermediate tensors
            del (
                log_probs,
                valid_indices,
                gathered_log_probs,
                gathered_probs,
                ungathered_probs,
            )
            del (
                full_token_ent,
                per_token_ent,
                full_token_ent_masked,
                per_token_ent_masked,
            )

        # Normalize means over all valid tokens across chunks
        if chunk_count > 0:
            per_token_entropy_mean = per_token_entropy_mean / chunk_count

        return (
            per_token_entropy_sum,
            full_token_entropy_sum,
            per_token_entropy_mean,
            full_token_entropy_mean,
        )
