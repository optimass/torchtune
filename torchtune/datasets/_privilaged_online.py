import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Union
import copy
import numpy as np
import datasets
import torch

datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory=".": True

from datasets import load_dataset
from torch.utils.data import Dataset
from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX
from torchtune.data._messages import Message, validate_messages
from torchtune.modules.transforms import Transform

from torchtune.data import InputOutputToMessages
from torchtune.datasets._packed import PackedDataset

from torchtune.modules.tokenizers import ModelTokenizer


class priv_dataloader_online(Dataset):

    def __init__(
        self,
        *,
        source: str,
        message_transform: Transform,
        model_transform: Transform,
        filter_fn: Optional[Callable] = None,
        **load_dataset_kwargs: Dict[str, Any],
    ) -> None:
        self._message_transform = message_transform
        self._model_transform = model_transform
        self._data = load_dataset(source, **load_dataset_kwargs)
        if filter_fn is not None:
            self._data = self._data.filter(filter_fn)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self._data[index]
        return self._prepare_sample(sample)

    def _maybe_replace_action(self, output: str, sample: Mapping[str, Any]) -> str:
        """
        If match_reward is -1 and there is an <action>...</action> block, replace the last block
        with the expected action. If there is NO <action> block:
          - If a <think>...</think> segment exists, keep everything through its end and append the action.
          - Otherwise, replace the entire output with just the action block.
        """
        match_reward = sample["match_reward"]
        expected_action = sample["expected_action"]
        if match_reward != -1:
            return output

        action_matches = list(
            re.finditer(r"<action>.*?</action>", output, flags=re.DOTALL)
        )
        if not action_matches:
            think_block = None
            for m in re.finditer(r"<think>.*?</think>", output, flags=re.DOTALL):
                think_block = m
            if think_block is not None:
                kept = output[: think_block.end()]
                return f"{kept}{expected_action}"
            # No think and no action: fully replace
            return f"{expected_action}"

        # Existing action blocks present: replace the last one
        start, end = action_matches[-1].span()
        replacement = f"{expected_action}"
        return output[:start] + replacement + output[end:]

    def _extract_parts_privileged_prompt(self, prompt: str):
        """
        Extract privileged information from prompt and return both versions.

        Args:
            prompt: The original prompt that may contain secret information

        Returns:
            tuple: (prompt_with_secret, prompt_without_secret, privileged_found)
        """
        secret_pattern = r"<Secret information>.*?</Secret information>"
        privileged_found = (
            1 if re.search(secret_pattern, prompt, flags=re.DOTALL) else 0
        )
        prompt_no_secret = re.sub(secret_pattern, "", prompt, flags=re.DOTALL).strip()

        return (
            prompt,
            prompt_no_secret,
            privileged_found,
        )

    def _parse_assistant_content(self, content: str) -> List[tuple]:
        """
        Parse assistant message content into ordered segments.

        Args:
            content: The text content from an assistant message

        Returns:
            List of (segment_type, segment_text) tuples where segment_type is:
            - "think": <think>...</think> block
            - "action": <action>...</action> block
            - "other": everything else
        """
        think_pattern = r"<think>.*?</think>"
        action_pattern = r"<action>.*?</action>"

        # Find all matches with their positions
        think_matches = [
            (m.start(), m.end(), "think", m.group(0))
            for m in re.finditer(think_pattern, content, re.DOTALL)
        ]
        action_matches = [
            (m.start(), m.end(), "action", m.group(0))
            for m in re.finditer(action_pattern, content, re.DOTALL)
        ]

        # Combine and sort by position
        all_matches = sorted(think_matches + action_matches, key=lambda x: x[0])

        if not all_matches:
            # No think/action blocks, return entire content as "other"
            return [("other", content)] if content else []

        segments = []
        last_end = 0

        for start, end, match_type, matched_text in all_matches:
            # Add text before this match
            if start > last_end:
                before_text = content[last_end:start]
                if before_text:
                    segments.append(("other", before_text))

            # Add the matched segment
            segments.append((match_type, matched_text))
            last_end = end

        # Add remaining text after last match
        if last_end < len(content):
            remaining = content[last_end:]
            if remaining:
                segments.append(("other", remaining))

        return segments

    def _tokenize_messages_with_positions(
        self,
        messages: List[Message],
        add_eos: bool = True,
    ) -> Dict[str, Any]:
        """
        Tokenize messages while tracking think and action token positions.
        This mimics the tokenizer's tokenize_messages but adds position tracking.

        Returns:
            Dict containing:
                - tokens: List[int]
                - mask: List[bool]
                - think_positions: List[Tuple[int, int]]
                - action_positions: List[Tuple[int, int]]
        """
        tokenizer = self._model_transform

        # Apply template if present
        if (
            hasattr(tokenizer, "prompt_template")
            and tokenizer.prompt_template is not None
        ):
            templated_messages = tokenizer.prompt_template(messages)
        else:
            templated_messages = messages

        tokenized_messages = []
        mask = []
        think_positions = []
        action_positions = []

        for i, message in enumerate(templated_messages):
            # Tokenize header (always masked)
            if hasattr(tokenizer, "_tokenize_header"):
                header_tokens = tokenizer._tokenize_header(templated_messages, i)
            else:
                # Fallback for other tokenizers
                header_tokens = self._tokenize_header_fallback(message)

            tokenized_messages.extend(header_tokens)
            mask.extend([True] * len(header_tokens))

            # Process message content
            for item in message.content:
                if item["type"] == "text":
                    content_text = item["content"]

                    # If this is an assistant message, parse for think/action segments
                    if message.role == "assistant":
                        segments = self._parse_assistant_content(content_text)

                        for seg_type, seg_text in segments:
                            start_pos = len(tokenized_messages)

                            # Encode segment
                            seg_tokens = tokenizer.encode(
                                seg_text,
                                add_bos=False,
                                add_eos=False,
                            )

                            tokenized_messages.extend(seg_tokens)
                            mask.extend([message.masked] * len(seg_tokens))

                            end_pos = len(tokenized_messages)

                            # Track positions for think/action (excluding tag tokens)
                            if seg_type == "think":
                                # Calculate tag token lengths
                                opening_tag_tokens = len(
                                    tokenizer.encode(
                                        "<think>", add_bos=False, add_eos=False
                                    )
                                )
                                closing_tag_tokens = len(
                                    tokenizer.encode(
                                        "\n\n/think>", add_bos=False, add_eos=False
                                    )
                                )
                                # Adjust positions to exclude tags
                                content_start = start_pos + opening_tag_tokens
                                content_end = end_pos - closing_tag_tokens
                                think_positions.append((content_start, content_end))
                            elif seg_type == "action":
                                # Calculate tag token lengths
                                opening_tag_tokens = len(
                                    tokenizer.encode(
                                        "<action>", add_bos=False, add_eos=False
                                    )
                                )
                                closing_tag_tokens = len(
                                    tokenizer.encode(
                                        "</action>", add_bos=False, add_eos=False
                                    )
                                )
                                # Adjust positions to exclude tags
                                content_start = start_pos + opening_tag_tokens
                                content_end = end_pos - closing_tag_tokens
                                action_positions.append((content_start, content_end))
                    else:
                        # For non-assistant messages, encode normally
                        content_tokens = tokenizer.encode(
                            content_text,
                            add_bos=False,
                            add_eos=False,
                        )
                        tokenized_messages.extend(content_tokens)
                        mask.extend([message.masked] * len(content_tokens))
                else:
                    raise RuntimeError(
                        f"Unsupported message content type: {item['type']}"
                    )

            # Tokenize footer (always masked)
            if hasattr(tokenizer, "_tokenize_footer"):
                footer_tokens = tokenizer._tokenize_footer(templated_messages, i)
            else:
                # Fallback for other tokenizers
                footer_tokens = self._tokenize_footer_fallback(
                    message, i, len(templated_messages)
                )

            tokenized_messages.extend(footer_tokens)
            mask.extend([True] * len(footer_tokens))

            # Break early if max_seq_len reached
            if hasattr(tokenizer, "max_seq_len") and tokenizer.max_seq_len:
                if len(tokenized_messages) >= tokenizer.max_seq_len:
                    break

        # Add EOS token
        if add_eos and hasattr(tokenizer, "eos_id"):
            tokenized_messages.append(tokenizer.eos_id)
            mask.append(mask[-1] if mask else True)

        # Truncate if necessary
        if hasattr(tokenizer, "max_seq_len") and tokenizer.max_seq_len:
            from torchtune.data import truncate

            eos_id = (
                tokenizer.eos_id if add_eos and hasattr(tokenizer, "eos_id") else None
            )
            tokenized_messages = truncate(
                tokenized_messages, tokenizer.max_seq_len, eos_id
            )
            mask = truncate(mask, tokenizer.max_seq_len, True if add_eos else None)

        return {
            "tokens": tokenized_messages,
            "mask": mask,
            "think_positions": think_positions,
            "action_positions": action_positions,
        }

    def _tokenize_header_fallback(self, message: Message) -> List[int]:
        """Fallback header tokenization for tokenizers without _tokenize_header."""
        tokenizer = self._model_transform
        tokens = []

        # Try to use im_start_id if available (Qwen-style)
        if hasattr(tokenizer, "im_start_id"):
            tokens.append(tokenizer.im_start_id)

        tokens.extend(
            tokenizer.encode(
                f"{message.role}\n",
                add_bos=False,
                add_eos=False,
            )
        )
        return tokens

    def _tokenize_footer_fallback(
        self, message: Message, idx: int, total: int
    ) -> List[int]:
        """Fallback footer tokenization for tokenizers without _tokenize_footer."""
        tokenizer = self._model_transform
        tokens = []

        # Add end token if available
        if hasattr(tokenizer, "im_end_id"):
            tokens.append(tokenizer.im_end_id)

        # Add newline for non-final assistant messages
        if message.role != "assistant" or idx != total - 1:
            tokens.extend(tokenizer.encode("\n", add_bos=False, add_eos=False))

        return tokens

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Dict[str, Any]:

        sample_without_privileged = copy.deepcopy(sample)
        (
            prompt_with_privileged,
            prompt_without_privileged,
            privileged_found,
        ) = self._extract_parts_privileged_prompt(sample["traj"][0]["content"])
        sample_without_privileged["traj"][0]["content"] = prompt_without_privileged

        transformed_sample_with_privilege = self._message_transform(sample)
        transformed_sample_without_privileged = self._message_transform(
            sample_without_privileged
        )

        # Check if we have messages (for the new tokenization path)
        if "messages" in transformed_sample_with_privilege:
            # Use custom tokenization that tracks think/action positions
            tokenized_dict_with_privilege = self._tokenize_messages_with_positions(
                transformed_sample_with_privilege["messages"],
                add_eos=True,
            )
            tokenized_dict_without_privilege = self._tokenize_messages_with_positions(
                transformed_sample_without_privileged["messages"],
                add_eos=True,
            )
        else:
            # Fallback to old tokenization (for backward compatibility)
            tokenized_dict_with_privilege = self._model_transform(
                transformed_sample_with_privilege
            )
            tokenized_dict_without_privilege = self._model_transform(
                transformed_sample_without_privileged
            )
            # Add empty position lists for backward compatibility
            tokenized_dict_with_privilege["think_positions"] = []
            tokenized_dict_with_privilege["action_positions"] = []
            tokenized_dict_without_privilege["think_positions"] = []
            tokenized_dict_without_privilege["action_positions"] = []

        if not (
            "tokens" in tokenized_dict_with_privilege
            and "mask" in tokenized_dict_with_privilege
        ):
            keys_str = ", ".join(tokenized_dict_with_privilege.keys())
            error_message = (
                "model_transform returned the following keys: "
                f"{keys_str}. Must return 'tokens' and 'mask' as keys."
            )
            raise ValueError(error_message)

        # Wherever mask == True, set to CROSS_ENTROPY_IGNORE_IDX. Otherwise keep as tokens
        tokenized_dict_with_privilege["labels"] = list(
            np.where(
                tokenized_dict_with_privilege["mask"],
                CROSS_ENTROPY_IGNORE_IDX,
                tokenized_dict_with_privilege["tokens"],
            )
        )
        assert len(tokenized_dict_with_privilege["tokens"]) == len(
            tokenized_dict_with_privilege["labels"]
        )
        tokenized_dict_without_privilege["labels"] = list(
            np.where(
                tokenized_dict_without_privilege["mask"],
                CROSS_ENTROPY_IGNORE_IDX,
                tokenized_dict_without_privilege["tokens"],
            )
        )
        assert len(tokenized_dict_without_privilege["tokens"]) == len(
            tokenized_dict_without_privilege["labels"]
        )

        return_dict = {
            "with_privilege": tokenized_dict_with_privilege,
            "without_privilege": tokenized_dict_without_privilege,
            "privileged_found": privileged_found,
            "reward": sample["reward"],
            "og_reward": sample["og_reward"],
            "goal": sample["instruction"],
        }

        return return_dict

    def tests(self, with_privilege, without_privilege):
        """
        Ensure that the number of tokens in the action/think region differs
        between the privileged and non-privileged processed scenarios.

        Raises AssertionError if the action lengths are identical (which would
        indicate the privileged content did not change the generated region size).
        """
        a_start = with_privilege["action_start_pos"]
        a_end = with_privilege["action_end_pos"]
        b_start = without_privilege["action_start_pos"]
        b_end = without_privilege["action_end_pos"]

        a_end_prompt = with_privilege["end_of_prompt"]
        b_end_prompt = without_privilege["end_of_prompt"]
        a_len = a_end - a_start
        b_len = b_end - b_start

        a_think_len = a_start - a_end_prompt
        b_think_len = b_start - b_end_prompt

        # We expect the privileged and non-privileged action/think lengths to differ.
        assert (
            a_len == b_len
        ), f"action/think lengths should not differ: {a_len} == {b_len}"
        assert (
            a_think_len == b_think_len
        ), f"think lengths should not differ: {a_think_len} == {b_think_len}"
        return True

    def _encode_with_role(
        self,
        content: str,
        role: str,
        add_bos: bool = False,
        add_eos: bool = False,
        eot_for_message: bool = True,
    ) -> List[int]:
        """
        Helper to encode text with role-specific tokens by creating a temporary
        Message object and using the model's tokenizer.

        Args:
            content (str): The text content to encode.
            role (str): The role of the message ('user' or 'assistant').
            add_bos (bool): Whether to add the beginning-of-sequence token.
            add_eos (bool): Whether to add the end-of-sequence token.
            eot_token_for_message (bool): Whether to consider this message as the
                end of a turn, which influences the addition of role-specific
                end-of-turn tokens (like <|eot_id|>).
        """
        if not hasattr(self._model_transform, "tokenize_messages"):
            # Fallback to simple encoding if the tokenizer doesn't support messages
            return self._model_transform.encode(
                content, add_bos=add_bos, add_eos=add_eos
            )

        # Create a temporary message to get the role-specific tokens
        temp_message = Message(role=role, content=content, eot=eot_for_message)

        # Use the tokenizer's message processing logic
        # Note: tokenize_messages will strip the BOS token from the start of the message
        # if add_bos is False, which is the behavior we want for partial sequences.
        # It returns a list of lists, so we take the first element.
        return self._model_transform.tokenize_messages(
            [temp_message],
        )[0]

    # Public controller -------------------------------------------------
    def _process_scenario(
        self, prompt, output: str, before_action: str, remaining_prompt=None
    ) -> Dict[str, Any]:
        """Dispatch to single or multi message processing.

        If ``prompt`` is a list of prior messages (each a mapping with keys
        ``role`` and ``content``) we treat them as immutable history: all
        their tokens are assigned ignore_index (masked from loss). Otherwise
        we fall back to the original single-string behavior.
        """
        if remaining_prompt is not None:  # multi-turn history
            return self._process_scenario_multi(
                prompt, output, before_action, remaining_prompt
            )
        return self._process_scenario_single(prompt, output, before_action)

    # Single (original) -------------------------------------------------
    def _process_scenario_single(
        self, prompt: str, output: str, before_action: str
    ) -> Dict[str, Any]:
        """Original single-prompt implementation (unchanged logic)."""
        # Extract the action text from the output
        action_blocks = list(
            re.finditer(r"<action>.*?</action>", output, flags=re.DOTALL)
        )
        action_text = action_blocks[-1].group(0) if action_blocks else ""
        after_action = (
            output[output.find(action_text) + len(action_text) :] if action_text else ""
        )

        # Step 1: Encode the prompt with user role (includes BOS and role headers)
        prompt_tokens = self._encode_with_role(
            prompt, role="user", add_bos=True, add_eos=False, eot_for_message=True
        )[:-1]
        # Step 2: Encode the "before action" part with assistant role but NO EOS
        before_action_tokens = []
        if before_action:
            before_action_tokens = self._encode_with_role(
                before_action,
                role="assistant",
                add_bos=False,
                add_eos=False,
                eot_for_message=False,
            )[:-2]

        # Step 3: Encode the action with NO role headers but WITH EOS if it's the end
        action_tokens = []
        if action_text:
            # Simple encode without role processing since it's a continuation
            action_tokens = self._model_transform.encode(
                action_text, add_bos=False, add_eos=False
            )

        # Step 4: Encode anything after the action (if any) with EOS
        after_action_tokens = []
        if after_action:
            # Simple encode for any text after action
            after_action_tokens = self._model_transform.encode(
                after_action, add_bos=False, add_eos=True
            )
        # elif action_tokens:
        #     # If action is the last part, add EOS to the sequence
        #     after_action_tokens = (
        #         [self._model_transform.eos_id]
        #         if hasattr(self._model_transform, "eos_id")
        #         else []
        #     )
        end_of_sentence_tokens = [128009, 128001]
        # Step 5: Concatenate all parts to build the full sequence
        tokens = (
            prompt_tokens
            + before_action_tokens
            + action_tokens
            + end_of_sentence_tokens
        )

        # Step 6: Calculate positions directly since we built it piece by piece
        end_of_prompt = len(prompt_tokens)
        action_start_pos = len(prompt_tokens) + len(before_action_tokens)
        action_end_pos = action_start_pos + len(action_tokens)

        # Step 7: Create mask - prompt is not masked (True), output is masked (False)
        # This assumes train_on_input=False behavior
        mask = [True] * len(prompt_tokens) + [False] * (
            len(tokens) - len(prompt_tokens)
        )

        # Step 8: Create labels following SFTDataset pattern
        labels = list(
            np.where(
                mask,
                CROSS_ENTROPY_IGNORE_IDX,
                tokens,
            )
        )

        return {
            "tokens": tokens,
            "labels": labels,
            "mask": mask,
            "action_start_pos": action_start_pos,
            "action_end_pos": action_end_pos,
            "end_of_prompt": end_of_prompt,
        }

    # Multi (chat history) ----------------------------------------------
    def _process_scenario_multi(
        self,
        prompt: List[Mapping[str, Any]],
        output: str,
        before_action: str,
        remaining_prompt: Optional[List[Mapping[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Process a scenario where ``messages`` is a prior dialogue history.

        Design choices:
        - Every prior message token is masked from loss (labels set to ignore idx).
        - We re-use ``_encode_with_role`` for role-aware tokenization.
        - The final generated portion corresponds to ``before_action`` + ``<action>`` block
          (and optional terminators) exactly like the single variant.
        - Each message is encoded similarly to the single prompt (we strip the trailing
          end-of-turn token by slicing ``[:-1]`` to mirror the single path behaviour).
        """
        # 1. Tokenize history
        messages = [{"role": "system", "content": prompt}] + remaining_prompt
        history_tokens: List[int] = []
        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            # Add BOS only for very first message to keep sequence consistent
            encoded = self._encode_with_role(
                content,
                role=role,
                add_bos=True,
                add_eos=False,
                eot_for_message=True,
            )
            # # Mirror single behaviour: drop last token (assumed eot / delimiter)
            # if encoded:
            #     encoded = encoded[:-1]
            history_tokens.extend(encoded)

        # 2. Extract action as everything after the last </think>
        think_matches = list(
            re.finditer(r"<think>.*?</think>", output, flags=re.DOTALL)
        )
        if think_matches:
            action_text = output[think_matches[-1].end() :]
        else:
            action_text = output
        after_action = ""  # always empty by design

        # 3. Encode before_action (assistant continuation, no BOS/EOS, strip trailing delim similar to single)
        before_action_tokens: List[int] = []
        if before_action:
            before_action_tokens = self._encode_with_role(
                before_action,
                role="assistant",
                add_bos=False,
                add_eos=False,
                eot_for_message=False,
            )
            # Strip the last two tokens as done in single path ([:-2]) if long enough
            if len(before_action_tokens) >= 2:
                before_action_tokens = before_action_tokens[:-2]

        # 4. Encode action and optional after_action raw
        action_tokens: List[int] = []
        if action_text:
            action_tokens = self._model_transform.encode(
                action_text, add_bos=False, add_eos=False
            )
        # We ignore after_action_tokens for parity with single (they are not appended)

        end_of_sentence_tokens = [128009, 128001]

        # 5. Concatenate
        tokens = (
            history_tokens
            + before_action_tokens
            + action_tokens
            + end_of_sentence_tokens
        )

        end_of_prompt = len(history_tokens)
        action_start_pos = len(history_tokens) + len(before_action_tokens)
        action_end_pos = action_start_pos + len(action_tokens)

        # 6. Mask: history True (ignored), rest False (trainable)
        mask = [True] * len(history_tokens) + [False] * (
            len(tokens) - len(history_tokens)
        )
        labels = list(
            np.where(
                mask,
                CROSS_ENTROPY_IGNORE_IDX,
                tokens,
            )
        )
        return {
            "tokens": tokens,
            "labels": labels,
            "mask": mask,
            "action_start_pos": action_start_pos,
            "action_end_pos": action_end_pos,
            "end_of_prompt": end_of_prompt,
        }


def find_subsequence(sequence: List[Any], subsequence: List[Any]) -> int:
    """
    Finds the starting index of a subsequence within a sequence.
    Returns -1 if the subsequence is not found.
    """
    if not subsequence:
        return 0
    if not sequence:
        return -1
    for i in range(len(sequence) - len(subsequence) + 1):
        if sequence[i : i + len(subsequence)] == subsequence:
            return i
    return -1


def priv_dataset_online(
    tokenizer: ModelTokenizer,
    *,
    source: str,
    column_map: Optional[Dict[str, str]] = None,
    train_on_input: bool = False,
    new_system_prompt: Optional[str] = None,
    packed: bool = False,
    filter_fn: Optional[Callable] = None,
    split: str = "train",
    **load_dataset_kwargs: Dict[str, Any],
):
    # Set up column mapping for prompt/output structure
    if column_map is None:
        column_map = {"input": "prompt", "output": "output"}

    message_transform = InputOutputToMessages(
        train_on_input=train_on_input,
        column_map=column_map,
        new_system_prompt=new_system_prompt,
    )

    ds = priv_dataloader_online(
        source=source,
        message_transform=message_transform,
        model_transform=tokenizer,
        filter_fn=filter_fn,
        split=split,
        **load_dataset_kwargs,
    )

    if packed:
        if tokenizer.max_seq_len is None:
            raise ValueError(
                "PackedDataset requires a max_seq_len to be set on the tokenizer."
            )
        return PackedDataset(ds, max_seq_len=tokenizer.max_seq_len)
    return ds
