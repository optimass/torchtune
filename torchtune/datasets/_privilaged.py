import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

import numpy as np
import datasets

datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory=".": True

from datasets import load_dataset
from torch.utils.data import Dataset
from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX
from torchtune.data._messages import Message, validate_messages
from torchtune.modules.transforms import Transform

from torchtune.data import InputOutputToMessages
from torchtune.datasets._packed import PackedDataset

from torchtune.modules.tokenizers import ModelTokenizer


class priv_dataloader(Dataset):

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

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        prompt = sample["prompt"]
        output = sample["output"]

        # Detect if privileged information tags are present
        secret_pattern = r"<Secret information>.*?</Secret information>"
        privileged_found = (
            1 if re.search(secret_pattern, prompt, flags=re.DOTALL) else 0
        )

        # Remove secret information from the prompt
        prompt_no_secret = re.sub(secret_pattern, "", prompt, flags=re.DOTALL).strip()

        # Extract the last <action>...</action> block from the output, inclusive of tags
        action_blocks = list(
            re.finditer(r"<action>.*?</action>", output, flags=re.DOTALL)
        )

        # Get character positions of action block in original output
        action_start_char, action_end_char = (
            action_blocks[-1].span() if action_blocks else (0, 0)
        )

        # Split output into parts: before_action + action + after_action
        before_action = output[:action_start_char]

        # Scenario 1: p(output | prompt_with_secret)
        with_privilege = self._process_scenario(prompt, output, before_action)

        # Scenario 2: p(output | prompt_without_secret)
        without_privilege = self._process_scenario(
            prompt_no_secret, output, before_action
        )

        return {
            "with_privilege": with_privilege,
            "without_privilege": without_privilege,
            "reward": sample.get("match_reward", 1.0),
            "og_reward": sample["og_reward"],
            "goal": sample.get("trajectory_key", ""),
            "privileged_found": privileged_found,
            'trajectory_index' : sample.get('trajectory_index', 0),
            "step": sample.get("step_id", 0),
        }

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

    def _process_scenario(
        self, prompt: str, output: str, before_action: str
    ) -> Dict[str, Any]:
        """
        Helper function to process a single scenario (e.g. with or without privilege).
        Builds the tokenized sequence piece by piece for complete control over positions.
        """
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


def priv_dataset(
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

    ds = priv_dataloader(
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
