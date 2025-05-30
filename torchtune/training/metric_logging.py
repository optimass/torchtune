# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from collections import defaultdict
import os
import pickle
import re
import sys
import time
from pathlib import Path

from typing import Any, Dict, List, Mapping, Optional, Union

import torch

from numpy import ndarray
from omegaconf import DictConfig, OmegaConf
from torchtune.training._distributed import get_world_size_and_rank

from torchtune.utils import get_logger
from typing_extensions import Protocol

Scalar = Union[torch.Tensor, ndarray, int, float]

log = get_logger("DEBUG")


class MetricLoggerInterface(Protocol):
    """Abstract metric logger."""

    def log(
        self,
        name: str,
        data: Scalar,
        step: int,
    ) -> None:
        """Log scalar data.

        Args:
            name (str): tag name used to group scalars
            data (Scalar): scalar data to log
            step (int): step value to record
        """
        pass

    def log_config(self, config: DictConfig) -> None:
        """Logs the config

        Args:
            config (DictConfig): config to log
        """
        pass

    def log_dict(self, payload: Mapping[str, Scalar], step: int) -> None:
        """Log multiple scalar values.

        Args:
            payload (Mapping[str, Scalar]): dictionary of tag name and scalar value
            step (int): step value to record
        """
        pass

    def close(self) -> None:
        """
        Close log resource, flushing if necessary.
        Logs should not be written after `close` is called.
        """
        pass


class DiskLogger(MetricLoggerInterface):
    """Logger to disk with in-memory accumulation of logs for pickling.

    Args:
        log_dir (str): directory to store logs
        filename (Optional[str]): optional filename to write logs to.
            Default: None, in which case log_{unixtimestamp}.txt will be used.
    Warning:
        This logger is not thread-safe.
    Note:
        This logger creates a new file based on the current time.
    """
    def __init__(self, log_dir: str, filename: Optional[str] = None, **kwargs):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if not filename:
            unix_timestamp = int(time.time())
            filename = f"log_{unix_timestamp}.txt"
        self._file_name = self.log_dir / filename
        self._file = open(self._file_name, "a")
        print(f"Writing logs to {self._file_name}")

        # In-memory log accumulation
        self._log_dict = defaultdict(list)

    def path_to_log_file(self) -> Path:
        return self._file_name

    def log(self, name: str, data: Scalar, step: int) -> None:
        self._file.write(f"Step {step} | {name}:{data}\n")
        self._file.flush()
        # Convert tensor to CPU value if needed

        self._log_dict[name].append((step, data))

    def log_dict(self, payload: Mapping[str, Scalar], step: int) -> None:
        self._file.write(f"Step {step} | ")
        for name, data in payload.items():
            if isinstance(data, torch.Tensor):
                data = data.cpu().item()
           
            self._file.write(f"{name}:{data} ")
            self._log_dict[name].append((step, data))
        self._file.write("\n")
        self._file.flush()
        
    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

        # Save log_dict to a pickle file
        pickle_path = self.log_dir / "log_dict.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump(dict(self._log_dict), f)
        print(f"Saved log dictionary to {pickle_path}")


class StdoutLogger(MetricLoggerInterface):
    """Logger to standard output."""

    def log(self, name: str, data: Scalar, step: int) -> None:
        print(f"Step {step} | {name}:{data}")

    def log_dict(self, payload: Mapping[str, Scalar], step: int) -> None:
        print(f"Step {step} | ", end="")
        for name, data in payload.items():
            print(f"{name}:{data} ", end="")
        print("\n", end="")

    def __del__(self) -> None:
        sys.stdout.flush()

    def close(self) -> None:
        sys.stdout.flush()


class WandBLogger(MetricLoggerInterface):
    """Logger for use w/ Weights and Biases application (https://wandb.ai/).
    For more information about arguments expected by WandB, see https://docs.wandb.ai/ref/python/init.

    Args:
        project (str): WandB project name. Default is `torchtune`.
        entity (Optional[str]): WandB entity name. If you don't specify an entity,
            the run will be sent to your default entity, which is usually your username.
        group (Optional[str]): WandB group name for grouping runs together. If you don't
            specify a group, the run will be logged as an individual experiment.
        log_dir (Optional[str]): WandB log directory. If not specified, use the `dir`
            argument provided in kwargs. Else, use root directory.
        run_id (Optional[str]): WandB run ID to resume. If specified, will attempt to continue
            logging to an existing run with this ID.
        resume (Optional[str]): How to resume the run. Options are "auto", "must", "allow", "never".
            Defaults to "auto". See https://docs.wandb.ai/guides/track/resuming for details.
        seed (Optional[int]): Random seed used for the experiment. Will be used to tag logs.
        **kwargs: additional arguments to pass to wandb.init

    Example:
        >>> from torchtune.training.metric_logging import WandBLogger
        >>> # Start a new run
        >>> logger = WandBLogger(project="my_project", entity="my_entity", group="my_group")
        >>> # Resume an existing run
        >>> logger = WandBLogger(project="my_project", run_id="abcd1234", resume="must")
        >>> logger.log("my_metric", 1.0, 1)
        >>> logger.log_dict({"my_metric": 1.0}, 1)
        >>> logger.close()

    Raises:
        ImportError: If ``wandb`` package is not installed.

    Note:
        This logger requires the wandb package to be installed.
        You can install it with `pip install wandb`.
        In order to use the logger, you need to login to your WandB account.
        You can do this by running `wandb login` in your terminal.
    """

    def __init__(
        self,
        project: str = "agents",
        entity: Optional[str] = None,
        group: Optional[str] = None,
        log_dir: Optional[str] = None,
        run_id: Optional[str] = None,
        epoch: Optional[int] = 0,
        seed: Optional[int] = None,
        **kwargs,
    ):
        
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "``wandb`` package not found. Please install wandb using `pip install wandb` to use WandBLogger."
                "Alternatively, use the ``StdoutLogger``, which can be specified by setting metric_logger_type='stdout'."
            ) from e
        self._wandb = wandb

        # Use dir if specified, otherwise use log_dir.
        self.log_dir = kwargs.pop("dir", log_dir)
        # Extract seed from log_dir if not provided explicitly
        self.seed = seed

        _, self.rank = get_world_size_and_rank()

        if self._wandb.run is None and self.rank == 0:
            # we check if wandb.init got called externally
            resume = 'allow'
            
            # Add seed to tags if available
            tags = kwargs.pop("tags", []) or []
            if self.seed is not None:
                tags.append(f"seed_{self.seed}")


            run_id_seed = f"torchtune_{run_id}"
            run = self._wandb.init(
                project='Torchtune_dummy',
                entity=entity,
                group=group,
                dir=self.log_dir,
                id=run_id_seed,
                name = run_id_seed,
                resume=resume,
                **kwargs,
            )
            
        if self._wandb.run:
            self._wandb.run._label(repo="torchtune")
            # Add seed as a config parameter to make it more visible
            if self.seed is not None:
                self._wandb.config.update({"seed": self.seed}, allow_val_change=True)

        # define default x-axis (for latest wandb versions)
        if getattr(self._wandb, "define_metric", None):
            self._wandb.define_metric("global_step")
            self._wandb.define_metric("*", step_metric="global_step", step_sync=True)

        self.config_allow_val_change = kwargs.get("allow_val_change", False)

    def log_config(self, config: DictConfig) -> None:
        """Saves the config locally and also logs the config to W&B. The config is
        stored in the same directory as the checkpoint. You can
        see an example of the logged config to W&B in the following link:
        https://wandb.ai/capecape/torchtune/runs/6053ofw0/files/torchtune_config_j67sb73v.yaml

        Args:
            config (DictConfig): config to log
        """
        if self._wandb.run:
            resolved = OmegaConf.to_container(config, resolve=True)
            self._wandb.config.update(
                resolved, allow_val_change=True
            )
            try:
                output_config_fname = Path(
                    os.path.join(
                        config.checkpointer.checkpoint_dir,
                        "torchtune_config.yaml",
                    )
                )
                OmegaConf.save(config, output_config_fname)

                log.info(f"Logging {output_config_fname} to W&B under Files")
                self._wandb.save(
                    output_config_fname, base_path=output_config_fname.parent
                )

            except Exception as e:
                log.warning(
                    f"Error saving {output_config_fname} to W&B.\nError: \n{e}."
                    "Don't worry the config will be logged the W&B workspace"
                )

    def log(self, name: str, data: Scalar, step: int) -> None:
        if self._wandb.run:
            payload = {name: data, "global_step": step}
            # Add seed to the log data for identification without changing the metric name
            if self.seed is not None:
                payload["seed"] = self.seed
            self._wandb.log(payload)

    def log_dict(self, payload: Mapping[str, Scalar], step: int) -> None:
        if self._wandb.run:
            
            metrics = {**payload, "global_step": step}
            metrics = {f'torchtune/{k}':v for k,v in metrics.items()}
            # Add seed to the log data for identification
            if self.seed is not None:
                metrics["seed"] = self.seed
            self._wandb.log(metrics)

    def __del__(self) -> None:
        # extra check for when there is an import error
        if hasattr(self, "_wandb") and self._wandb.run:
            self._wandb.finish()

    def close(self) -> None:
        if self._wandb.run:
            self._wandb.finish()


class TensorBoardLogger(MetricLoggerInterface):
    """Logger for use w/ PyTorch's implementation of TensorBoard (https://pytorch.org/docs/stable/tensorboard.html).

    Args:
        log_dir (str): torch.TensorBoard log directory
        organize_logs (bool): If `True`, this class will create a subdirectory within `log_dir` for the current
            run. Having sub-directories allows you to compare logs across runs. When TensorBoard is
            passed a logdir at startup, it recursively walks the directory tree rooted at logdir looking for
            subdirectories that contain tfevents data. Every time it encounters such a subdirectory,
            it loads it as a new run, and the frontend will organize the data accordingly.
            Recommended value is `True`. Run `tensorboard --logdir my_log_dir` to view the logs.
        **kwargs: additional arguments

    Example:
        >>> from torchtune.training.metric_logging import TensorBoardLogger
        >>> logger = TensorBoardLogger(log_dir="my_log_dir")
        >>> logger.log("my_metric", 1.0, 1)
        >>> logger.log_dict({"my_metric": 1.0}, 1)
        >>> logger.close()

    Note:
        This utility requires the tensorboard package to be installed.
        You can install it with `pip install tensorboard`.
        In order to view TensorBoard logs, you need to run `tensorboard --logdir my_log_dir` in your terminal.
    """

    def __init__(self, log_dir: str, organize_logs: bool = True, **kwargs):
        from torch.utils.tensorboard import SummaryWriter

        self._writer: Optional[SummaryWriter] = None
        _, self._rank = get_world_size_and_rank()

        # In case organize_logs is `True`, update log_dir to include a subdirectory for the
        # current run
        self.log_dir = (
            os.path.join(log_dir, f"run_{self._rank}_{time.time()}")
            if organize_logs
            else log_dir
        )

        # Initialize the log writer only if we're on rank 0.
        if self._rank == 0:
            self._writer = SummaryWriter(log_dir=self.log_dir)

    def log(self, name: str, data: Scalar, step: int) -> None:
        if self._writer:
            self._writer.add_scalar(name, data, global_step=step, new_style=True)

    def log_dict(self, payload: Mapping[str, Scalar], step: int) -> None:
        for name, data in payload.items():
            self.log(name, data, step)

    def __del__(self) -> None:
        if self._writer:
            self._writer.close()
            self._writer = None

    def close(self) -> None:
        if self._writer:
            self._writer.close()
            self._writer = None


class CometLogger(MetricLoggerInterface):
    """Logger for use w/ Comet (https://www.comet.com/site/).
    Comet is an experiment tracking tool that helps ML teams track, debug,
    compare, and reproduce their model training runs.

    For more information about arguments expected by Comet, see
    https://www.comet.com/docs/v2/guides/experiment-management/configure-sdk/#for-the-experiment.

    Args:
        api_key (Optional[str]): Comet API key. It's recommended to configure the API Key with `comet login`.
        workspace (Optional[str]): Comet workspace name. If not provided, uses the default workspace.
        project (Optional[str]): Comet project name. Defaults to Uncategorized.
        experiment_key (Optional[str]): The key for comet experiment to be used for logging. This is used either to
            append data to an Existing Experiment or to control the ID of new experiments (for example to match another
            ID). Must be an alphanumeric string whose length is between 32 and 50 characters.
        mode (Optional[str]): Control how the Comet experiment is started.

            * ``"get_or_create"``: Starts a fresh experiment if required, or persists logging to an existing one.
            * ``"get"``: Continue logging to an existing experiment identified by the ``experiment_key`` value.
            * ``"create"``: Always creates of a new experiment, useful for HPO sweeps.
        online (Optional[bool]): If True, the data will be logged to Comet server, otherwise it will be stored locally
            in an offline experiment. Default is ``True``.
        experiment_name (Optional[str]): Name of the experiment. If not provided, Comet will auto-generate a name.
        tags (Optional[List[str]]): Tags to associate with the experiment.
        log_code (bool): Whether to log the source code. Defaults to True.
        **kwargs (Dict[str, Any]): additional arguments to pass to ``comet_ml.start``. See
            https://www.comet.com/docs/v2/api-and-sdk/python-sdk/reference/Experiment-Creation/#comet_ml.ExperimentConfig

    Example:
        >>> from torchtune.training.metric_logging import CometLogger
        >>> logger = CometLogger(project_name="my_project", workspace="my_workspace")
        >>> logger.log("my_metric", 1.0, 1)
        >>> logger.log_dict({"my_metric": 1.0}, 1)
        >>> logger.close()

    Raises:
        ImportError: If ``comet_ml`` package is not installed.

    Note:
        This logger requires the comet_ml package to be installed.
        You can install it with ``pip install comet_ml``.
        You need to set up your Comet.ml API key before using this logger.
        You can do this by calling ``comet login`` in your terminal.
        You can also set it as the `COMET_API_KEY` environment variable.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        project: Optional[str] = None,
        experiment_key: Optional[str] = None,
        mode: Optional[str] = None,
        online: Optional[bool] = None,
        experiment_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        log_code: bool = True,
        **kwargs: Dict[str, Any],
    ):
        try:
            import comet_ml
        except ImportError as e:
            raise ImportError(
                "``comet_ml`` package not found. Please install comet_ml using `pip install comet_ml` to use CometLogger."
                "Alternatively, use the ``StdoutLogger``, which can be specified by setting metric_logger_type='stdout'."
            ) from e

        _, self.rank = get_world_size_and_rank()

        # Declare it early so further methods don't crash in case of
        # Experiment Creation failure due to mis-named configuration for
        # example
        self.experiment = None

        if self.rank == 0:
            self.experiment = comet_ml.start(
                api_key=api_key,
                workspace=workspace,
                project=project,
                experiment_key=experiment_key,
                mode=mode,
                online=online,
                experiment_config=comet_ml.ExperimentConfig(
                    log_code=log_code, tags=tags, name=experiment_name, **kwargs
                ),
            )

    def log(self, name: str, data: Scalar, step: int) -> None:
        if self.experiment is not None:
            self.experiment.log_metric(name, data, step=step)

    def log_dict(self, payload: Mapping[str, Scalar], step: int) -> None:
        if self.experiment is not None:
            self.experiment.log_metrics(payload, step=step)

    def log_config(self, config: DictConfig) -> None:
        if self.experiment is not None:
            resolved = OmegaConf.to_container(config, resolve=True)
            self.experiment.log_parameters(resolved)

            # Also try to save the config as a file
            try:
                self._log_config_as_file(config)
            except Exception as e:
                log.warning(f"Error saving Config to disk.\nError: \n{e}.")
                return

    def _log_config_as_file(self, config: DictConfig):
        output_config_fname = Path(
            os.path.join(
                config.checkpointer.checkpoint_dir,
                "torchtune_config.yaml",
            )
        )
        OmegaConf.save(config, output_config_fname)

        self.experiment.log_asset(
            output_config_fname, file_name="torchtune_config.yaml"
        )

    def close(self) -> None:
        if self.experiment is not None:
            self.experiment.end()

    def __del__(self) -> None:
        self.close()
