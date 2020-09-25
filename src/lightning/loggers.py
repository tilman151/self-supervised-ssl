import os
import tempfile
from typing import Optional

import pytorch_lightning.loggers as loggers
from pytorch_lightning.callbacks import ModelCheckpoint

ExperimentNaming = {1: "one", 2: "two", 3: "three", 4: "four"}


def baseline_experiment_name(source):
    assert source in ExperimentNaming, f"Unknown FD number {source}."
    return f"cmapss_{ExperimentNaming[source]}_baseline"


def transfer_experiment_name(source, target):
    assert source in ExperimentNaming, f"Unknown source FD number {source}."
    assert target in ExperimentNaming, f"Unknown target FD number {target}."
    return f"{ExperimentNaming[source]}2{ExperimentNaming[target]}"


def pretraining_experiment_name(source, target):
    if target is None:
        return pretraining_baseline_experiment_name(source)
    else:
        return pretraining_adaption_experiment_name(source, target)


def pretraining_baseline_experiment_name(dataset):
    assert dataset in ExperimentNaming, f"Unknown dataset FD number {dataset}."

    return f"pretraining_{ExperimentNaming[dataset]}"


def pretraining_adaption_experiment_name(source, target):
    assert source in ExperimentNaming, f"Unknown source FD number {source}."
    assert target in ExperimentNaming, f"Unknown target FD number {target}."

    return f"pretraining_{ExperimentNaming[source]}&{ExperimentNaming[target]}"


def transfer_hyperopt_name(source, target, percent_broken):
    assert source in ExperimentNaming, f"Unknown source FD number {source}."
    assert target in ExperimentNaming, f"Unknown target FD number {target}."
    assert 0 <= percent_broken <= 1, f"Invalid percent broken {percent_broken}."

    return f"hyperopt_{ExperimentNaming[source]}&{ExperimentNaming[target]}@{percent_broken:.2f}"


def pretraining_hyperopt_name(source, target, percent_broken):
    assert source in ExperimentNaming, f"Unknown source FD number {source}."
    assert target in ExperimentNaming, f"Unknown target FD number {target}."
    assert 0 <= percent_broken <= 1, f"Invalid percent broken {percent_broken}."

    return f"pre_hyperopt_{ExperimentNaming[source]}&{ExperimentNaming[target]}@{percent_broken:.2f}"


def semi_supervised_hyperopt_name(source, percent_broken):
    assert source in ExperimentNaming, f"Unknown source FD number {source}."
    assert 0 <= percent_broken <= 1, f"Invalid percent broken {percent_broken}."

    return f"pre_hyperopt_{ExperimentNaming[source]}@{percent_broken:.2f}"


class MLTBLogger(loggers.LoggerCollection):
    """Combined MlFlow and Tensorboard logger that saves models as MlFlow artifacts."""

    def __init__(self, log_dir, experiment_name, tag=None, tensorboard_struct=None):
        """
        This logger combines a MlFlow and Tensorboard logger.
        It creates a directory (mlruns/tensorboard) for each logger in log_dir.

        If a tensorboard_struct dict is provided, it is used to create additional
        sub-directories for tensorboard to get a better overview over the runs.

        The difference to a simple LoggerCollection is that the save dir points
        to the artifact path of the MlFlow run. This way the model is logged as
        a MlFlow artifact.

        :param log_dir: directory to put the mlruns and tensorboard directories
        :param experiment_name: name for the experiment
        :param tensorboard_struct: dictionary containing information to refine the tensorboard directory structure
        """
        tensorboard_path = os.path.join(log_dir, "tensorboard", experiment_name)
        sub_dirs = self._dirs_from_dict(tensorboard_struct)
        self._tf_logger = loggers.TensorBoardLogger(tensorboard_path, name=sub_dirs)

        mlflow_path = "file:" + os.path.normpath(os.path.join(log_dir, "mlruns"))
        self._mlflow_logger = loggers.MLFlowLogger(
            experiment_name, tracking_uri=mlflow_path, tags=self._build_tags(tag)
        )

        super().__init__([self._tf_logger, self._mlflow_logger])

    @staticmethod
    def _dirs_from_dict(tensorboard_struct):
        if tensorboard_struct is not None:
            dirs = []
            for key, value in tensorboard_struct.items():
                if isinstance(value, float) and value >= 0.1:
                    dirs.append(f"{value:.1f}{key}")
                elif isinstance(value, str):
                    dirs.append(f"{key}-{value}")
                else:
                    dirs.append(f"{value}{key}")
            dirs = os.path.join(*dirs)
        else:
            dirs = ""

        return dirs

    def _build_tags(self, tag):
        if tag is not None:
            tags = {"version": tag}
        else:
            tags = None

        return tags

    @property
    def name(self) -> str:
        return self._mlflow_logger.name

    @property
    def version(self) -> str:
        return os.path.join(self._mlflow_logger.version, "artifacts")

    @property
    def save_dir(self):
        return self._mlflow_logger.save_dir

    @property
    def checkpoint_path(self):
        return os.path.join(self.save_dir, self.name, self.version, "checkpoints")

    @property
    def tf_experiment(self):
        return self._tf_logger.experiment

    @property
    def mlflow_experiment(self):
        return self._mlflow_logger.experiment

    @property
    def run_id(self):
        return self._mlflow_logger.run_id

    def log_figure(self, tag, figure, step):
        tag_tail, tag_head = os.path.split(tag)
        _, tmp_file = tempfile.mkstemp(".pdf", f"{tag_head}_{step:05}_")
        figure.savefig(tmp_file)
        self.mlflow_experiment.log_artifact(
            self._mlflow_logger.run_id, tmp_file, tag_tail
        )
        self.tf_experiment.add_figure(tag, figure, step)


class MinEpochModelCheckpoint(ModelCheckpoint):
    """Checkpoints models only after training for a minimum of epochs."""

    def __init__(
        self,
        filepath: Optional[str] = None,
        monitor: Optional[str] = None,
        verbose: bool = False,
        save_last: Optional[bool] = None,
        save_top_k: Optional[int] = None,
        save_weights_only: bool = False,
        mode: str = "auto",
        period: int = 1,
        prefix: str = "",
        min_epochs_before_saving: int = 0,
    ):
        super().__init__(
            filepath,
            monitor,
            verbose,
            save_last,
            save_top_k,
            save_weights_only,
            mode,
            period,
            prefix,
        )

        self.min_epochs_before_saving = min_epochs_before_saving

    def save_checkpoint(self, trainer, pl_module):
        if trainer.current_epoch > self.min_epochs_before_saving:
            super().save_checkpoint(trainer, pl_module)
