"""
Run RSEN experiment on bibtex dataset with file-based logging.

Logs all training/validation metrics to a text file with immediate flush
so results are visible in real-time.

Usage:
    python run_rsen_experiment.py
"""

import os
import sys
import json
import time
import datetime
import shutil
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "best_models_configs" / "bibtex_strat_rsen" / "config.json"
OUTPUT_DIR = ROOT_DIR / "output" / "rsen_bibtex"
LOG_FILE = OUTPUT_DIR / "experiment_log.txt"


class FlushFileLogger:
    """Writes structured experiment logs to a text file with immediate flush."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, "w", encoding="utf-8")
        self._write_header()

    def _write_header(self):
        self.file.write("=" * 80 + "\n")
        self.file.write("RSEN (Representation-Space Energy Networks) Experiment Log\n")
        self.file.write(f"Started: {datetime.datetime.now().isoformat()}\n")
        self.file.write("=" * 80 + "\n\n")
        self.file.flush()

    def log(self, message: str):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}\n"
        self.file.write(line)
        self.file.flush()
        print(line, end="")  # also print to console

    def log_metrics(self, epoch: int, phase: str, metrics: dict):
        self.file.write(f"\n{'─' * 60}\n")
        self.file.write(f"Epoch {epoch} | {phase}\n")
        self.file.write(f"{'─' * 60}\n")
        for key in sorted(metrics.keys()):
            val = metrics[key]
            if isinstance(val, float):
                self.file.write(f"  {key:40s}: {val:.6f}\n")
            else:
                self.file.write(f"  {key:40s}: {val}\n")
        self.file.flush()

    def log_final(self, metrics: dict):
        self.file.write(f"\n{'=' * 80}\n")
        self.file.write("FINAL RESULTS\n")
        self.file.write(f"{'=' * 80}\n")
        for key in sorted(metrics.keys()):
            val = metrics[key]
            if isinstance(val, float):
                self.file.write(f"  {key:40s}: {val:.6f}\n")
            else:
                self.file.write(f"  {key:40s}: {val}\n")
        self.file.write(f"\nFinished: {datetime.datetime.now().isoformat()}\n")
        self.file.flush()

    def close(self):
        self.file.close()


# ── Global logger instance (set during run) ────────────────────────────
_flog: Optional[FlushFileLogger] = None


def _register_file_logging_callback():
    """Register a TrainerCallback that logs metrics to our file after each epoch."""
    from allennlp.training.callbacks import TrainerCallback
    from allennlp.training import GradientDescentTrainer
    from allennlp.data import TensorDict

    @TrainerCallback.register("rsen_file_logger")
    class RSENFileLoggerCallback(TrainerCallback):
        def __init__(self, serialization_dir: str, **kwargs):
            super().__init__(serialization_dir, **kwargs)

        def on_epoch(
            self,
            trainer: "GradientDescentTrainer",
            metrics: Dict[str, Any],
            epoch: int,
            is_primary: bool = True,
            **kwargs,
        ) -> None:
            if not is_primary or _flog is None:
                return

            # Extract train metrics
            train_metrics = {
                k.replace("training_", ""): v
                for k, v in metrics.items()
                if k.startswith("training_")
            }
            if train_metrics:
                _flog.log_metrics(epoch, "TRAIN", train_metrics)

            # Extract validation metrics
            val_metrics = {
                k.replace("validation_", ""): v
                for k, v in metrics.items()
                if k.startswith("validation_")
            }
            if val_metrics:
                _flog.log_metrics(epoch, "VALIDATION", val_metrics)

            # Log best epoch info
            best_epoch = metrics.get("best_epoch")
            best_f1 = metrics.get("best_validation_fixed_f1")
            if best_f1 is not None:
                _flog.log(
                    f"Epoch {epoch} done | "
                    f"best_epoch={best_epoch} | "
                    f"best_val_f1={best_f1:.6f}"
                )

        def on_end(
            self,
            trainer: "GradientDescentTrainer",
            metrics: Optional[Dict[str, Any]] = None,
            epoch: Optional[int] = None,
            is_primary: bool = True,
            **kwargs,
        ) -> None:
            if not is_primary or _flog is None:
                return
            if metrics:
                _flog.log("Training ended.")
                _flog.log_final(metrics)


def build_config():
    """Load and patch the config to remove wandb dependencies."""
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    # Remove wandb-dependent top-level type
    config.pop("type", None)

    # Replace callbacks: remove slurm, add our file logger
    config["trainer"]["callbacks"] = [
        "track_epoch_callback",
        "rsen_file_logger",
    ]

    return config


def run_experiment():
    global _flog
    _flog = FlushFileLogger(LOG_FILE)
    _flog.log(f"Config: {CONFIG_PATH}")
    _flog.log(f"Output: {OUTPUT_DIR}")

    # ── Import allennlp and seal (register all components) ──────────────
    from allennlp.common import util as common_util
    common_util.import_module_and_submodules("seal")

    # Register the file logging callback
    _register_file_logging_callback()

    from allennlp.common.params import Params
    from allennlp.commands.train import train_model
    from allennlp.training import util as training_util

    serialization_dir = str(OUTPUT_DIR / "serialization")

    # Load config
    config_dict = build_config()
    _flog.log("Config loaded successfully")
    _flog.log(f"Model type: {config_dict['model']['type']}")
    _flog.log(f"Score NN type: {config_dict['model']['score_nn']['type']}")
    _flog.log(f"Global score type: {config_dict['model']['score_nn']['global_score']['type']}")
    _flog.log(f"Loss fn type: {config_dict['model']['loss_fn']['type']}")
    _flog.log(
        f"Inference loss types: "
        f"{[cl['type'] for cl in config_dict['model']['inference_module']['loss_fn']['constituent_losses']]}"
    )
    _flog.log(f"Trainer: {config_dict['trainer']['type']}")
    _flog.log(f"Num epochs: {config_dict['trainer']['num_epochs']}")
    _flog.log(f"Inner mode: {config_dict['trainer']['inner_mode']}")
    _flog.log(f"Num steps: {config_dict['trainer']['num_steps']}")

    # Log full config
    _flog.file.write("\n--- Full Config ---\n")
    _flog.file.write(json.dumps(config_dict, indent=2))
    _flog.file.write("\n--- End Config ---\n\n")
    _flog.file.flush()

    # ── Clean up previous run ──────────────────────────────────────────
    if os.path.exists(serialization_dir):
        _flog.log("Removing previous serialization directory...")
        shutil.rmtree(serialization_dir)

    # ── Run training ───────────────────────────────────────────────────
    _flog.log("Starting training...")
    params = Params(config_dict)

    try:
        model = train_model(
            params=params,
            serialization_dir=serialization_dir,
        )
        _flog.log("Training completed successfully.")
    except Exception as e:
        _flog.log(f"Training failed with error: {e}")
        _flog.file.write(traceback.format_exc())
        _flog.file.flush()
        _flog.close()
        raise

    # ── Evaluate on test set ───────────────────────────────────────────
    _flog.log("Evaluating on test set...")
    try:
        import torch
        from allennlp.data.data_loaders import SimpleDataLoader
        from allennlp.data.dataset_readers import DatasetReader

        # Re-read config since Params consumes it
        with open(CONFIG_PATH, "r") as f:
            raw_config = json.load(f)

        test_data_path = raw_config.get("test_data_path")
        if test_data_path:
            reader_params = raw_config.get(
                "validation_dataset_reader", raw_config["dataset_reader"]
            )
            reader = DatasetReader.from_params(Params(reader_params))
            test_instances = list(reader.read(test_data_path))

            test_loader = SimpleDataLoader(
                test_instances,
                batch_size=raw_config["data_loader"]["batch_size"],
            )
            test_loader.index_with(model.vocab)

            cuda_device = raw_config["trainer"].get("cuda_device", -1)
            if cuda_device >= 0:
                test_loader.set_target_device(torch.device(f"cuda:{cuda_device}"))

            test_metrics = training_util.evaluate(
                model,
                test_loader,
                cuda_device=cuda_device,
            )
            _flog.log_metrics(-1, "TEST", test_metrics)
        else:
            _flog.log("No test_data_path specified, skipping test evaluation.")
    except Exception as e:
        _flog.log(f"Test evaluation failed: {e}")
        _flog.file.write(traceback.format_exc())
        _flog.file.flush()

    # ── Final summary ──────────────────────────────────────────────────
    final_metrics_path = os.path.join(serialization_dir, "metrics.json")
    if os.path.exists(final_metrics_path):
        with open(final_metrics_path) as f:
            final_metrics = json.load(f)
        _flog.log_final(final_metrics)

    _flog.close()
    print(f"\nExperiment log written to: {LOG_FILE}")


if __name__ == "__main__":
    run_experiment()
