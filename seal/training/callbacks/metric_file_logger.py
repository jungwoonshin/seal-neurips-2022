from typing import Dict, Any, List, Optional
import os
import logging

from allennlp.training import GradientDescentTrainer
from allennlp.training.callbacks import TrainerCallback

logger = logging.getLogger(__name__)


@TrainerCallback.register("metric-file-logger")
class MetricFileLogger(TrainerCallback):
    """
    Writes per-epoch metrics to a text file with aligned columns and flushes after each epoch.
    """

    def __init__(
        self,
        serialization_dir: str,
        log_filename: str = "epoch_metrics.log",
    ) -> None:
        super().__init__(serialization_dir=serialization_dir)
        self.log_path = os.path.join(serialization_dir, log_filename)
        self._file = None
        self._header_written = False
        self._keys: List[str] = []

    def _collect_keys(self, metrics: Dict[str, Any]) -> List[str]:
        return sorted(
            k for k, v in metrics.items()
            if isinstance(v, (int, float, str, bool))
        )

    def _format_value(self, v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.6f}"
        return str(v)

    def on_start(
        self,
        trainer: "GradientDescentTrainer",
        is_primary: bool = True,
        **kwargs: Any,
    ) -> None:
        if is_primary:
            self._file = open(self.log_path, "w")
            self._header_written = False
            logger.info("MetricFileLogger: writing to %s", self.log_path)

    def on_epoch(
        self,
        trainer: "GradientDescentTrainer",
        metrics: Dict[str, Any],
        epoch: int,
        is_primary: bool = True,
        **kwargs: Any,
    ) -> None:
        if not is_primary or self._file is None:
            return

        serializable = {
            k: v for k, v in metrics.items()
            if isinstance(v, (int, float, str, bool))
        }

        if not self._header_written:
            self._keys = self._collect_keys(serializable)
            widths = [max(len(k), 12) for k in self._keys]
            header = "  ".join(k.rjust(w) for k, w in zip(self._keys, widths))
            self._file.write(f"{'epoch':>7}  {header}\n")
            self._file.write("-" * (9 + sum(w + 2 for w in widths)) + "\n")
            self._header_written = True

        # handle new keys that appeared after header
        for k in sorted(serializable.keys()):
            if k not in self._keys:
                self._keys.append(k)

        widths = [max(len(k), 12) for k in self._keys]
        vals = []
        for k, w in zip(self._keys, widths):
            v = serializable.get(k, "")
            vals.append(self._format_value(v).rjust(w))
        line = f"{epoch:>7}  {'  '.join(vals)}\n"
        self._file.write(line)
        self._file.flush()

    def on_end(
        self,
        trainer: "GradientDescentTrainer",
        metrics: Dict[str, Any] = None,
        epoch: int = None,
        is_primary: bool = True,
        **kwargs: Any,
    ) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
