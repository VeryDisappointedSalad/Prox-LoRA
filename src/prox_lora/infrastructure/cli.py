import logging
import os
import random
import signal
import sys
import threading
import time
import traceback
import warnings
from collections.abc import Callable
from typing import Any

import dotenv
import numpy as np


class CLI:
    """
    Our common entry point: initializes env vars, loggers, seeds, torch options, etc.

    Args:
    - main: the main function to be called.
    - skip_torch: if True, don't import and initialize torch and Lightning (which takes a few seconds).
    """

    def __init__(self, main: Callable[[], None], *, skip_torch: bool = False) -> None:
        self.main = main
        self.skip_torch = skip_torch

    def run(self) -> None:
        self.before_main()
        self.main()
        self.after_main()

    def before_main(self) -> None:
        dotenv.load_dotenv()

        _setup_sigusr_handlers()
        _silence_spurious_warnings()
        if not self.skip_torch:
            _setup_torch()

    def after_main(self) -> None:
        print("Finished.")


def seed_everything(seed: int, *, skip_torch: bool = False) -> None:
    """Seed like `lightning.fabric.utilities.seed.seed_everything`, without importing it."""
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    os.environ["PL_SEED_WORKERS"] = "1"
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002

    if not skip_torch:
        import torch  # noqa: PLC0415

        torch.manual_seed(seed)


def _setup_torch() -> None:
    """Setup torch compute precision options (and some printing options too)."""
    import torch  # noqa: PLC0415

    torch.set_printoptions(precision=2, threshold=6, edgeitems=3, linewidth=200)  # type: ignore[no-untyped-call]

    torch.backends.fp32_precision = "tf32"  # type: ignore[attr-defined]
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.fp32_precision = "tf32"  # type: ignore[attr-defined]
    torch.backends.cudnn.conv.fp32_precision = "tf32"  # type: ignore[attr-defined]
    torch.backends.cudnn.rnn.fp32_precision = "tf32"  # type: ignore[attr-defined]
    torch.set_float32_matmul_precision("medium")

    if not torch.backends.cuda.flash_sdp_enabled():  # type: ignore[no-untyped-call]
        print("Warning: flash scaled-dot-product attention is not available :(")


def _setup_sigusr_handlers() -> None:
    """Setup a signal handler for SIGUSR1 to print the current stack trace and SIGUSR2 to sleep 10s."""

    def sigusr1_handler(signum: int, frame: Any) -> None:
        signame = "SIGUSR1" if signum == signal.SIGUSR1 else ("SIGUSR2" if signum == signal.SIGUSR2 else f"{signum=}")
        print(f"Received {signame}, printing stack trace:")
        for th in threading.enumerate():
            if th.name.startswith("loguru"):
                continue
            print(f"\n ### Thread {th.name} (tid={th.native_id}) ###")
            bts = traceback.format_stack(sys._current_frames()[th.ident or 0])  # noqa: SLF001
            print("".join(bts[2:]))

    def sigusr2_handler(signum: int, frame: Any) -> None:
        sigusr1_handler(signum, frame)
        print("Pausing for 10 seconds...")
        time.sleep(10)
        print("Resuming...")

    signal.signal(signal.SIGUSR1, sigusr1_handler)
    signal.signal(signal.SIGUSR2, sigusr2_handler)


def _silence_spurious_warnings() -> None:
    """Setup warning filters to silence a few useless ones."""

    # open-clip-torch=3.2.0's TimmModel uses a deprecated import from timm.
    warnings.filterwarnings("ignore", message=".*Importing from timm.models.helpers is deprecated")

    # torch=2.10.0 uses it's own deprecated functionality.
    warnings.filterwarnings("ignore", message=r"(?s).*isinstance\(treespec, LeafSpec\)")

    # torchvision=0.25.0's cifar10 dataset uses an pickle with old numpy arrays.
    warnings.filterwarnings("ignore", message=".* align should be passed as Python or NumPy boolean but got")

    logging.getLogger("lightning.pytorch.utilities.rank_zero").addFilter(_TipFilter())


class _TipFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "💡 Tip: For seamless cloud logging" not in record.getMessage()
