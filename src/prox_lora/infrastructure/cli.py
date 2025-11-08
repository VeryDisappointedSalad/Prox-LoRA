import os
import random
import signal
import sys
import threading
import time
import traceback
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

    def __init__(self, main: Callable[[], None], seed: int, *, skip_torch: bool = False) -> None:
        self.main = main
        self.seed = seed
        self.skip_torch = skip_torch

    def run(self) -> None:
        self.before_main()
        self.main()
        self.after_main()

    def before_main(self) -> None:
        dotenv.load_dotenv()

        _setup_sigusr_handlers()

        # Seed like `lightning.fabric.utilities.seed.seed_everything`, without importing it.
        os.environ["PL_GLOBAL_SEED"] = str(self.seed)
        os.environ["PL_SEED_WORKERS"] = "1"
        random.seed(self.seed)
        np.random.seed(self.seed)  # noqa: NPY002

        if not self.skip_torch:
            _setup_torch(self.seed)

    def after_main(self) -> None:
        print("Finished.")


def _setup_torch(seed: int) -> None:
    """Setup torch compute precision options (and some printing options too)."""
    import torch  # noqa: PLC0415

    torch.manual_seed(seed)

    torch.set_float32_matmul_precision("medium")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.set_printoptions(precision=2, threshold=6, edgeitems=3, linewidth=200)  # type: ignore[no-untyped-call]
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
