#!/usr/bin/env python3
"""Browse trained model runs and choose a checkpoint to launch.

Usage:
    python -m mm_transfer.gui.launch_gui
    python -m mm_transfer.gui.launch_gui --model-folder _trained_models
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mm_transfer.gui.browser import ModelBrowser
from mm_transfer.gui.util import resolve_model_folder


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browse trained model folders and choose a checkpoint.")
    parser.add_argument(
        "--model-folder",
        default="_trained_models",
        metavar="PATH",
        help="Directory of trained models to browse (default: %(default)s).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    ModelBrowser(resolve_model_folder(args.model_folder)).mainloop()


if __name__ == "__main__":
    main()
