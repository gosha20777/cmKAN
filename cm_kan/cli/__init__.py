from pathlib import Path
import importlib
from .rich_argparse import (
    RichHelpFormatter,
    ArgumentDefaultsRichHelpFormatter,
)

__all__ = [
    "register_parsers",
    "RichHelpFormatter",
    "ArgumentDefaultsRichHelpFormatter",
]


def register_parsers(subparser):
    """Adds modules parsers to subparser"""

    for module_path in Path(__file__).parent.glob("*.py"):
        module = importlib.import_module("." + module_path.stem, __name__)
        if getattr(module, "add_parser", None) is not None:
            module.add_parser(subparser)
