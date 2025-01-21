import argparse
import pytest
from cm_kan import cli


def add_parser(subparser: argparse) -> None:
    parser = subparser.add_parser(
        "unit-test",
        help="Run unit tests",
        formatter_class=cli.ArgumentDefaultsRichHelpFormatter,
    )

    parser.set_defaults(func=run_unit_tests)


def run_unit_tests(args: argparse.Namespace) -> None:
    pytest.main(["--rich", "tests"])
