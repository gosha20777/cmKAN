import logging
from rich.logging import RichHandler
from rich.console import Console
from rich.traceback import install


class Logger:
    install(show_locals=False)

    rich_handler = RichHandler(
        level="INFO",
        console=Console(),
        show_time=False,
        show_path=False,
        markup=True,
    )
    rich_handler.setFormatter(logging.Formatter("%(message)s"))

    _logger = logging.getLogger()
    _logger.handlers.clear()

    _logger = logging.getLogger("lightning.pytorch.trainer")
    _logger.handlers.clear()

    _logger = logging.getLogger("lightning.pytorch")
    _logger.handlers.clear()
    _logger.addHandler(rich_handler)

    _logger = logging.getLogger("lightning")
    _logger.handlers.clear()

    logging.basicConfig(level=logging.INFO, handlers=[rich_handler])
    logging.captureWarnings(True)

    @staticmethod
    def info(msg: object) -> None:
        Logger._logger.info(msg)

    @staticmethod
    def error(msg: object) -> None:
        Logger._logger.error(msg)

    @staticmethod
    def critical(msg: str) -> None:
        Logger._logger.critical(msg)
