import argparse
import yaml
from ..core import Logger
from ..core.selector import (
    ModelSelector,
    DataSelector,
    PipelineSelector
)
from ..core.config import Config
from ..core.config.pipeline import PipelineType
import lightning as L
import os
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
    LearningRateMonitor,
)
from cm_kan.ml.callbacks import GenerateCallback
from lightning.pytorch.loggers import CSVLogger
from cm_kan import cli


def add_parser(subparser: argparse) -> None:
    parser = subparser.add_parser(
        "test",
        help="Test color transfer model",
        formatter_class=cli.ArgumentDefaultsRichHelpFormatter,
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        help="Path to config file",
        default="config.yaml",
        required=False,
    )
    parser.add_argument(
        "-w", "--weights",
        type=str,
        help="Path to checkpoint file in the experiment folder",
        default="logs/checkpoints/last.ckpt",
        required=False,
    )

    parser.set_defaults(func=test)


def test(args: argparse.Namespace) -> None:
    Logger.info(f"Loading config from '{args.config}'")
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    config = Config(**config)
    inference_mode = config.pipeline.type != PipelineType.pair_based
    if not inference_mode:
        Logger.info(f'Inference mode: {inference_mode}. Use optimization while testing.')
    Logger.info('Config:')
    config.print()
    
    dm = DataSelector.select(config)
    model = ModelSelector.select(config)
    pipeline = PipelineSelector.select(config, model)

    logger = CSVLogger(
        save_dir=os.path.join(config.save_dir, config.experiment),
        name='logs',
        version='',
    )

    trainer = L.Trainer(
        logger=logger,
        default_root_dir=os.path.join(config.save_dir, config.experiment),
        max_epochs=config.pipeline.params.epochs,
        accelerator=config.accelerator,
        callbacks=[
            RichProgressBar(),
            GenerateCallback(
                every_n_epochs=1,
            ),
        ],
        inference_mode=inference_mode,
    )

    ckpt_path = os.path.join(config.save_dir, config.experiment, args.weights)

    if not os.path.exists(ckpt_path):
        ckpt_path = None

    trainer.test(
        model=pipeline, 
        datamodule=dm,
        ckpt_path=ckpt_path,
    )
