import argparse
from concurrent.futures import ThreadPoolExecutor, ALL_COMPLETED, wait
import os
from pathlib import Path
import random
import numpy as np
from rich.progress import Progress
from typing import List
import imageio
from .utils import concurrent
from ..ml.transforms import ColorMatching
from color_transfer import cli

THREADS = 8

def add_parser(subparser: argparse) -> None:
    parser = subparser.add_parser(
        "build-features",
        help="Create features",
        formatter_class=cli.ArgumentDefaultsRichHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Path to input directory",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path to output directory",
        required=False,
    )
    parser.add_argument(
        "-k",
        "--save_keypoints",
        action='store_true',
        help="Save features key points",
        default=True,
        required=False,
    )
    parser.set_defaults(func=build_features)


@concurrent
def _build_features(
    input_src_dir,
    input_ref_dir,
    output_features_dir,
    file_name,
    save_keypoints,
):
    src_file = input_src_dir.joinpath(file_name)
    ref_file = input_ref_dir.joinpath(file_name)

    src_img = imageio.v3.imread(src_file)
    ref_img = imageio.v3.imread(ref_file)

    matching = ColorMatching()
    src_points, ref_points = matching.match_features(src_img, ref_img)

    if save_keypoints:
        points = np.concatenate([src_points, ref_points], axis=1)
        out_file = output_features_dir.joinpath(src_file.stem + '_kp.npy')
        np.save(str(out_file), points)

    src_features = matching.yank_colors(src_img, src_points, 3)
    ref_features = matching.yank_colors(ref_img, ref_points, 3)

    features = np.concatenate([src_features, ref_features], axis=1)
    out_file = output_features_dir.joinpath(src_file.stem + '.npy')
    np.save(str(out_file), features)


def build_features(args: argparse.Namespace) -> None:

    input_path = Path(args.input)
    input_src_dir = input_path.joinpath('src')
    input_ref_dir = input_path.joinpath('ref')
    output_features_dir = Path(args.output)

    if not input_src_dir.is_dir():
        raise Exception(f'No such directory: {input_src_dir}')
    if not input_ref_dir.is_dir():
        raise Exception(f'No such directory: {input_ref_dir}')

    output_features_dir.mkdir(exist_ok=True, parents=True)

    files = list(input_src_dir.glob('*.[jpg png bmp]*'))

    with Progress() as progress:
        progress_id = progress.add_task("[cyan]Train features",
                                        total=len(files))

        with ThreadPoolExecutor(max_workers=THREADS) as executor:
            tasks = [
                _build_features(executor, input_src_dir, input_ref_dir,
                                output_features_dir, file, args.save_keypoints)
                for file in files
            ]
            for task in tasks:
                task.add_done_callback(
                    lambda _: progress.update(progress_id, advance=1))

            _, not_done = wait(tasks, return_when=ALL_COMPLETED)

            if len(not_done) > 0:
                print(f'[Warn] Skipped {len(not_done)} image pairs.')
