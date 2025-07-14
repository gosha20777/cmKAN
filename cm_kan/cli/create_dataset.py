import argparse
import os
from pathlib import Path
import random
import numpy as np
from rich.progress import Progress, TaskID
from typing import List
import imageio
import asyncio
from cm_kan import cli


def add_parser(subparser: argparse) -> None:
    parser = subparser.add_parser(
        "create-dataset",
        help="Create dataset",
        formatter_class=cli.ArgumentDefaultsRichHelpFormatter,
    )
    parser.add_argument(
        "-s",
        "--source",
        type=str,
        help="Path to input directory with source images",
        default=os.path.join("data", "volga2k", "source"),
        required=False,
    )
    parser.add_argument(
        "-t",
        "--target",
        type=str,
        help="Path to input directory with target (reference) images",
        default=os.path.join("data", "volga2k", "reference"),
        required=False,
    )
    parser.add_argument(
        "-f",
        "--feature",
        type=str,
        help="Path to input directory with color features",
        default=os.path.join("data", "volga2k", "feature"),
        required=False,
    )
    parser.add_argument(
        "--use_feature",
        action="store_true",
        help=(
            "Generate dataset with *.npy color features "
            "(e.g. for classic pair based approaches)"
        ),
        default=False,
        required=False,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=os.path.join("data", "volga2k"),
        help="Path to output directory",
        required=False,
    )
    parser.add_argument(
        "-r",
        "--seed",
        type=int,
        default=42,
        help="Seed",
        required=False,
    )
    parser.add_argument(
        "-c",
        "--crop_size",
        type=int,
        default=1024,
        help="Crop size, set 0 to skip cropping",
        required=False,
    )
    parser.add_argument(
        "-j",
        "--threads",
        type=int,
        default=min(os.cpu_count(), 8),
        help="Number of threads",
        required=False,
    )
    parser.set_defaults(func=generate_dataset)


def parallel(f) -> callable:
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(
            None, f, *args, **kwargs)

    return wrapped


def _crop_image(image: np.ndarray, crop_size: int) -> List[np.ndarray]:
    if crop_size == 0:
        return [image]
    h, w, c = image.shape
    crop_list = []
    for y in range(crop_size, h, crop_size):
        for x in range(crop_size, w, crop_size):
            crop = image[y-crop_size:y, x-crop_size:x, 0:c]
            crop_list.append(crop)
    return crop_list


@parallel
def _prepare_data(
    input_src_img_dir: Path,
    input_ref_img_dir: Path,
    input_feature_dir: Path,
    save_train_src_dir: Path,
    save_train_ref_dir: Path,
    name: str,
    progress: Progress,
    pb: TaskID,
    args,
) -> None:
    source_path = input_src_img_dir.joinpath(name)
    if not source_path.is_file():
        raise Exception("No source file")

    target_path = input_ref_img_dir.joinpath(name.replace("_src_m", "_ref_m"))
    if not target_path.is_file():
        raise Exception("No target file")

    if input_feature_dir is not None:
        feature_path = input_feature_dir.joinpath(Path(name).stem + ".npy")
        if not feature_path.is_file():
            raise Exception("No feature file")
    try:
        image = imageio.v3.imread(source_path)
        crop_list = _crop_image(image, args.crop_size)
        for i, image in enumerate(crop_list):
            save_name = name.replace("_src_m", f"_{i}")
            imageio.v3.imwrite(save_train_src_dir.joinpath(save_name), image)

        image = imageio.v3.imread(target_path)
        crop_list = _crop_image(image, args.crop_size)
        for i, image in enumerate(crop_list):
            save_name = name.replace("_src_m", f"_{i}")
            imageio.v3.imwrite(save_train_ref_dir.joinpath(save_name), image)

        if input_feature_dir is not None:
            feature = np.load(feature_path)
            src = feature[:, 0:3]
            ref = feature[:, 3:6]
            for i, _ in enumerate(crop_list):
                save_name = Path(name.replace("_src_m", f"_{i}")).stem + ".npy"
                np.save(os.path.join(save_train_src_dir, save_name), src)
                np.save(os.path.join(save_train_ref_dir, save_name), ref)
    except Exception as e:
        print(f"Error: {e}")
        raise Exception(f"Error during processing {name}!", e)

    progress.update(pb, advance=1)


def generate_dataset(args: argparse.Namespace) -> None:
    input_src_img_dir = Path(args.source)
    input_ref_img_dir = Path(args.target)
    input_feature_dir = Path(args.feature) if args.use_feature else None
    output_dir = Path(args.output)

    if not input_src_img_dir.is_dir():
        raise Exception(f"No such directory: {input_src_img_dir}")
    if not input_ref_img_dir.is_dir():
        raise Exception(f"No such directory: {input_ref_img_dir}")
    if input_feature_dir is not None and not input_ref_img_dir.is_dir():
        raise Exception(f"No such directory: {input_ref_img_dir}")

    save_test_src_dir = output_dir.joinpath("test", "source")
    save_test_ref_dir = output_dir.joinpath("test", "target")
    save_val_src_dir = output_dir.joinpath("val", "source")
    save_val_ref_dir = output_dir.joinpath("val", "target")
    save_train_src_dir = output_dir.joinpath("train", "source")
    save_train_ref_dir = output_dir.joinpath("train", "target")

    save_test_src_dir.mkdir(parents=True, exist_ok=True)
    save_test_ref_dir.mkdir(parents=True, exist_ok=True)
    save_val_src_dir.mkdir(parents=True, exist_ok=True)
    save_val_ref_dir.mkdir(parents=True, exist_ok=True)
    save_train_src_dir.mkdir(parents=True, exist_ok=True)
    save_train_ref_dir.mkdir(parents=True, exist_ok=True)

    files = list(input_src_img_dir.glob("*.[jpg png bmp]*"))
    random.seed(args.seed)
    random.shuffle(files)

    n = len(files)
    split = np.cumsum([int(0.7 * n), int(0.1 * n)])
    train_files = files[:split[0]]
    val_files = files[split[0]:split[1]]
    test_files = files[split[1]:]

    with Progress() as progress:
        train_pb = progress.add_task(
            "[cyan]Train images",
            total=len(train_files)
        )
        val_pb = progress.add_task("[cyan]Val images", total=len(val_files))
        test_pb = progress.add_task("[cyan]Test images", total=len(test_files))

        loop = asyncio.get_event_loop()  # Have a new event loop
        looper = asyncio.gather(
            *[
                _prepare_data(
                    input_src_img_dir,
                    input_ref_img_dir,
                    input_feature_dir,
                    save_train_src_dir,
                    save_train_ref_dir,
                    filename.name,
                    progress,
                    train_pb,
                    args,
                )
                for filename in train_files
            ]
        )
        _ = loop.run_until_complete(looper)

        loop = asyncio.get_event_loop()  # Have a new event loop
        looper = asyncio.gather(
            *[
                _prepare_data(
                    input_src_img_dir,
                    input_ref_img_dir,
                    input_feature_dir,
                    save_val_src_dir,
                    save_val_ref_dir,
                    filename.name,
                    progress,
                    val_pb,
                    args,
                )
                for filename in val_files
            ]
        )
        _ = loop.run_until_complete(looper)

        loop = asyncio.get_event_loop()  # Have a new event loop
        looper = asyncio.gather(
            *[
                _prepare_data(
                    input_src_img_dir,
                    input_ref_img_dir,
                    input_feature_dir,
                    save_test_src_dir,
                    save_test_ref_dir,
                    filename.name,
                    progress,
                    test_pb,
                    args,
                )
                for filename in test_files
            ]
        )
        _ = loop.run_until_complete(looper)
