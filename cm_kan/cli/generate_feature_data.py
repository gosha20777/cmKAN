import argparse
import os
import random
import numpy as np
from rich.progress import Progress
from typing import List
import imageio
import asyncio
from color_transfer import cli


def add_parser(subparser: argparse) -> None:
    parser = subparser.add_parser(
        "create-feature-data",
        help="Create dataset",
        formatter_class=cli.ArgumentDefaultsRichHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        help="Path to input directory",
        required=True,
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=os.path.join('data', 'huawei'),
        help="Path to output directory",
        required=False,
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=42,
        help="Seed",
        required=False,
    )
    parser.add_argument(
        "-c", "--crop_size",
        type=int,
        default=1024,
        help="Crop size, set 0 to skip cropping",
        required=False,
    )
    parser.set_defaults(func=generate_dataset)


def parallel(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped


def _crop_image(image: np.ndarray, crop_size: int) -> List[np.ndarray]:
    if crop_size == 0:
        return [image]
    h,w,c = image.shape
    crop_list = []
    for y in range(crop_size, h, crop_size):
        for x in range(crop_size, w, crop_size):
            crop = image[
                y-crop_size:y,
                x-crop_size:x,
                0:c
            ]
            crop_list.append(crop)
    return crop_list


@parallel
def _prepare_train_data(
        input_feature_dir,
        save_train_src_feat_dir,
        save_train_ref_feat_dir,
        progress,
        train_pb,
        name,
    ):
    feature = np.load(os.path.join(input_feature_dir, name))
    src = feature[:,0:3]
    ref = feature[:,3:6]
    save_name = name.replace('_src_m.npy', '.npy')
    np.save(os.path.join(save_train_src_feat_dir, save_name), src)
    np.save(os.path.join(save_train_ref_feat_dir, save_name), ref)
    progress.update(train_pb, advance=1)


@parallel
def _prepare_val_data(
        input_src_img_dir,
        input_ref_img_dir,
        save_val_src_img_dir,
        save_val_ref_img_dir,
        progress,
        val_pb,
        args,
        name,
    ):
    src_name = name.replace('_src_m.npy', '_src_m.jpg')
    image = imageio.v3.imread(os.path.join(input_src_img_dir, src_name))
    crop_list = _crop_image(image, args.crop_size)
    for (i, image) in enumerate(crop_list):
        save_name = name.replace('_src_m.npy', f'_{i}.jpg')
        imageio.v3.imwrite(os.path.join(save_val_src_img_dir, save_name), image)

    ref_name = name.replace('_src_m.npy', '_ref_m.jpg')
    image = imageio.v3.imread(os.path.join(input_ref_img_dir, ref_name))
    crop_list = _crop_image(image, args.crop_size)
    for (i, image) in enumerate(crop_list):
        save_name = name.replace('_src_m.npy', f'_{i}.jpg')
        imageio.v3.imwrite(os.path.join(save_val_ref_img_dir, save_name), image)

    progress.update(val_pb, advance=1)


@parallel
def _prepare_test_data(
        input_src_img_dir,
        input_ref_img_dir,
        save_test_src_img_dir,
        save_test_ref_img_dir,
        progress,
        test_pb,
        args,
        name,
    ):
    src_name = name.replace('_src_m.npy', '_src_m.jpg')
    image = imageio.v3.imread(os.path.join(input_src_img_dir, src_name))
    crop_list = _crop_image(image, args.crop_size)
    for (i, image) in enumerate(crop_list):
        save_name = name.replace('_src_m.npy', f'_{i}.jpg')
        imageio.v3.imwrite(os.path.join(save_test_src_img_dir, save_name), image)

    ref_name = name.replace('_src_m.npy', '_ref_m.jpg')
    image = imageio.v3.imread(os.path.join(input_ref_img_dir, ref_name))
    crop_list = _crop_image(image, args.crop_size)
    for (i, image) in enumerate(crop_list):
        save_name = name.replace('_src_m.npy', f'_{i}.jpg')
        imageio.v3.imwrite(os.path.join(save_test_ref_img_dir, save_name), image)
            
    progress.update(test_pb, advance=1)


def generate_dataset(args: argparse.Namespace) -> None:
    input_src_img_dir = os.path.join(args.input, 'SRC matched')
    input_ref_img_dir = os.path.join(args.input, 'REF matched')
    input_feature_dir = os.path.join(args.input, 'src_ref_colors')

    if not os.path.exists(input_src_img_dir):
        raise Exception(f'No such directory: {input_src_img_dir}')
    if not os.path.exists(input_ref_img_dir):
        raise Exception(f'No such directory: {input_ref_img_dir}')
    if not os.path.exists(input_feature_dir):
        raise Exception(f'No such directory: {input_feature_dir}')
    
    save_test_src_img_dir = os.path.join(args.output, 'test', 'source')
    save_test_ref_img_dir = os.path.join(args.output, 'test', 'target')
    save_val_src_img_dir = os.path.join(args.output, 'val', 'source')
    save_val_ref_img_dir = os.path.join(args.output, 'val', 'target')
    save_train_src_feat_dir = os.path.join(args.output, 'train', 'source')
    save_train_ref_feat_dir = os.path.join(args.output, 'train', 'target')
    os.makedirs(save_test_src_img_dir, exist_ok=True)
    os.makedirs(save_test_ref_img_dir, exist_ok=True)
    os.makedirs(save_val_src_img_dir, exist_ok=True)
    os.makedirs(save_val_ref_img_dir, exist_ok=True)
    os.makedirs(save_train_src_feat_dir, exist_ok=True)
    os.makedirs(save_train_ref_feat_dir, exist_ok=True)

    names = os.listdir(input_feature_dir)
    random.seed(args.seed)
    random.shuffle(names)

    split = int(len(names)*0.3)
    val_names = names[:split]
    test_names = names[split:]

    with Progress() as progress:
        train_pb = progress.add_task("[cyan]Train features", total=len(names))
        val_pb = progress.add_task("[cyan]Val images", total=len(val_names))
        test_pb = progress.add_task("[cyan]Test images", total=len(test_names))

        loop = asyncio.get_event_loop()                                              # Have a new event loop
        looper = asyncio.gather(*[
            _prepare_train_data(
                input_feature_dir,
                save_train_src_feat_dir,
                save_train_ref_feat_dir,
                progress,
                train_pb,
                name,
            ) for name in names
        ])                       
        _ = loop.run_until_complete(looper) 

        loop = asyncio.get_event_loop()                                              # Have a new event loop
        looper = asyncio.gather(*[
            _prepare_val_data(
                input_src_img_dir,
                input_ref_img_dir,
                save_val_src_img_dir,
                save_val_ref_img_dir,
                progress,
                val_pb,
                args,
                name,
            ) for name in val_names
        ])                       
        _ = loop.run_until_complete(looper) 

        loop = asyncio.get_event_loop()                                              # Have a new event loop
        looper = asyncio.gather(*[
            _prepare_test_data(
                input_src_img_dir,
                input_ref_img_dir,
                save_test_src_img_dir,
                save_test_ref_img_dir,
                progress,
                test_pb,
                args,
                name,
            ) for name in test_names
        ])                       
        _ = loop.run_until_complete(looper)
