"""
Download instructions & image features for Diagnose-VLN.
"""
import os
import sys
import urllib.request
import zipfile
import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument('--download_instructions', action='store_true')
parser.add_argument('--download_raw_instructions', action='store_true')
parser.add_argument('--instruction_dataset', type=str, default='r2r,rxr,touchdown')
parser.add_argument('--download_image_features', action='store_true')
parser.add_argument('--image_fearture_dataset', type=str, default='r2r,rxr')


def _progress_hook(count, block_size, total_size):
    percent = float(count * block_size) / float(total_size) * 100.
    sys.stdout.write(f'\r>> Downloading ... {percent:.1f}%')
    sys.stdout.flush()


def download_raw_instructions(args):
    dst_dir = f'./data_processing/process_instructions/'
    zip_filename = os.path.join(dst_dir, f'raw.zip')
    url = 'https://diagnose-vln.s3.amazonaws.com/instructions/raw_instructions.zip'
    urllib.request.urlretrieve(url, zip_filename, _progress_hook)
    print('\nExtracting ...')
    with zipfile.ZipFile(zip_filename) as zfile:
        zfile.extractall(dst_dir)
    os.remove(zip_filename)
    print(f'Successfully downloaded raw instructions. Extracted to {dst_dir}')


def download_processed_instructions(args):
    datasets = args.instruction_dataset.split(',')
    print(f'Will download instructions for the follwing datasets:\t{datasets}')
    for dataset in datasets:
        dst_dir = f'./{dataset}/data/'
        zip_filename = os.path.join(dst_dir, f'{dataset}_instr.zip')
        url = f'https://diagnose-vln.s3.amazonaws.com/instructions/{dataset}.zip'
        urllib.request.urlretrieve(url, zip_filename, _progress_hook)
        print('\nExtracting ...')
        with zipfile.ZipFile(zip_filename) as zfile:
            zfile.extractall(dst_dir)
        os.remove(zip_filename)
        print(f'Successfully downloaded {dataset} instructions for abaltions. Extracted to {dst_dir}')


def download_processed_image_features(args):
    img_feat_type = {
        'r2r': 'resnet152',
        'rxr': 'clipvit',
    }
    datasets = args.image_fearture_dataset.split(',')
    print(f'Will download image features for the follwing datasets:\t{datasets}')
    for dataset in datasets:
        dst_dir = f'./{dataset}/data/'
        zip_filename = os.path.join(dst_dir, f'{dataset}_img_feat.zip')
        url = f'https://diagnose-vln.s3.amazonaws.com/image_features/{img_feat_type[dataset]}.zip'
        urllib.request.urlretrieve(url, zip_filename, _progress_hook)
        print('\nExtracting ...')
        with zipfile.ZipFile(zip_filename) as zfile:
            zfile.extractall(dst_dir)
        os.remove(zip_filename)
        print(f'Successfully downloaded {dataset} image features for abaltions. Extracted to {dst_dir}')


if __name__ == '__main__':
    args = parser.parse_args()
    if args.download_instructions:
        download_processed_instructions(args)
    if args.download_raw_instructions:
        download_raw_instructions(args)
    if args.download_image_features:
        download_processed_image_features(args)
