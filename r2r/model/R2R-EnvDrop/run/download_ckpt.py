"""
Download instructions & image features for Diagnose-VLN.
"""
import os
import sys
import urllib.request
import zipfile


def _progress_hook(count, block_size, total_size):
    percent = float(count * block_size) / float(total_size) * 100.
    sys.stdout.write(f'\r>> Downloading ... {percent:.1f}%')
    sys.stdout.flush()


def check_dir(d):
    if os.path.isdir(d):
        print(d, '\tEXISTS!')
    else:
        os.mkdir(d)
        print(d, '\tCREATED!')


if __name__ == '__main__':
    zip_filename = './r2r_ckpt.zip'
    dst_dir = './snap'
    check_dir(dst_dir)
    url = f'https://diagnose-vln.s3.amazonaws.com/checkpoint/r2r_ckpt.zip'
    urllib.request.urlretrieve(url, zip_filename, _progress_hook)
    print('\nExtracting ...')
    with zipfile.ZipFile(zip_filename) as zfile:
        zfile.extractall(dst_dir)
    os.remove(zip_filename)
    print(f'Successfully downloaded R2R checkpoints. Extracted to {dst_dir}')
