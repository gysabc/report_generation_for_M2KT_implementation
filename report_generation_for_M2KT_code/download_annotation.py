# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from torchvision.datasets import MNIST
from zipfile import ZipFile
import argparse
import tarfile
# import gdown
import os
os.system("python download_annotation.py --data_dir C:/Users/LJP6/datasets")

# from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
# from wilds.datasets.fmow_dataset import FMoWDataset


def stage_path(data_dir, name):
    full_path = os.path.join(data_dir, name)

    if not os.path.exists(full_path):
        os.makedirs(full_path)

    return full_path


def download_and_extract(url, dst, remove=True):
    gdown.download(url, dst, quiet=False)
    if dst.endswith(".tar.gz"):
        tar = tarfile.open(dst, "r:gz")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".tar"):
        tar = tarfile.open(dst, "r:")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".zip"):
        zf = ZipFile(dst, "r")
        zf.extractall(os.path.dirname(dst))
        zf.close()

    if remove:
        os.remove(dst)

# MNIST #######################################################################


def download_mnist(data_dir):    
    # Original URL: http://yann.lecun.com/exdb/mnist/
    full_path = stage_path(data_dir, "MNIST")
    MNIST(full_path, download=True)

# Office-Home #################################################################


def download_office_home(data_dir):
    # Original URL: http://hemanthdv.org/OfficeHome-Dataset/
    full_path = stage_path(data_dir, "office_home")

    download_and_extract("https://drive.google.com/uc?id=1uY0pj7oFsjMxRwaD3Sxy0jgel0fsYXLC",
                         os.path.join(data_dir, "office_home.zip"))

    os.rename(os.path.join(data_dir, "OfficeHomeDataset_10072016"),
              full_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download datasets')
    parser.add_argument('--data_dir', type=str, default="data_dir C:/Users/LJP6/datasets")
    args = parser.parse_args()

    # download_mnist(args.data_dir)
    # download_office_home(args.data_dir)
    # download_domain_net(args.data_dir)
    # download_sviro(args.data_dir)
    # Camelyon17Dataset(root_dir=args.data_dir, download=True)
    # FMoWDataset(root_dir=args.data_dir, download=True)
