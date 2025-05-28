import csv
import os
from data.subsample import create_mask_for_mask_type
import logging
import pickle
import random
import xml.etree.ElementTree as etree
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
from warnings import warn
import pathlib

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset
from .transforms import build_transforms
from models.mri_ixi_t2net import IXIdataset
from matplotlib import pyplot as plt


def fetch_dir(key, data_config_file=pathlib.Path("fastmri_dirs.yaml")):
    """
    Data directory fetcher.

    This is a brute-force simple way to configure data directories for a
    project. Simply overwrite the variables for `knee_path` and `brain_path`
    and this function will retrieve the requested subsplit of the data for use.

    Args:
        key (str): key to retrieve path from data_config_file.
        data_config_file (pathlib.Path,
            default=pathlib.Path("fastmri_dirs.yaml")): Default path config
            file.

    Returns:
        pathlib.Path: The path to the specified directory.
    """
    if not data_config_file.is_file():
        default_config = dict(
            knee_path="/home/jc3/Data/",
            brain_path="/home/jc3/Data/",
        )
        with open(data_config_file, "w") as f:
            yaml.dump(default_config, f)

        raise ValueError(f"Please populate {data_config_file} with directory paths.")

    with open(data_config_file, "r") as f:
        data_dir = yaml.safe_load(f)[key]

    data_dir = pathlib.Path(data_dir)

    if not data_dir.exists():
        raise ValueError(f"Path {data_dir} from {data_config_file} does not exist.")

    return data_dir


def et_query(
        root: etree.Element,
        qlist: Sequence[str],
        namespace: str = "http://www.ismrm.org/ISMRMRD",
) -> str:
    """
    ElementTree query function.
    This can be used to query an xml document via ElementTree. It uses qlist
    for nested queries.
    Args:
        root: Root of the xml to search through.
        qlist: A list of strings for nested searches, e.g. ["Encoding",
            "matrixSize"]
        namespace: Optional; xml namespace to prepend query.
    Returns:
        The retrieved data as a string.
    """
    s = "."
    prefix = "ismrmrd_namespace"

    ns = {prefix: namespace}

    for el in qlist:
        s = s + f"//{prefix}:{el}"

    value = root.find(s, ns)
    if value is None:
        raise RuntimeError("Element not found")

    return str(value.text)


class SliceDataset(Dataset):
    def __init__(
            self,
            root,
            transform,
            challenge,
            sample_rate=1,
            mode='train'
    ):
        self.mode = mode

        # challenge
        if challenge not in ("singlecoil", "multicoil"):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')
        self.recons_key = (
            "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"
        )
        # transform
        self.transform = transform

        self.examples = []

        self.cur_path = root
        self.csv_file = os.path.join(self.cur_path, "singlecoil_" + self.mode + "_split_less.csv")

        # 读取CSV
        with open(self.csv_file, 'r') as f: # 读取 CSV 文件的每一行，提取文件名及对应的切片信息。
            reader = csv.reader(f)

            # 遍历 CSV 文件的每一行，获取与文件名相关的元数据和切片数量。使用私有方法 _retrieve_metadata 从 HDF5 文件中获取。
            id = 0

            for row in reader:
                pd_num_slices = self._retrieve_metadata(os.path.join(self.cur_path, row[0] + '.h5')) # 单模态只要一个输入 pd_metadata,

                # pdfs_metadata, pdfs_num_slices = self._retrieve_metadata(os.path.join(self.cur_path, row[1] + '.h5'))
                # 为每个切片创建一个元组，包含对应的 HDF5 文件路径（两个）、切片 ID 和相关的元数据，并将这个元组添加到 self.examples 列表中。
                for slice_id in range(pd_num_slices): # , pdfs_num_slices
                    self.examples.append( # 将一对.h5文件中的文件路径索引，每个kspace切片的id，两个文件的元数据和当前一对文件的id,append到examples中
                        (os.path.join(self.cur_path, row[0] + '.h5')
                         , slice_id, id)) # , os.path.join(self.cur_path, row[1] + '.h5')     # , pdfs_metadata , pd_metadata
                id += 1 # 处理完一对.h5文件

        if sample_rate < 1: # 如果 sample_rate 小于 1，随机打乱 self.examples 列表，然后选择一定比例的数据样本。
            random.shuffle(self.examples)
            num_examples = round(len(self.examples) * sample_rate)

            self.examples = self.examples[0:num_examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i): # 从 DataLoader 中获取一个批次的数据（例如：for data in dataloader_train:），DataLoader 会调用 __getitem__(i) 方法来加载具体的数据样本。
        # 允许使用索引 i 访问数据集中的数据样本

        # 读取pd  , pdfs_fname，, pdfs_metadata
        pd_fname, slice,  id = self.examples[i] # 列表中提取第 i 个样本的相关信息，包括 PD 和 PDF 文件的路径、切片索引、元数据和 ID。pd_metadata,

        with h5py.File(pd_fname, "r") as hf:
            #  pd_kspace ：numpy(640,368)
            pd_kspace = hf["kspace"][slice] # 从 HDF5 文件中读取 k-space 数据，使用 slice 索引获取对应的切片。

            pd_mask = np.asarray(hf["mask"]) if "mask" in hf else None

            # 使用预先设定的键 self.recons_key 从 HDF5 文件中读取目标图像。如果对应的键存在，则读取目标切片；如果不存在，则设为 None。
            # "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"
            pd_target = hf[self.recons_key][slice] if self.recons_key in hf else None

            # attrs = dict(hf.attrs) # 获取 HDF5 文件的所有属性，通过 hf.attrs 将其转换为字典，以便后续使用。

            #attrs.update(pd_metadata) # 将提取的元数据 pd_metadata 更新到文件属性中，以便在处理样本时能够获得更多信息。

        if self.transform is None:
            pd_sample = (pd_kspace, pd_mask, pd_target,  pd_fname, slice) # attrs,
        else:
            pd_sample = self.transform(pd_kspace, pd_mask, pd_target,  pd_fname, slice) # attrs,

        # with h5py.File(pdfs_fname, "r") as hf: # 根据 pdfs_fname，读到这个.h5文件，
        #     pdfs_kspace = hf["kspace"][slice]
        #     pdfs_mask = np.asarray(hf["mask"]) if "mask" in hf else None
        #
        #     pdfs_target = hf[self.recons_key][slice] if self.recons_key in hf else None
        #
        #     attrs = dict(hf.attrs)
        #
        #     attrs.update(pdfs_metadata)
        #
        # if self.transform is None:
        #     pdfs_sample = (pdfs_kspace, pdfs_mask, pdfs_target, attrs, pdfs_fname, slice)
        # else:
        #     pdfs_sample = self.transform(pdfs_kspace, pdfs_mask, pdfs_target, attrs, pdfs_fname, slice)


        # vis_data(pdfs_sample[0], pdfs_target[0], pd_fname, pdfs_fname, slice, 'vis_noise')

        return (pd_sample, id) # , pdfs_sample

    def _retrieve_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            # et_root = etree.fromstring(hf["ismrmrd_header"][()]) # 从 HDF5 文件中读取 "ismrmrd_header" 数据并解析为 XML 树结构，使用 etree 库进行解析。
            #
            # enc = ["encoding", "encodedSpace", "matrixSize"]
            # enc_size = (
            #     int(et_query(et_root, enc + ["x"])),
            #     int(et_query(et_root, enc + ["y"])),
            #     int(et_query(et_root, enc + ["z"])),
            # )
            # rec = ["encoding", "reconSpace", "matrixSize"]
            # recon_size = (
            #     int(et_query(et_root, rec + ["x"])),
            #     int(et_query(et_root, rec + ["y"])),
            #     int(et_query(et_root, rec + ["z"])),
            # )
            #
            # lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
            # enc_limits_center = int(et_query(et_root, lims + ["center"]))
            # enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1
            #
            # padding_left = enc_size[1] // 2 - enc_limits_center
            # padding_right = padding_left + enc_limits_max

            num_slices = hf["kspace"].shape[0]

        # metadata = {
        #     "padding_left": padding_left,
        #     "padding_right": padding_right,
        #     "encoding_size": enc_size,
        #     "recon_size": recon_size,
        # }

        return  num_slices # metadata,


def build_dataset(args, mode='train', sample_rate=1):
    assert mode in ['train', 'val', 'test2'], 'unknown mode'
    # mask = create_mask_for_mask_type(
    #     args.TRANSFORMS.MASKTYPE, args.TRANSFORMS.CENTER_FRACTIONS, args.TRANSFORMS.ACCELERATIONS,
    # )
    transforms = build_transforms(args, mode)  # 实例化这个方法。return LR_image, target, mean, std, fname, slice_num
    return SliceDataset(os.path.join(args.DATASET.ROOT, 'singlecoil_' + mode), transforms, args.DATASET.CHALLENGE,
                        sample_rate=sample_rate, mode=mode)




# 在 train函数里调用这个方法，并根据其传递的mode的值决定是找train数据集还是验证数据集
def _create_data_loader(args, mode='train'):
    assert mode in ['train', 'val', 'test1'], 'unknown mode'
    return IXIdataset(
        data_dir=os.path.join(args.DATASET.ROOT, mode)
        # data_dir=self.data_path,
        # alidtion_flag=data_partition is not 'train'  # 根据当前data_partition是不是train而决定validtion_flag True or False
    )