import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib_inline

from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    CenterSpatialCropd,
    Resized,
    SpatialPadd,
    apply_transform,
    RandZoomd,
    RandCropByLabelClassesd,
    NormalizeIntensityd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandAffined
)

import collections.abc
import math
import pickle
import shutil
import sys
import tempfile
import threading
import time
import warnings
from copy import copy, deepcopy
import h5py


import numpy as np
import torch
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

from torch.utils.data import Subset, ConcatDataset

from monai.data import DataLoader, Dataset, list_data_collate, DistributedSampler, CacheDataset
from monai.config import DtypeLike, KeysCollection
from monai.transforms.transform import Transform, MapTransform
from monai.utils.enums import TransformBackends
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.io.array import LoadImage, SaveImage
from monai.utils import GridSamplePadMode, ensure_tuple, ensure_tuple_rep
from monai.data.image_reader import ImageReader
from monai.utils.enums import PostFix
DEFAULT_POST_FIX = PostFix.meta()

from monai.data import PersistentDataset

def get_data_dict(root_dir='./data1', modality='CT', num_train=35, num_val=19, random_seed=42, CT_by_MR=1):
    # CT_by_MR: the value of #CT divide #MR, 1 means 1:1 (default), 2 means 2:1, 3 means 3:1
    if modality == 'MR':
        data_dir_train = root_dir + '/data1_mr_train'
        data_dir_val = root_dir + '/data1_mr_val'
    elif modality == 'CT':
        num_train *= CT_by_MR
        num_val *= CT_by_MR 
        data_dir_train = root_dir + '/data1_ct_train'
        data_dir_val = root_dir + '/data1_ct_val'
    
    # for the train, validation
    data_lst = []
    for data_dir in [data_dir_train, data_dir_val]:
        img_list = sorted(list(filter(lambda x: ('gt' not in x and 'nii.gz' in x), os.listdir(data_dir)))) 
        lab_list = sorted(list(filter(lambda x: ('gt' in x and 'nii.gz' in x), os.listdir(data_dir))))   
        data_lst.append([{'image': os.path.join(data_dir,i), 'label': os.path.join(data_dir,j), 'modality': modality, 'name': i.split('/')[-1][:9]} for i, j in zip(img_list, lab_list)])
    
    random.seed(random_seed)
    data_lst_sampled = []
    for tmp_lst, num in zip(data_lst, [num_train, num_val]):
        if len(tmp_lst) <= num:
            data_lst_sampled.append(tmp_lst)
        else:
            data_lst_sampled.append(random.sample(tmp_lst, num))

    # to dict format for each sample
    return data_lst_sampled

def get_loader(root_dir='./data1', modality='CT', mix_batch=False, train_num_samples=2, num_train=35, num_val=19, random_seed = 42, cache_rate = 1.0, return_which='train', batch_size=1, persistent= True, cache_dir = "", CT_by_MR=1):
    
    
    ct_train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], ensure_channel_first=True, image_only=True), #0
            Spacingd(keys=["image", "label"], pixdim=(
                1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"], a_min=-125, a_max=275,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96), mode='constant'),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=train_num_samples,
                image_key="image",
                image_threshold=0,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.50,
            ),
            RandAffined(
                keys=['image', 'label'],
                mode=('bilinear', 'nearest'),
                prob=1.0, spatial_size=(96, 96, 96),
                rotate_range=(0, 0, np.pi / 30),
                scale_range=(0.1, 0.1, 0.1)),
            ToTensord(keys=["image", "label"]),
        ]
    )
        
    ct_val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], ensure_channel_first=True, image_only=True), #0
            Spacingd(keys=["image", "label"], pixdim=(
                1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"], a_min=-125, a_max=275,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96), mode='constant'),
            ToTensord(keys=["image", "label"]),
        ]
    )
        
    ## train_transforms for MRI image
    mr_train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], ensure_channel_first=True, image_only=True), #0
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96), mode='constant'),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96), #192, 192, 64
                pos=1,
                neg=1,
                num_samples=train_num_samples,
                image_key="image",
                image_threshold=0,
            ),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
            ToTensord(keys=["image", "label"]),
        ]
    )
        
    ## val_transforms for MRI image
    mr_val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], ensure_channel_first=True, image_only=True),
            Spacingd(keys=["image", "label"], pixdim=(
                1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96), mode='constant'),
            ToTensord(keys=["image", "label"]),
        ]
    )
    
    if modality == 'CT':
        train_transforms = ct_train_transforms
        val_transforms = ct_val_transforms
    elif modality == 'MR':
        train_transforms = mr_train_transforms
        val_transforms = mr_val_transforms
        
        
    data_dicts_train, data_dicts_val = get_data_dict(root_dir=root_dir, modality=modality, num_train=num_train, num_val=num_val, random_seed=random_seed, CT_by_MR=CT_by_MR)
    
    # 35 35 --> 45 25 --> 25 45 --> CT MR backbone unet textembedding V3
    
    if mix_batch == True:
        data_dicts_train_ct, data_dicts_val_ct = get_data_dict(root_dir=root_dir, modality='CT', num_train=num_train, num_val=num_val, random_seed=random_seed, CT_by_MR=CT_by_MR)
        data_dicts_train_mr, data_dicts_val_mr = get_data_dict(root_dir=root_dir, modality='MR', num_train=num_train, num_val=num_val, random_seed=random_seed, CT_by_MR=CT_by_MR)
        if return_which == 'train':
            if persistent:
                dataset_ct = PersistentDataset(data=data_dicts_train_ct, transform=ct_train_transforms, cache_dir=cache_dir)
                dataset_mr = PersistentDataset(data=data_dicts_train_mr, transform=mr_train_transforms, cache_dir=cache_dir)
            else:
                dataset_ct = CacheDataset(data=data_dicts_train_ct, transform=ct_train_transforms, cache_rate = cache_rate)
                dataset_mr = CacheDataset(data=data_dicts_train_mr, transform=mr_train_transforms, cache_rate = cache_rate)
            dataset = ConcatDataset([dataset_ct, dataset_mr])
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0,collate_fn=list_data_collate)
        elif return_which == 'val':
            if persistent:
                dataset_ct = PersistentDataset(data=data_dicts_val_ct, transform=ct_val_transforms, cache_dir=cache_dir)
                dataset_mr = PersistentDataset(data=data_dicts_val_mr, transform=mr_val_transforms, cache_dir=cache_dir)
            else:
                dataset_ct = CacheDataset(data=data_dicts_val_ct, transform=ct_val_transforms, cache_rate = cache_rate)
                dataset_mr = CacheDataset(data=data_dicts_val_mr, transform=mr_val_transforms, cache_rate = cache_rate)
            dataset = dataset_ct if modality == 'CT' else dataset_mr
            loader = DataLoader(dataset, batch_size=1, shuffle= False, num_workers=0, collate_fn=list_data_collate)
        return loader
        
    if return_which == 'train':
        if persistent:
            dataset = PersistentDataset(data=data_dicts_train, transform=train_transforms, cache_dir=cache_dir)
        else:
            dataset = CacheDataset(data=data_dicts_train, transform=train_transforms, cache_rate = cache_rate)
        
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0,collate_fn=list_data_collate)
    elif return_which == 'val':
        if persistent:
            dataset = PersistentDataset(data=data_dicts_val, transform=val_transforms, cache_dir=cache_dir)
        else:
            dataset = CacheDataset(data=data_dicts_val, transform=val_transforms, cache_rate = cache_rate)
        loader = DataLoader(dataset, batch_size=1, shuffle= False, num_workers=0, collate_fn=list_data_collate)
        
    return loader
    
def get_loader_without_gt(args):
    val_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            # ToTemplatelabeld(keys=['label']),
            # RL_Splitd(keys=['label']),
            Spacingd(
                keys=["image"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear"),
            ), # process h5 to here
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            CropForegroundd(keys=["image"], source_key="image"),
            ToTensord(keys=["image"]),
        ]
    )
    
    ## test dict part
    ## to be done

def get_loader_data1(train_modality='CT', mix_batch=False, phase='train', root_dir = '/home/cli6/Projects/MulModSeg/dataset/data1', persistent=True, cache_dir='cache_dir_new', CT_by_MR=1, num_train=35):
    # cache_dir = cache_dir + '_' + str(CT_by_MR)
    cache_dir = os.path.join(root_dir, cache_dir)
    
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    if mix_batch == True:
        if phase == 'train':
            return get_loader(root_dir, modality=train_modality, mix_batch=True, return_which='train', persistent= persistent, cache_dir=os.path.join(cache_dir, 'data1_mix_batch_train'), CT_by_MR=CT_by_MR)
        elif phase == 'val':
            if train_modality == 'CT':
                loader = get_loader(root_dir, modality=train_modality, mix_batch=True, return_which='val', persistent= persistent, cache_dir=os.path.join(cache_dir, 'data1_ct_mix_batch_val'), CT_by_MR=CT_by_MR)
            elif train_modality == 'MR':
                loader = get_loader(root_dir, modality=train_modality, mix_batch=True, return_which='val', persistent= persistent, cache_dir=os.path.join(cache_dir, 'data1_mr_mix_batch_val'), CT_by_MR=CT_by_MR)
            return loader
    
    if train_modality == 'CT':
        if phase == 'train':
            return get_loader(root_dir, modality=train_modality, num_train=num_train, return_which='train', persistent= persistent, cache_dir=os.path.join(cache_dir, 'data1_ct_train'), CT_by_MR=CT_by_MR)
        elif phase == 'val':
            return  get_loader(root_dir, modality=train_modality, num_train=num_train, return_which='val', persistent= persistent, cache_dir=os.path.join(cache_dir, 'data1_ct_val'), CT_by_MR=CT_by_MR) 
    elif train_modality == 'MR':
        if phase == 'train':
            return get_loader(root_dir, modality=train_modality, num_train=num_train, return_which='train', persistent= persistent, cache_dir=os.path.join(cache_dir, 'data1_mr_train'), CT_by_MR=CT_by_MR)
        elif phase == 'val':
            return  get_loader(root_dir, modality=train_modality, num_train=num_train, return_which='val', persistent= persistent, cache_dir=os.path.join(cache_dir, 'data1_mr_val'), CT_by_MR=CT_by_MR) 
    elif train_modality == 'MIX':
        if phase == 'train':
            loader_CT = get_loader(root_dir, modality='CT', num_train=num_train,  return_which='train', persistent= persistent, cache_dir=os.path.join(cache_dir, 'data1_ct_train'), CT_by_MR=CT_by_MR) 
            loader_MR =  get_loader(root_dir, modality='MR', num_train=num_train, return_which='train', persistent= persistent, cache_dir=os.path.join(cache_dir, 'data1_mr_train'), CT_by_MR=CT_by_MR)
        elif phase == 'val':
            loader_CT = get_loader(root_dir, modality='CT', num_train=num_train, return_which='val', persistent= persistent, cache_dir=os.path.join(cache_dir, 'data1_ct_val'), CT_by_MR=CT_by_MR) 
            loader_MR =  get_loader(root_dir, modality='MR', num_train=num_train, return_which='val', persistent= persistent, cache_dir=os.path.join(cache_dir, 'data1_mr_val'), CT_by_MR=CT_by_MR)
        return [loader_CT, loader_MR]

if __name__ == "__main__":
    # num_train = 35
    # num_val = 19
    # random_seed = 42
    # CT_by_MR = 1
    # root_dir = './dataset/data1'
    # modality = 'CT'
    # data_dicts_train, data_dicts_val = get_data_dict(root_dir=root_dir, modality=modality, num_train=num_train, num_val=num_val, random_seed=random_seed, CT_by_MR=CT_by_MR)

    print('shjssoai')
    import time 
    start = time.time()
    
    CT_by_MR=3
    loader = get_loader_data1(train_modality= 'MR', phase='val', CT_by_MR=CT_by_MR)
    for batch in loader:
        img, lab, mod, name = batch['image'], batch['label'], batch['modality'], batch['name']
        print(img.shape, lab.shape, mod, name )
    
    
    loader = get_loader_data1(train_modality= 'MR', phase='train', CT_by_MR=CT_by_MR)
    for batch in loader:
        img, lab, mod, name = batch['image'], batch['label'], batch['modality'], batch['name']
        print(img.shape, lab.shape, mod, name )
    
    loader = get_loader_data1(train_modality= 'CT', phase='val', CT_by_MR=CT_by_MR)
    for batch in loader:
        img, lab, mod, name = batch['image'], batch['label'], batch['modality'], batch['name']
        print(img.shape, lab.shape, mod, name )
        
    loader = get_loader_data1(train_modality= 'CT', phase='train', CT_by_MR=CT_by_MR)
    for batch in loader:
        img, lab, mod, name = batch['image'], batch['label'], batch['modality'], batch['name']
        print(img.shape, lab.shape, mod, name )
    
    end = time.time()
    print(f"time spend for loading: {end-start} s with persistant")


