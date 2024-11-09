import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse
import time
import glob

from monai.losses import DiceCELoss
from monai.data import load_decathlon_datalist, decollate_batch
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric
from monai.inferers.utils import sliding_window_inference_cy, sliding_window_inference

from model.MulModSeg import MulModSeg, UNet3D_cy, SwinUNETR_cy
from dataset.dataloader_amos import get_loader_amos

# for 
import nibabel as nib
import numpy as np
import pandas as pd
from collections import defaultdict
from utils.utils import dice, resample_3d
from utils.utils import calculate_distance, calculate_metric_percase, calculate_dice_re
from monai.metrics import compute_hausdorff_distance, compute_average_surface_distance
from monai.inferers import sliding_window_inference_cy

from monai.metrics import DiceMetric
dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)


def count_parameters(model, logger=None):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if logger:
        logger.info(f"Total: {pytorch_total_params}, {pytorch_total_params/1e6:.1f} M")
        logger.info(f"Total trainable: {pytorch_total_params_trainable}, {pytorch_total_params_trainable/1e6:.1f} M")
    else:
        print(f"Total: {pytorch_total_params}, {pytorch_total_params/1e6:.1f} M")
        print(f"Total trainable: {pytorch_total_params_trainable}, {pytorch_total_params_trainable/1e6:.1f} M")
    return [pytorch_total_params, pytorch_total_params_trainable]

def test(args):
    # prepare the 3D model
    args.device = torch.device(f"cuda:{args.device}") # assign the device as default device
    torch.cuda.set_device(f"{args.device}")
    if args.with_text_embedding == 1:
        model = MulModSeg(img_size=(args.roi_x, args.roi_y, args.roi_z),
                    in_channels=1,
                    out_channels=args.num_class,
                    backbone=args.backbone,
                    )
    else:
        if args.backbone == 'unet':
            model = UNet3D_cy(
            out_channels=args.num_class
            )
        elif args.backbone == 'swinunetr':
            model = SwinUNETR_cy(
                out_channels=args.num_class
            )
    #Load pre-trained weights
    store_path_root = f'{args.log_folder}/epoch_{args.test_epoch}.pt'
    print(f'which model to be tested: {store_path_root}')
    
    model = model.to(args.device)
    
    store_path = store_path_root
    store_dict = model.state_dict()
    load_dict = torch.load(store_path)['net']

    for key, value in load_dict.items():
        if key == 'organ_embedding':
            if args.test_mod == 'CT':
                # store_dict[key] = value[0] #default
                store_dict[key] = value[1]
            elif args.test_mod == 'MR':
                # store_dict[key] = value[1] #default   
                store_dict[key] = value[0]                                              
        else:
            store_dict[key] = value
    model.load_state_dict(store_dict)
    
    print(args.backbone)
    print(f'Load {store_path} weights')
    
    output_directory = args.log_folder + f'/output_nii_epoch{args.test_epoch}_{args.test_mod}'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    df_dict = defaultdict(list)
    post_label = AsDiscrete(to_onehot=args.num_class)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.num_class)
    spacing = (args.space_x, args.space_y, args.space_z)
    test_loader = get_loader_amos(train_modality=args.test_mod, phase='val', persistent=True)
    
    start_time = time.time()
    with torch.no_grad():
        dice_list_case = []
        for i, batch in enumerate(test_loader):
            # break
            img_name = batch['name'][0] + ".nii.gz"
            # if not (img_name =="amos_0070.nii.gz"):
            #     continue
            
            df_dict['img_name'].append(img_name)
            print("Inference on case {}".format(img_name))
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            # original_affine = batch["label_meta_dict"]["affine"][0].numpy()
            
            # with torch.cuda.amp.autocast():
            if args.with_text_embedding == 1:
                val_outputs = sliding_window_inference_cy(val_inputs, (args.roi_x, args.roi_y, args.roi_z), 4, model, modality=args.test_mod, overlap=args.overlap, mode='gaussian')
            else:
                val_outputs = sliding_window_inference(val_inputs, (args.roi_x, args.roi_y, args.roi_z), 4, model, overlap=args.overlap, mode='gaussian')
            # val_outputs = net(val_inputs)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [
                post_label(val_label_tensor) for val_label_tensor in val_labels_list
            ]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [
                post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]
            
            # not including the background
            per_cls_dice = dice_metric(y_pred=val_output_convert[0].unsqueeze(0), y=val_labels_convert[0].unsqueeze(0))
            per_cls_dice = list(per_cls_dice[0].cpu().numpy())
            
            # print(f"per class dice: {per_cls_dice}")
            for j in range(1, args.num_class):
                tmp_dice = per_cls_dice[j-1]
                df_dict[f'Dice_{j}'].append(tmp_dice)
            print(f"per class mean_dice: {np.mean(per_cls_dice)}")     
    end_time = time.time()
    count_parameters(model) # print the number of parameters for this model
    print(f"Time for testing: {(end_time - start_time)/len(test_loader)} per case")
    
    df = pd.DataFrame.from_dict(dict(df_dict))
    print(df.describe())
    descriptive_stats = df.describe()
    combined_df = pd.concat([df, descriptive_stats])
    combined_df.to_csv(output_directory+'/results.csv', index=True)    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=7, type=int)
    parser.add_argument('--backbone', default='unet', help='backbone [swinunetr or unet or dints or unetpp]')
    parser.add_argument("--log_folder", default='/home/cli6/Projects/Multi_Modality_Seg_v3/out/UNet3D/with_txt/CLIP_V3/unet_newT_MIX_lr0.001_max_epoch1000_02_22_15_51')
    parser.add_argument("--test_epoch", default=10, type=int)
    parser.add_argument('--a_min', default=-175, type=float, help='a_min in ScaleIntensityRanged')
    parser.add_argument('--a_max', default=250, type=float, help='a_max in ScaleIntensityRanged')
    parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
    parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
    parser.add_argument('--space_x', default=1.5, type=float, help='spacing in x direction')
    parser.add_argument('--space_y', default=1.5, type=float, help='spacing in y direction')
    parser.add_argument('--space_z', default=2.0, type=float, help='spacing in z direction')
    parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
    parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
    parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')
    parser.add_argument('--num_class', default=14, type=int, help='the number of class for the segmentation')
    parser.add_argument('--test_mod', default='CT', choices=['CT', 'MR'], help='the modality you want to test')
    parser.add_argument('--with_text_embedding', default=1, type=int, choices=[0, 1], help='whether use text embedding')


                        
    parser.add_argument('--store_result', action="store_true", default=False, help='whether save prediction result')
    parser.add_argument('--overlap', default=0.5, type=float, help='overlap for sliding_window_inference')

    args = parser.parse_args()
    
    test(args)
    

if __name__ == '__main__':
    main()
    print('pass')