import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse
import time

import warnings
warnings.filterwarnings("ignore")

from tensorboardX import SummaryWriter

from monai.losses import DiceCELoss
from monai.data import load_decathlon_datalist, decollate_batch
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric
from monai.networks.nets import UNet

from model.MulModSeg import MulModSeg, UNet3D_cy, SwinUNETR_cy
from dataset.dataloader_amos import get_loader_amos

from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from monai.losses import DiceCELoss
from monai.transforms import AsDiscrete
from monai.inferers import sliding_window_inference_cy
from itertools import cycle


# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# torch.multiprocessing.set_sharing_strategy('file_system')

def train_mix(args, train_loader_ct, train_loader_mr, model, optimizer, loss_seg_DICE, loss_seg_CE):
    model.train()
    loss_bce_ave = 0
    loss_dice_ave = 0
    _len = min(len(train_loader_ct), len(train_loader_mr))
    
    for step, (batch_ct, batch_mr) in enumerate(zip(train_loader_ct, train_loader_mr)):
        for batch in [batch_ct, batch_mr]:
            x, y, z, name = batch["image"].to(args.device), batch["label"].float().to(args.device), batch['modality'], batch['name']
            if args.with_text_embedding == 1:
                logit_map = model(x, z[0])
            else:
                logit_map = model(x)

            term_seg_Dice = loss_seg_DICE.forward(logit_map, y)
            term_seg_BCE = loss_seg_CE.forward(logit_map, y)
            loss = term_seg_Dice + term_seg_BCE
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if step % 10 == 0:
                print(
                    "Epoch=%d: Training (%d / %d Steps) (dice_loss=%2.5f, bce_loss=%2.5f)" % (
                        args.epoch, step, _len, term_seg_Dice.item(), term_seg_BCE.item())
                )
            loss_bce_ave += term_seg_BCE.item()
            loss_dice_ave += term_seg_Dice.item()
            torch.cuda.empty_cache()
    print('Epoch=%d: ave_dice_loss=%2.5f, ave_bce_loss=%2.5f' % (args.epoch, loss_dice_ave/_len, loss_bce_ave/_len))
    
    return loss_dice_ave/_len, loss_bce_ave/_len

def train_mix_withloop(args, train_loader_ct, train_loader_mr, model, optimizer, loss_seg_DICE, loss_seg_CE):
    model.train()
    max_iterations = max(len(train_loader_ct), len(train_loader_mr))  # Use the length of the longer loader
    loss_bce_ave = 0
    loss_dice_ave = 0
    _len = min(len(train_loader_ct), len(train_loader_mr))
    
    CTL_cycle = cycle(train_loader_ct)
    MRL_cycle = cycle(train_loader_mr)
    
    for step in range(max_iterations):
        batch_ct = next(CTL_cycle)
        batch_mr = next(MRL_cycle)
 
    
    # for step, (batch_ct, batch_mr) in enumerate(zip(train_loader_ct, train_loader_mr)):
        for batch in [batch_ct, batch_mr]:
            x, y, z, name = batch["image"].to(args.device), batch["label"].float().to(args.device), batch['modality'], batch['name']
            if args.with_text_embedding == 1:
                logit_map = model(x, z[0])
            else:
                logit_map = model(x)

            term_seg_Dice = loss_seg_DICE.forward(logit_map, y)
            term_seg_BCE = loss_seg_CE.forward(logit_map, y)
            loss = term_seg_Dice + term_seg_BCE
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if step % 100 == 0:
                print(
                    "Epoch=%d: Training (%d / %d Steps) (dice_loss=%2.5f, bce_loss=%2.5f)" % (
                        args.epoch, step, _len, term_seg_Dice.item(), term_seg_BCE.item())
                )
            loss_bce_ave += term_seg_BCE.item()
            loss_dice_ave += term_seg_Dice.item()
            torch.cuda.empty_cache()
    print('Epoch=%d: ave_dice_loss=%2.5f, ave_bce_loss=%2.5f' % (args.epoch, loss_dice_ave/_len, loss_bce_ave/_len))
    
    return loss_dice_ave/_len, loss_bce_ave/_len


def train(args, train_loader, model, optimizer, loss_seg_DICE, loss_seg_CE):
    model.train()
    loss_bce_ave = 0
    loss_dice_ave = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        # x, y, name = batch["image"].to(args.device), batch["post_label"].float().to(args.device), batch['name']
        x, y, z, name = batch["image"].to(args.device), batch["label"].float().to(args.device), batch['modality'], batch['name']
        if args.with_text_embedding == 1:
            logit_map = model(x, z[0])
        else:
            logit_map = model(x)
        
        term_seg_Dice = loss_seg_DICE.forward(logit_map, y)
        term_seg_BCE = loss_seg_CE.forward(logit_map, y)
        loss = term_seg_Dice + term_seg_BCE
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_iterator.set_description(
            "Epoch=%d: Training (%d / %d Steps) (dice_loss=%2.5f, bce_loss=%2.5f)" % (
                args.epoch, step, len(train_loader), term_seg_Dice.item(), term_seg_BCE.item())
        )
        loss_bce_ave += term_seg_BCE.item()
        loss_dice_ave += term_seg_Dice.item()
        torch.cuda.empty_cache()
    print('Epoch=%d: ave_dice_loss=%2.5f, ave_bce_loss=%2.5f' % (args.epoch, loss_dice_ave/len(epoch_iterator), loss_bce_ave/len(epoch_iterator)))
    
    return loss_dice_ave/len(epoch_iterator), loss_bce_ave/len(epoch_iterator)

# def validation(model, val_loader, args, modality='CT', overlap=0.5):
#     model.eval()
#     dice_list = {}
    
#     post_label = AsDiscrete(to_onehot=args.num_class)
#     post_pred = AsDiscrete(argmax=True, to_onehot=args.num_class, threshold=0.5)

#     for index, batch in enumerate(tqdm(val_loader)):
#         # print('%d processd' % (index))
#         image, label, name = batch["image"].cuda(), batch["post_label"], batch["name"]
#         print(name, image.shape)
#         with torch.no_grad():
#             pred = sliding_window_inference_cy(image, (args.roi_x, args.roi_y, args.roi_z), 1, model, 
#                                             modality=modality, overlap=overlap, mode='gaussian')
#             pred_sigmoid = F.sigmoid(pred)
#             pred_final = post_pred(pred_sigmoid)
        
#         B = pred_sigmoid.shape[0]
            


def process(args):
    args.device = torch.device(f"cuda:{args.device}") # assign the device as default device
    torch.cuda.set_device(f"{args.device}")  # assign the device as default device for cuda

    # prepare the 3D model
    if args.with_text_embedding == 1:
        model = MulModSeg(img_size=(args.roi_x, args.roi_y, args.roi_z),
                    in_channels=1,
                    out_channels=args.num_class,
                    backbone=args.backbone,
                    encoding=args.trans_encoding
                    )
    else:
        if args.backbone == 'unet':
            model = UNet3D_cy(
            out_channels=args.num_class
            )
        elif args.backbone == 'swinunetr':
            model = SwinUNETR_cy(
                img_size=(args.roi_x, args.roi_y, args.roi_z),
                out_channels=args.num_class
            )

    #Load pre-trained weights
    if args.pretrain is not None:
        model.load_params(torch.load(args.pretrain)["state_dict"])

    if args.with_text_embedding == 1 and args.trans_encoding == 'word_embedding':
        word_embedding = torch.load(args.word_embedding)
        model.organ_embedding.data = word_embedding.float()
        print('load word embedding')

    model.to(args.device)
    model.train()
    
    # criterion and optimizer
    loss_seg_DICE = DiceCELoss(to_onehot_y=True, softmax=True, lambda_dice=1.0, lambda_ce=0.0)
    loss_seg_CE = DiceCELoss(to_onehot_y=True, softmax=True, lambda_dice=0.0, lambda_ce=1.0)
    if args.backbone == 'unetpp':
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, 
                                    nesterov=False, weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epoch, max_epochs=args.max_epoch)

    if args.resume:
        checkpoint = torch.load(args.resume)
        store_dict = model.state_dict()
        model_dict = checkpoint['net']
        model.load_state_dict(store_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.epoch = checkpoint['epoch']
        scheduler.load_state_dict(checkpoint['scheduler'])
        
        print('success resume from ', args.resume)

    torch.backends.cudnn.benchmark = True

    # train loader
    # train_loader, train_sampler = get_loader_amos(modality='CT', phase='train')
    if args.train_modality == 'MIX':
        train_loader_ct = get_loader_amos(train_modality='CT', phase='train', persistent=True)
        train_loader_mr = get_loader_amos(train_modality='MR', phase='train', persistent=True)
    else:
        train_loader = get_loader_amos(train_modality=args.train_modality, phase='train', persistent=True)
    

    if args.rank == 0:
        if args.with_text_embedding == 1:
            str_tmp = 'out/'+args.backbone+'/with_txt/CLIP_V3/' + args.log_name + '_' + args.train_modality + f'_lr{args.lr}' + f'_max_epoch{args.max_epoch}' + time.strftime('_%m_%d_%H_%M', time.localtime())
            writer = SummaryWriter(log_dir=str_tmp)
            print('Writing Tensorboard logs to ', str_tmp)
        else:
            str_tmp = 'out/'+args.backbone+'/no_txt/' + args.log_name + '_' + args.train_modality + f'_lr{args.lr}' + f'_max_epoch{args.max_epoch}' + time.strftime('_%m_%d_%H_%M', time.localtime())
            writer = SummaryWriter(log_dir=str_tmp)
            print('Writing Tensorboard logs to ', str_tmp)
    
    # print('end')
    # return 0

    while args.epoch < args.max_epoch:
        scheduler.step()
        if args.train_modality == 'MIX':
            loss_dice, loss_bce = train_mix(args, train_loader_ct, train_loader_mr, model, optimizer, loss_seg_DICE, loss_seg_CE)
        else:
            loss_dice, loss_bce = train(args, train_loader, model, optimizer, loss_seg_DICE, loss_seg_CE)

        if args.rank == 0:
            writer.add_scalar('train_dice_loss', loss_dice, args.epoch)
            writer.add_scalar('train_bce_loss', loss_bce, args.epoch)
            writer.add_scalar('total_loss', loss_bce + loss_dice, args.epoch)
            writer.add_scalar('lr', scheduler.get_lr(), args.epoch)

        if (args.epoch % args.store_num == 0 and args.epoch != 0) and args.rank == 0 or (args.max_epoch - args.epoch <= 5):
            checkpoint = {
                "net": model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                "epoch": args.epoch
            }
            if not os.path.isdir(str_tmp):
                os.mkdir(str_tmp)
            torch.save(checkpoint, str_tmp + '/epoch_' + str(args.epoch) + '.pt')
            print('save model success')

        args.epoch += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=0, choices=[0, 1, 2, 3, 4, 5, 6, 7], type=int, help="GPU device")
    parser.add_argument("--epoch", default=0)
    ## logging
    parser.add_argument('--log_name', default='unet', help='The path resume from checkpoint')
    parser.add_argument('--rank', default=0, type=int, help='use tensorboardX to log the training process')
    ## model load
    parser.add_argument('--backbone', default='unet', help='backbone [swinunetr or unet or dints or unetpp]')
    parser.add_argument('--resume', default=None, help='The path resume from checkpoint')
    parser.add_argument('--pretrain', default=None,  #swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt
                        help='The path of pretrain model. Eg, ./pretrained_weights/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt')
    parser.add_argument('--trans_encoding', default='word_embedding', 
                        help='the type of encoding: rand_embedding or word_embedding')
    parser.add_argument('--word_embedding', default='./pretrained_weights/CLIP_V3_mod_cls_txt_encoding.pth', 
                        help='The path of word embedding')
    parser.add_argument('--with_text_embedding', default=1, type=int, choices=[0, 1], help='whether use text embedding')
    ## hyperparameter
    parser.add_argument('--max_epoch', default=1000, type=int, help='Number of training epoches')
    parser.add_argument('--store_num', default=5, type=int, help='Store model how often')
    parser.add_argument('--warmup_epoch', default=10, type=int, help='number of warmup epochs')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-5, help='Weight Decay')
    ## dataset
    ### please check this argment carefully
    parser.add_argument('--data_root_path', default='./dataset/data1', help='data root path')
    parser.add_argument('--batch_size', default=1, help='batch size')
    parser.add_argument('--num_workers', default=8, type=int, help='workers numebr for DataLoader')
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
    parser.add_argument('--num_samples', default=2, type=int, help='sample number in each ct/mri scan')
    parser.add_argument('--num_class', default=14, type=int, help='the number of class for the segmentation')
    parser.add_argument('--phase', default='train', help='train or val or test')
    
    ## training mode
    parser.add_argument('--train_modality', default='MIX', type=str, choices=['CT', 'MR', 'MIX'], help='CT or MR or MIX')

    args = parser.parse_args()
    
    process(args=args)

if __name__ == "__main__":
    main()