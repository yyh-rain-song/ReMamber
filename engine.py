# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from timm.utils import ModelEma

import utils


def trainMetricGPU(output, target, threshold=0.5):
    assert (output.dim() in [2, 3, 4])
    assert output.shape == target.shape
    output = output.flatten(1)
    target = target.flatten(1)
    output = torch.sigmoid(output)
    output[output < threshold] = 0.
    output[output >= threshold] = 1.
    # inter & union
    inter = (output.bool() & target.bool()).sum(dim=1)  # b
    union = (output.bool() | target.bool()).sum(dim=1)  # b
    ious = inter / (union + 1e-6)  # 0 ~ 1

    iou = ious.mean()
    return iou

def train_one_epoch(model: torch.nn.Module, 
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, amp_autocast, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, 
                    set_training_mode=True, args = None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
        
    for batch in metric_logger.log_every(data_loader, print_freq, header):
        imgs = batch['query_img'].to(device, non_blocking=True)
        masks = batch['query_mask'].to(device, non_blocking=True)
        sents = batch['sentence']
         
        with amp_autocast():
            pred, _, loss = model(imgs, sents, masks.float(), epoch=epoch)

        loss_value = loss.item()
        iou_train = trainMetricGPU(pred, masks)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        if loss_scaler != 'none':
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        else:
            loss.backward()
            if max_norm != None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(train_loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(train_iou=iou_train)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, amp_autocast, log_every=10):

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, log_every, header):
        imgs = batch['query_img'].to(device, non_blocking=True)
        masks = batch['query_mask'].to(device, non_blocking=True)
        sents = batch['sentence']
    
        unfold_img = []
        unfold_mask = []
        unfold_sent = []
        gt_mask_list = []
        for batch_idx, single_sent in enumerate(sents):
            new_bs = len(single_sent)
            img = imgs[batch_idx].unsqueeze(0)
            img = img.repeat(new_bs, 1, 1, 1)
            unfold_img.append(img)
            
            mask = masks[batch_idx].unsqueeze(0)
            mask = mask.repeat(new_bs, 1, 1, 1)
            unfold_mask.append(mask)
            
            gt_mask = batch['org_gt'][batch_idx]
            for _ in range(new_bs):
                gt_mask_list.append(gt_mask)
            
            unfold_sent.extend(single_sent)
            
        unfold_img = torch.cat(unfold_img, dim=0)
        unfold_mask = torch.cat(unfold_mask, dim=0)
        pred = model(unfold_img, unfold_sent, unfold_mask.float())
        
        for single_pred, gt_mask in zip(pred, gt_mask_list):
            single_pred = F.interpolate(single_pred[None,], gt_mask.shape)[0]
            single_pred = single_pred.sigmoid()
            single_pred = single_pred.cpu().numpy()
            single_pred = single_pred > 0.5
            inter = np.logical_and(single_pred, gt_mask).sum()
            union = np.logical_or(single_pred, gt_mask).sum()
            iou = inter / (union  + 1e-6)
            metric_logger.meters['inter'].update(inter)
            metric_logger.meters['union'].update(union)
            metric_logger.meters['iou'].update(iou)
            
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    IOU = metric_logger.iou.global_avg
    OIOU = metric_logger.inter.global_avg / (metric_logger.union.global_avg + 1e-6)
    print('* IoU {iou:.3f} oIoU {oiou:.3f}'.format(iou=IOU, oiou=OIOU))

    return {
        'iou': IOU,
        'oiou': OIOU,
    }

