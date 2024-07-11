import argparse
import datetime
import json
import time
from contextlib import suppress
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.utils import ModelEma, NativeScaler, get_state_dict

import model.conv_segmenter
import model.mamba_segmenter
import utils as utils
from engine import evaluate, train_one_epoch
from model.utils import create_optimizer
from ref_dataset import build_dataset, collate_fn


def get_args_parser():
    parser = argparse.ArgumentParser('ReMamber training and evaluation script', add_help=False)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--epochs', default=50, type=int)

    # Model parameters
    parser.add_argument('--model', default='ReMamber_Conv', type=str, metavar='MODEL', choices=['ReMamber_Conv', 'ReMamber_Mamba'],
                        help='Name of model to train')
    parser.add_argument('--pretrain-path', default='./pretrain', type=str)
    parser.add_argument('--input-size', default=480, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=False)
    parser.add_argument('--model-ema-decay', type=float, default=0.9999, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='weight decay')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-5, metavar='LR',
                        help='learning rate')
    parser.add_argument('--lr-decoder', type=float, default=5e-5, metavar='LR',
                        help='learning rate')
    parser.add_argument('--lr-backbone', type=float, default=2.5e-5, metavar='LR',
                        help='learning rate')
    parser.add_argument('--lr-vssm', type=float, default=2.5e-5, metavar='LR',
                        help='learning rate')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    parser.add_argument('--train-mode', action='store_true')
    parser.add_argument('--no-train-mode', action='store_false', dest='train_mode')
    parser.set_defaults(train_mode=True)
    
    
    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    
    # Dataset parameters
    parser.add_argument('--data-path', default='./ref_dataset/data', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='refcoco', choices=['refcoco', 'refcoco+', 'refcocog'],
                        type=str)

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--distributed', action='store_true', default=False, help='Enabling distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    # amp about
    parser.add_argument('--if_amp', action='store_true')
    parser.add_argument('--no_amp', action='store_false', dest='if_amp')
    parser.set_defaults(if_amp=True)

    parser.add_argument('--debug_mode', action='store_true', default=False)

    parser.add_argument('--local-rank', default=0, type=int)
    return parser


def main(args):
    utils.init_distributed_mode(args)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    dataset_train = build_dataset(is_train=True, args=args)
    dataset_val = build_dataset(is_train=False, args=args)
    if args.debug_mode:
        dataset_train = torch.utils.data.Subset(dataset_train, list(range(1000)))
        dataset_val = torch.utils.data.Subset(dataset_val, list(range(300)))

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()

        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        collate_fn=collate_fn,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        collate_fn=collate_fn,
    )

    print(f"Creating model: {args.model}")
    model, new_param = create_model(
        args.model,
        img_size=args.input_size,
        model_size="base",
    )

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint['model']

        ret = model.load_state_dict(checkpoint_model, strict=False)
        print(ret[0], ret[1])
        
    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    optimizer = create_optimizer(args, model_without_ddp, new_param)
    
    # amp about
    amp_autocast = suppress
    loss_scaler = "none"
    if args.if_amp:
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint and args.if_amp: # change loss_scaler if not amp
                loss_scaler.load_state_dict(checkpoint['scaler'])
            elif 'scaler' in checkpoint and not args.if_amp:
                loss_scaler = 'none'
        lr_scheduler.step(args.start_epoch)
        
    if args.eval:
        test_stats = evaluate(data_loader_val, model, device, amp_autocast)
        return
    

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model=model, data_loader=data_loader_train,
            optimizer=optimizer, device=device, epoch=epoch, loss_scaler=loss_scaler, amp_autocast=amp_autocast,
            max_norm=args.clip_grad, model_ema=model_ema, 
            set_training_mode=args.train_mode,  # keep in eval mode for finetuning / train mode for training and finetuning
            # args = args, 
        )
        lr_scheduler.step(epoch)
        if args.output_dir:
            checkpoint_paths = [output_dir / f'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'epoch': epoch,
                }, checkpoint_path)

        test_stats = evaluate(data_loader_val, model, device, amp_autocast, log_every=50)
        print(f"IoU of the network on the {len(dataset_val)} test images: {test_stats['iou']:.1f}%")
        
        if max_accuracy < test_stats["iou"]:
            max_accuracy = test_stats["iou"]
            if args.output_dir:
                checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'model_ema': get_state_dict(model_ema) if model_ema is not None else None,
                        'scaler': loss_scaler.state_dict() if loss_scaler != 'none' else loss_scaler,
                        'args': args,
                    }, checkpoint_path)
            
        print(f'Max IoU: {max_accuracy:.2f}%')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        # test ema here
        if model_ema:
            test_stats_ema = evaluate(data_loader_val, model_ema.ema, device, amp_autocast, log_every=50)
            log_stats.update({f'test_ema_{k}': v for k, v in test_stats_ema.items()})
        
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
