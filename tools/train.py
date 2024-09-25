import argparse
import copy
import os
import os.path as osp
import time
import mmcv
import torch
import sys
sys.path.append('D:\Document\End_project_2023\hrnet\Lite-HRNet')
# from mmpose.apis.train import train_
from mmcv import Config, DictAction
from mmcv.runner import init_dist, set_random_seed
from mmcv.utils import get_git_hash
from mmpose import __version__
from mmpose.apis.train import train_model
from mmpose.datasets import build_dataset
from mmpose.utils import collect_env, get_root_logger
from models.builder import build_posenet


# Parse the command-line arguments for configuration and options
def parse_args():
    parser = argparse.ArgumentParser(description='Train a pose model')
    parser.add_argument('config', help='train config file path')  # Config file path
    parser.add_argument('--work-dir', help='the dir to save logs and models')  # Directory to save logs and models
    parser.add_argument('--resume-from',
                        help='the checkpoint file to resume from')  # Path to checkpoint to resume training
    parser.add_argument('--load-from',
                        help='the checkpoint file to load for fine-tuning')  # Path to checkpoint for fine-tuning
    parser.add_argument('--no-validate', action='store_true',
                        help='whether not to evaluate the checkpoint during training')  # Flag to disable validation during training
    group_gpus = parser.add_mutually_exclusive_group()  # GPU-related options
    group_gpus.add_argument('--gpus', type=int, help='number of gpus to use')  # Number of GPUs
    group_gpus.add_argument('--gpu-ids', type=int, nargs='+', help='ids of gpus to use')  # Specific GPU IDs to use
    parser.add_argument('--seed', type=int, default=None, help='random seed')  # Seed for random number generation
    parser.add_argument('--deterministic', action='store_true',
                        help='whether to set deterministic options for CUDNN backend.')  # Flag for deterministic behavior
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, default={},
                        help='override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file.')  # Override config options
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none',
                        help='job launcher')  # Distributed job launcher
    parser.add_argument('--local_rank', type=int, default=0)  # Local rank for distributed training
    parser.add_argument('--autoscale-lr', action='store_true',
                        help='automatically scale lr with the number of gpus')  # Auto scale learning rate
    args = parser.parse_args()

    # Set the local rank environment variable if it's not already set
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # Load the configuration file
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)  # Merge additional config options from command line

    # Enable CUDNN benchmarking if specified in the config
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # Set the working directory for saving logs and models
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    # Set paths to resume or load from checkpoints
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.load_from is not None:
        cfg.load_from = args.load_from

    # Set GPU IDs or the number of GPUs to use
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # Automatically scale learning rate based on the number of GPUs if specified
    if args.autoscale_lr:
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # Initialize distributed training if required
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # Create the directory for saving logs and models if it doesn't exist
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    # Initialize logging
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # Collect and log environment information
    meta = dict()
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    meta['env_info'] = env_info

    # Log distributed training status and the config
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # Set random seed and configure deterministic behavior if needed
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed

    # Build the model and dataset for training
    model = build_posenet(cfg.model)
    datasets = [build_dataset(cfg.data.train)]

    # If validation is part of the workflow, prepare the validation dataset
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))

    # Set meta information for checkpoint configuration
    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(
            mmpose_version=__version__ + get_git_hash(digits=7),
            config=cfg.pretty_text,
        )

    # Start training the model
    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()