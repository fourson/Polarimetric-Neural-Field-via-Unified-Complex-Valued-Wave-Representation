import os
from pathlib import Path
import argparse
import importlib
import sys
import json

import torch
from tqdm import tqdm
import numpy as np


def infer_default():
    G = (data_loader.img_size[0] * data_loader.img_size[1]) // data_loader.num_points

    with torch.no_grad():
        # batch size is always 1
        for batch_idx, sample in enumerate(tqdm(data_loader, ascii=True)):
            # get name and configure path
            scene_name = sample['name'][0]
            pred_dir = Path(result_dir) / scene_name
            pred_dir.mkdir(parents=True, exist_ok=True)

            # get data
            flattened_xy = sample['flattened_xy'].to(device)  # (1 G num_points 2), real
            flattened_S0 = sample['flattened_S0'].to(device) / 2  # (1 G num_points 3), real
            flattened_k = sample['flattened_k'].to(device)  # (1 G num_points 3), complex
            flattened_theta = sample['flattened_theta'].to(device) / np.pi  # (1 G num_points 3), real

            # temporarily store the flattened predictions for rendering the full results
            flattened_S0_pred = torch.zeros_like(flattened_S0)  # (1 G num_points 3), real
            flattened_k_pred = torch.zeros_like(flattened_k)  # (1 G num_points 3), complex
            flattened_theta_pred = torch.zeros_like(flattened_theta)  # (1 G num_points 3), real

            # iterate each group of points
            for i in range(0, G):
                xy_group = flattened_xy[:, i:i + 1, :, :]  # (1 1 num_points 2), real

                # infer
                S0_group_pred, k_group_pred, theta_group_pred = model(xy_group)  # (1 1 num_points 3)

                # store the results
                flattened_S0_pred[:, i:i + 1, :, :] = S0_group_pred
                flattened_k_pred[:, i:i + 1, :, :] = k_group_pred
                flattened_theta_pred[:, i:i + 1, :, :] = theta_group_pred

            # unflatten the flattened predictions
            S0_pred = flattened_S0_pred.permute(0, 3, 1, 2).reshape(
                1, 3, data_loader.img_size[0], data_loader.img_size[1])  # (1 3 H W), real
            k_pred = flattened_k_pred.permute(0, 3, 1, 2).reshape(
                1, 3, data_loader.img_size[0], data_loader.img_size[1])  # (1 3 H W), complex
            theta_pred = flattened_theta_pred.permute(0, 3, 1, 2).reshape(
                1, 3, data_loader.img_size[0], data_loader.img_size[1])  # (1 3 H W), real

            # calculate p_pred
            p_pred = k_pred.abs()

            # get gt data and save
            S0 = sample['S0'].to(device)  # (1 3 H W), real
            p = sample['p'].to(device)  # (1 3 H W), real
            theta = sample['theta'].to(device)  # (1 3 H W), real
            S0_numpy = np.transpose(S0.squeeze().cpu().numpy(), (1, 2, 0))
            np.save(os.path.join(pred_dir, 'S0.npy'), S0_numpy)
            p_numpy = np.transpose(p.squeeze().cpu().numpy(), (1, 2, 0))
            np.save(os.path.join(pred_dir, 'p.npy'), p_numpy)
            theta_numpy = np.transpose(theta.squeeze().cpu().numpy(), (1, 2, 0))
            np.save(os.path.join(pred_dir, 'theta.npy'), theta_numpy)

            # normalize the pred data and save
            S0_pred = S0_pred * 2
            theta_pred = theta_pred * np.pi
            S0_pred_numpy = np.transpose(S0_pred.squeeze().cpu().numpy(), (1, 2, 0))
            np.save(os.path.join(pred_dir, 'S0_pred.npy'), S0_pred_numpy)
            p_pred_numpy = np.transpose(p_pred.squeeze().cpu().numpy(), (1, 2, 0))
            np.save(os.path.join(pred_dir, 'p_pred.npy'), p_pred_numpy)
            theta_pred_numpy = np.transpose(theta_pred.squeeze().cpu().numpy(), (1, 2, 0))
            np.save(os.path.join(pred_dir, 'theta_pred.npy'), theta_pred_numpy)

            # compute metrics (avg_channel_pol=False) and save as .json, then visualize both gt and pred
            avg_channel_pol = False
            S0_md = util.compute_metrics_for_tensor(S0_pred, S0, input_type='S0')
            p_md = util.compute_metrics_for_tensor(p_pred, p, input_type='p', avg_channel_pol=avg_channel_pol)
            theta_md = util.compute_metrics_for_tensor(theta_pred, theta, input_type='theta',
                                                       avg_channel_pol=avg_channel_pol)
            md = {**S0_md, **p_md, **theta_md}
            print(f'metrics: {md}')
            with open(os.path.join(pred_dir, 'metrics.json'), "w", encoding="utf-8") as file:
                json.dump(md, file, ensure_ascii=False, indent=4)
            util.visualize_array(os.path.join(pred_dir, 'S0.png'), S0_numpy, input_type='S0')
            util.visualize_array(os.path.join(pred_dir, 'p.png'), p_numpy, input_type='p',
                                 avg_channel_pol=avg_channel_pol)
            util.visualize_array(os.path.join(pred_dir, 'theta.png'), theta_numpy, input_type='theta',
                                 avg_channel_pol=avg_channel_pol)
            util.visualize_array(os.path.join(pred_dir, 'S0_pred.png'), S0_pred_numpy, input_type='S0')
            util.visualize_array(os.path.join(pred_dir, 'p_pred.png'), p_pred_numpy, input_type='p',
                                 avg_channel_pol=avg_channel_pol)
            util.visualize_array(os.path.join(pred_dir, 'theta_pred.png'), theta_pred_numpy, input_type='theta',
                                 avg_channel_pol=avg_channel_pol)

            # compute metrics (avg_channel_pol=True) and save as .json, then visualize both gt and pred
            avg_channel_pol = True
            p_md = util.compute_metrics_for_tensor(p_pred, p, input_type='p', avg_channel_pol=avg_channel_pol)
            theta_md = util.compute_metrics_for_tensor(theta_pred, theta, input_type='theta',
                                                       avg_channel_pol=avg_channel_pol)
            md = {**p_md, **theta_md}
            print(f'metrics_avg_channel_pol: {md}')
            with open(os.path.join(pred_dir, 'metrics_avg_channel_pol.json'), "w", encoding="utf-8") as file:
                json.dump(md, file, ensure_ascii=False, indent=4)
            util.visualize_array(os.path.join(pred_dir, 'p_avg_channel_pol.png'), p_numpy, input_type='p',
                                 avg_channel_pol=avg_channel_pol)
            util.visualize_array(os.path.join(pred_dir, 'theta_avg_channel_pol.png'), theta_numpy,
                                 input_type='theta', avg_channel_pol=avg_channel_pol)
            util.visualize_array(os.path.join(pred_dir, 'p_pred_avg_channel_pol.png'), p_pred_numpy, input_type='p',
                                 avg_channel_pol=avg_channel_pol)
            util.visualize_array(os.path.join(pred_dir, 'theta_pred_avg_channel_pol.png'), theta_pred_numpy,
                                 input_type='theta', avg_channel_pol=avg_channel_pol)

            # visualize error map
            util.visualize_error_map_for_array(os.path.join(pred_dir, 'error_S0.png'), S0_pred_numpy, S0_numpy,
                                               input_type='S0')
            util.visualize_error_map_for_array(os.path.join(pred_dir, 'error_p.png'), p_pred_numpy, p_numpy,
                                               input_type='p')
            util.visualize_error_map_for_array(os.path.join(pred_dir, 'error_theta.png'), theta_pred_numpy, theta_numpy,
                                               input_type='theta')


if __name__ == '__main__':
    MODULE = 'full'
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume', required=True, type=str, help='path to latest checkpoint')
    parser.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    parser.add_argument('--data_dir', required=True, type=str, help='dir of input data')
    parser.add_argument('--H', default=1024, type=int, help='height of input data')
    parser.add_argument('--W', default=1024, type=int, help='width of input data')
    parser.add_argument('--num_points', default=262144, type=int, help='number of points')
    parser.add_argument('--result_dir', required=True, type=str, help='dir to save result')
    parser.add_argument('--data_loader_type', default='PCONDataLoader', type=str, help='which data loader to use')
    subparsers = parser.add_subparsers(help='which func to run', dest='func')

    # add subparsers and their args for each func
    subparser_default = subparsers.add_parser("default")

    args = parser.parse_args()

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add project root to PATH
    from utils import util

    # load checkpoint
    checkpoint = torch.load(args.resume)
    config = checkpoint['config']
    assert config['module'] == MODULE

    # setup data_loader instances
    # we choose batch_size=1(default value)
    module_data = importlib.import_module('.data_loader_' + MODULE, package='data_loader')  # share the same dataloader
    data_loader_class = getattr(module_data, args.data_loader_type)
    pCON_convention = config['data_loader']['args'].get('pCON_convention')
    if pCON_convention:
        data_loader = data_loader_class(data_dir=args.data_dir, scene_name=config['data_loader']['args']['scene_name'],
                                        img_size=(args.H, args.W), num_points=args.num_points, pCON_convention=True)
    else:
        data_loader = data_loader_class(data_dir=args.data_dir, scene_name=config['data_loader']['args']['scene_name'],
                                        img_size=(args.H, args.W), num_points=args.num_points)

    module_arch = importlib.import_module('.model_' + MODULE, package='model')
    model_class = getattr(module_arch, config['model']['type'])
    model = model_class(**config['model']['args'])

    # prepare model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.load_state_dict(checkpoint['model'])

    # set the model to validation mode
    model.eval()

    # ensure result_dir
    result_dir = args.result_dir
    util.ensure_dir(result_dir)

    # run the selected func
    if args.func == 'default':
        infer_default()
    else:
        infer_default()
