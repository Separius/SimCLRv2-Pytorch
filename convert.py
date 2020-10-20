import argparse

import torch
import numpy as np
import torch.nn as nn
import tensorflow as tf

from resnet import get_resnet, name_to_params

parser = argparse.ArgumentParser(description='SimCLR converter')
parser.add_argument('tf_path', type=str, help='path of the input tensorflow file (ex: model.ckpt-250228)')
parser.add_argument('--ema', action='store_true')
parser.add_argument('--supervised', action='store_true')
args = parser.parse_args()


def main():
    use_ema_model = args.ema
    prefix = ('ema_model/' if use_ema_model else '') + 'base_model/'
    head_prefix = ('ema_model/' if use_ema_model else '') + 'head_contrastive/'
    # 1. read tensorflow weight into a python dict
    vars_list = []
    contrastive_vars = []
    for v in tf.train.list_variables(args.tf_path):
        if v[0].startswith(prefix) and not v[0].endswith('/Momentum'):
            vars_list.append(v[0])
        elif v[0] in {'head_supervised/linear_layer/dense/bias', 'head_supervised/linear_layer/dense/kernel'}:
            vars_list.append(v[0])
        elif v[0].startswith(head_prefix) and not v[0].endswith('/Momentum'):
            contrastive_vars.append(v[0])

    sd = {}
    ckpt_reader = tf.train.load_checkpoint(args.tf_path)
    for v in vars_list:
        sd[v] = ckpt_reader.get_tensor(v)

    split_idx = 2 if use_ema_model else 1
    # 2. convert the state_dict to PyTorch format
    conv_keys = [k for k in sd.keys() if k.split('/')[split_idx].split('_')[0] == 'conv2d']
    conv_idx = []
    for k in conv_keys:
        mid = k.split('/')[split_idx]
        if len(mid) == 6:
            conv_idx.append(0)
        else:
            conv_idx.append(int(mid[7:]))
    arg_idx = np.argsort(conv_idx)
    conv_keys = [conv_keys[idx] for idx in arg_idx]

    bn_keys = list(set([k.split('/')[split_idx] for k in sd.keys()
                        if k.split('/')[split_idx].split('_')[0] == 'batch']))
    bn_idx = []
    for k in bn_keys:
        if len(k.split('_')) == 2:
            bn_idx.append(0)
        else:
            bn_idx.append(int(k.split('_')[2]))
    arg_idx = np.argsort(bn_idx)
    bn_keys = [bn_keys[idx] for idx in arg_idx]

    depth, width, sk_ratio = name_to_params(args.tf_path)
    model, head = get_resnet(depth, width, sk_ratio)

    conv_op = []
    bn_op = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            conv_op.append(m)
        elif isinstance(m, nn.BatchNorm2d):
            bn_op.append(m)
    assert len(vars_list) == (len(conv_op) + len(bn_op) * 4 + 2)  # 2 for fc

    for i_conv in range(len(conv_keys)):
        m = conv_op[i_conv]
        w = torch.from_numpy(sd[conv_keys[i_conv]]).permute(3, 2, 0, 1)
        assert w.shape == m.weight.shape, f'size mismatch {w.shape} <> {m.weight.shape}'
        m.weight.data = w

    for i_bn in range(len(bn_keys)):
        m = bn_op[i_bn]
        gamma = torch.from_numpy(sd[prefix + bn_keys[i_bn] + '/gamma'])
        assert m.weight.shape == gamma.shape, f'size mismatch {gamma.shape} <> {m.weight.shape}'
        m.weight.data = gamma
        m.bias.data = torch.from_numpy(sd[prefix + bn_keys[i_bn] + '/beta'])
        m.running_mean = torch.from_numpy(sd[prefix + bn_keys[i_bn] + '/moving_mean'])
        m.running_var = torch.from_numpy(sd[prefix + bn_keys[i_bn] + '/moving_variance'])

    w = torch.from_numpy(sd['head_supervised/linear_layer/dense/kernel']).t()
    assert model.fc.weight.shape == w.shape
    model.fc.weight.data = w
    b = torch.from_numpy(sd['head_supervised/linear_layer/dense/bias'])
    assert model.fc.bias.shape == b.shape
    model.fc.bias.data = b

    if args.supervised:
        save_location = f'r{depth}_{width}x_sk{1 if sk_ratio != 0 else 0}{"_ema" if use_ema_model else ""}.pth'
        torch.save({'resnet': model.state_dict(), 'head': head.state_dict()}, save_location)
        return
    sd = {}
    for v in contrastive_vars:
        sd[v] = ckpt_reader.get_tensor(v)
    linear_op = []
    bn_op = []
    for m in head.modules():
        if isinstance(m, nn.Linear):
            linear_op.append(m)
        elif isinstance(m, nn.BatchNorm1d):
            bn_op.append(m)
    for i, (l, m) in enumerate(zip(linear_op, bn_op)):
        l.weight.data = torch.from_numpy(sd[f'{head_prefix}nl_{i}/dense/kernel']).t()
        common_prefix = f'{head_prefix}nl_{i}/batch_normalization/'
        m.weight.data = torch.from_numpy(sd[f'{common_prefix}gamma'])
        if i != 2:
            m.bias.data = torch.from_numpy(sd[f'{common_prefix}beta'])
        m.running_mean = torch.from_numpy(sd[f'{common_prefix}moving_mean'])
        m.running_var = torch.from_numpy(sd[f'{common_prefix}moving_variance'])

    # 3. dump the PyTorch weights.
    save_location = f'r{depth}_{width}x_sk{1 if sk_ratio != 0 else 0}{"_ema" if use_ema_model else ""}.pth'
    torch.save({'resnet': model.state_dict(), 'head': head.state_dict()}, save_location)


if __name__ == '__main__':
    main()
