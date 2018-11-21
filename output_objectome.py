import argparse
import os
import time
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from util import load_model
import pandas as pd
from PIL import Image
import h5py
from tqdm import tqdm


def color_normalize(image):
    imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = image*1.0 / 255
    image = (image - imagenet_mean) / imagenet_std
    return image


class Objectome(object):
    """
    Load from unzipped folder
    """
    DATA_LEN = 2400

    def __init__(self, data_path, **kwargs):
        csv_path = os.path.join(data_path, 'stimulus_set.csv')
        stimuli = pd.read_csv(csv_path)
        image_paths = [
                os.path.join(data_path, image_id + '.png') \
                for image_id in stimuli['image_id']
                ]
        image_paths = filter(lambda x: os.path.exists(x), image_paths)
        self.image_paths = image_paths
        self.now_idx = 0

    def get_one_image(self):
        image_path = self.image_paths[self.now_idx]
        im_frame = Image.open(image_path)
        np_frame = np.array(im_frame, dtype=np.float32)
        np_frame = color_normalize(np_frame)
        np_frame = np.transpose(np_frame, [2, 0, 1])
        self.now_idx += 1
        self.now_idx %= self.DATA_LEN
        return np_frame

    def get_next_batch(self, batch_size):
        all_images = []
        for _ in range(batch_size):
            all_images.append(self.get_one_image())
        return np.stack(all_images, axis=0)


def get_parser():
    parser = argparse.ArgumentParser(
            description='Generating outputs for objectome dataset')

    parser.add_argument(
            '--data', type=str, 
            default='/home/chengxuz/objectome',
            help='path to objectome dataset')
    parser.add_argument(
            '--model', type=str, 
            default='/home/chengxuz/deepcluster_models/vgg16/checkpoint.pth.tar',
            help='path to model')
    parser.add_argument(
            '--conv', type=str,
            default='5,7,9,11,13',
            help='Separated by ","')
    parser.add_argument(
            '--batch_size', default=40, type=int,
            help='mini-batch size')
    parser.add_argument(
            '--save_path', type=str,
            default='/mnt/fs4/chengxuz/v4it_temp_results/dc_vgg13_obj.hdf5',
            help='hdf5 path for results')
    parser.add_argument('--seed', type=int, default=31, help='random seed')

    return parser


def forward(x, model, conv):
    if hasattr(model, 'sobel') and model.sobel is not None:
        x = model.sobel(x)
    count = 1
    outputs = []
    for m in model.features.modules():
        if not isinstance(m, nn.Sequential):
            x = m(x)
        if isinstance(m, nn.ReLU):
            if count in conv:
                outputs.append(x)
            count = count + 1
    return outputs


def load_model_to_eval(args):
    #fix random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # load model
    model = load_model(args.model)
    model.cuda()
    cudnn.benchmark = True

    # freeze the features layers
    for param in model.features.parameters():
        param.requires_grad = False
    model.eval()
    return model


def main():
    parser = get_parser()
    args = parser.parse_args()
    args.conv = [int(each_conv) for each_conv in args.conv.split(',')]
    save_keys = ['conv_%i' % each_conv for each_conv in args.conv]

    model = load_model_to_eval(args)

    objectome_data = Objectome(args.data)
    assert objectome_data.DATA_LEN % args.batch_size == 0, \
            "Batch size should be divisible for DATA_LEN (2400)"
    num_batches = objectome_data.DATA_LEN / args.batch_size

    device = torch.device("cpu")
    fout = h5py.File(args.save_path, 'w')
    for curr_batch_idx in tqdm(range(num_batches)):
        curr_batch = objectome_data.get_next_batch(args.batch_size)
        input_var = torch.autograd.Variable(torch.from_numpy(curr_batch).cuda())
        outputs = forward(input_var, model, args.conv)

        outputs = [
                output.float().to(device) \
                for output in outputs]
        outputs = [
                np.transpose(output, [0, 2, 3, 1]) \
                for output in outputs]
        for curr_key, curr_output in zip(save_keys, outputs):
            if curr_key not in fout:
                new_shape = list(curr_output.shape)
                new_shape[0] = objectome_data.DATA_LEN
                dataset_tmp = fout.create_dataset(
                        curr_key, new_shape, dtype='f')
            else:
                dataset_tmp = fout[curr_key]
            now_num = curr_batch_idx * args.batch_size
            end_num = (curr_batch_idx+1) * args.batch_size
            dataset_tmp[now_num : end_num] = curr_output
    fout.close()


if __name__ == '__main__':
    main()
