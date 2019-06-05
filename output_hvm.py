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

import tensorflow as tf
from output_objectome import color_normalize, load_model_to_eval
from eval_linear import forward
from scipy.misc import imresize
import cPickle


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_tfr_files(data_path):
    # Get tfrecord files
    all_tfrs_path = os.listdir(data_path)
    all_tfrs_path = filter(lambda x:'tfrecords' in x, all_tfrs_path)
    all_tfrs_path.sort()
    all_tfrs_path = [os.path.join(data_path, each_tfr) \
            for each_tfr in all_tfrs_path]

    return all_tfrs_path


def get_parser():
    parser = argparse.ArgumentParser(
            description='Generating outputs for hvm dataset')

    parser.add_argument(
            '--data', type=str, 
            default='/data5/chengxuz/Dataset/neural_resp'\
                    + '/images/V4IT/tf_records/images',
            help='path to hvm dataset, images')
    parser.add_argument(
            '--model', type=str, 
            default='/home/chengxuz/deepcluster_models/vgg16/checkpoint.pth.tar',
            help='path to model')
    parser.add_argument(
            '--conv', type=str,
            default='5,7,9,11,13',
            help='Separated by ","')
    parser.add_argument(
            '--batch_size', default=32, type=int,
            help='mini-batch size')
    parser.add_argument(
            '--save_path', type=str,
            default='/data/chengxuz/v4it_temp_results/V4IT',
            help='path for storing results')
    parser.add_argument(
            '--seed', type=int, 
            default=31, help='random seed')

    return parser


def get_one_image(string_record):
    example = tf.train.Example()
    example.ParseFromString(string_record)
    img_string = (example.features.feature['images']
                                  .bytes_list
                                  .value[0])
    img_array = np.fromstring(img_string, dtype=np.float32)
    img_array = img_array.reshape([256, 256, 3])
    img_array *= 255
    img_array = img_array.astype(np.uint8)
    img_array = imresize(img_array, [224, 224, 3])
    img_array = color_normalize(img_array)
    img_array = np.transpose(img_array, [2, 0, 1])
    img_array = img_array.astype(np.float32)
    return img_array


def get_batches(all_records):
    all_images = []
    for string_record in all_records:
        all_images.append(get_one_image(string_record))
    all_images = np.stack(all_images, axis=0)
    return all_images


def get_all_images(tfr_path):
    record_iterator = tf.python_io.tf_record_iterator(path=tfr_path)
    all_records = list(record_iterator)
    num_imgs = len(all_records)
    all_images = get_batches(all_records)
    return num_imgs, all_images


class HvmOutput(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cpu")
        self.make_meta = True
        self.save_keys = ['conv%i' % each_conv for each_conv in args.conv]
        self.model = load_model_to_eval(args)

        for save_key in self.save_keys:
            curr_folder = os.path.join(args.save_path, save_key)
            os.system('mkdir -p %s' % curr_folder)

    def _get_outputs(self, curr_batch):
        args = self.args
        input_var = torch.autograd.Variable(
                torch.from_numpy(curr_batch).cuda())
        outputs = forward(input_var, self.model, args.conv)

        outputs = [
                np.asarray(output.float().to(self.device)) \
                for output in outputs]
        outputs = [np.transpose(output, [0, 2, 3, 1]) for output in outputs]
        return outputs

    def _make_meta(self, curr_outputs):
        args = self.args
        
        for save_key, curr_output in zip(self.save_keys, curr_outputs):
            curr_meta = {
                    save_key: {
                        'dtype': tf.string, 
                        'shape': (), 
                        'raw_shape': tuple(curr_output.shape[1:]),
                        'raw_dtype': tf.float32,
                        }
                    }
            meta_path = os.path.join(args.save_path, save_key, 'meta.pkl')
            cPickle.dump(curr_meta, open(meta_path, 'w'))
        self.make_meta = False

    def _make_writers(self, tfr_path):
        args = self.args
        
        all_writers = []
        for save_key in self.save_keys:
            write_path = os.path.join(
                    args.save_path, save_key, 
                    os.path.basename(tfr_path))
            writer = tf.python_io.TFRecordWriter(write_path)
            all_writers.append(writer)
        self.all_writers = all_writers

    def _write_outputs(self, curr_outputs):
        for writer, curr_output, save_key in \
                zip(self.all_writers, curr_outputs, self.save_keys):
            for idx in range(curr_output.shape[0]):
                curr_value = curr_output[idx]
                save_feature = {
                        save_key: _bytes_feature(curr_value.tostring())
                        }
                example = tf.train.Example(
                        features=tf.train.Features(feature=save_feature))
                writer.write(example.SerializeToString())

    def _close_writers(self):
        for each_writer in self.all_writers:
            each_writer.close()

    def write_outputs_for_one_tfr(self, tfr_path):
        args = self.args
        num_imgs, all_images = get_all_images(tfr_path)
        self._make_writers(tfr_path)

        for start_idx in range(0, num_imgs, args.batch_size):
            curr_batch = all_images[start_idx : start_idx + args.batch_size]
            curr_outputs = self._get_outputs(curr_batch)

            if self.make_meta:
                self._make_meta(curr_outputs)
            self._write_outputs(curr_outputs)
        self._close_writers()


def main():
    parser = get_parser()
    args = parser.parse_args()
    args.conv = [int(each_conv) for each_conv in args.conv.split(',')]
    all_tfr_path = get_tfr_files(args.data)

    hvm_output = HvmOutput(args)
    
    for tfr_path in tqdm(all_tfr_path):
        hvm_output.write_outputs_for_one_tfr(tfr_path)


if __name__ == '__main__':
    main()
