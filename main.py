#-*- coding:utf-8 -*-
#Author:Tunan
#Creation Data:2020/10/16

import os
import scipy.misc
import numpy as np
import json

from model import DCGAN
from utils import pp, visualize, to_json, expand_path, show_all_variable, timestamp
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for ada [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("train_size", np.inf, "The siez of train images [np.inf]")
flags.DEFINE_integer("batch_size", 2, "The size of batch image [64]")
flags.DEFINE_integer("input_height", 108, "The size of image to use(will be center cropped).[108]")
flags.DEFINE_integer("input_width", None, "The size of image to use(will be center cropped). If Nonw, same value as input_height [None]")
flags.DEFINE_integer("output_height", 64, "The size of the output images to produce[64]")
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("data_dir", "./data", "path to datasets [e.g. $Home/data]")  #####   ?????  "/home/S319080106/anaconda3/envs/tensorflow-yyl/example/dcganexam/data/mnist/train-images.idx3-ubyte"
flags.DEFINE_string("output_dir", "./out", "Root directory for outputs [e.g. $Home/out]")
flags.DEFINE_string("out_name", "", "Folder(under out_root_dir) for all outputs.Generated automatically if left blank[]")
flags.DEFINE_string("checkpoint_dir","checkpint", "Folder (under out_root_dir/out_name) to save checkpoint[checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Folder (under out_root_dir/out_name) to save samples [samples]")
flags.DEFINE_boolean("train",False, "True for training, Fasle for testing [False]")
flags.DEFINE_boolean("crop", False, "True for training, Fasle for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visulizing, False for nothing [False]")
flags.DEFINE_boolean("export", False, "True for exporting with new batch size")
flags.DEFINE_boolean("freeze", False, "True for exporting with new batch size")
flags.DEFINE_integer("max_to_keep", 1, "maximum number of checkpoints to keep")
flags.DEFINE_integer("sample_freq", 200, "sample every this many iterations")
flags.DEFINE_integer("ckpt", 200, "save checkpoint every this many iterations")
flags.DEFINE_integer("z_dim", 100, "dimensions of z")
flags.DEFINE_string("z_dist", "uniform_signed", "'normal101' or 'uniform_unsigned' or uniform_signed")
flags.DEFINE_boolean("G_img_sum", False, "Save generator images summaries in log")
FLAGS = flags.FLAGS

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    FLAGS.data_dir = expand_path(FLAGS.data_dir)
    FLAGS.output_dir = expand_path(FLAGS.output_dir)
    FLAGS.out_name = expand_path(FLAGS.out_name)
    FLAGS.checkpoint_dir = expand_path(FLAGS.checkpoint_dir)
    FLAGS.sample_dir = expand_path(FLAGS.sample_dir)

    if FLAGS.output_height is None: FLAGS.output_height = FLAGS.input_height
    if FLAGS.input_width is None: FLAGS.input_width = FLAGS.input_height
    if FLAGS.output_width is None: FLAGS.output_width = FLAGS.output_height

    if FLAGS.out_name == "":
        FLAGS.out_name = '{} - {} - {}'.format(timestamp(), FLAGS.data_dir.split('/')[-1], FLAGS.dataset)
        if FLAGS.train:
            FLAGS.out_name += ' - x{}.z{}.{}.y{}.b{}'.format(FLAGS.input_width, FLAGS.z_dim, FLAGS.z_dist, FLAGS.output_width, FLAGS.batch_size)

    FLAGS.output_dir = os.path.join(FLAGS.output_dir, FLAGS.out_name)
    FLAGS.checkpoint_dir = os.path.join(FLAGS.output_dir, FLAGS.checkpoint_dir)
    FLAGS.sample_dir = os.path.join(FLAGS.output_dir, FLAGS.sample_dir)

    if not os.path.exists(FLAGS.checkpoint_dir): os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir): os.makedirs(FLAGS.sample_dir)

    with open(os.path.join(FLAGS.output_dir, 'FLAGS.json'), 'w') as f:
        flags_dict = {k: FLAGS[k].value for k in FLAGS}
        json.dump(flags_dict, f, indent=4, sort_keys=True, ensure_ascii=False)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        if FLAGS.dataset == 'mnist':
            dcgan = DCGAN(sess,
                          input_width=FLAGS.input_width,
                          input_height=FLAGS.input_height,
                          output_width=FLAGS.output_width,
                          output_height=FLAGS.output_height,
                          batch_size=FLAGS.batch_size,
                          sample_num=FLAGS.batch_size,
                          y_dim=10,
                          z_dim=FLAGS.z_dim,
                          dataset_name=FLAGS.dataset,
                          input_fname_pattern=FLAGS.input_fname_pattern,
                          crop=FLAGS.crop,
                          checkpoint_dir=FLAGS.checkpoint_dir,
                          sample_dir=FLAGS.sample_dir,
                          data_dir=FLAGS.data_dir,
                          output_dir=FLAGS.output_dir,
                          max_to_keep=FLAGS.max_to_keep
                          )
        else:
            dcgan = DCGAN(sess,
                          input_width=FLAGS.input_width,
                          input_height=FLAGS.input_height,
                          output_width=FLAGS.output_width,
                          output_height=FLAGS.output_height,
                          batch_size=FLAGS.batch_size,
                          sample_num=FLAGS.batch_size,
                          z_dim=FLAGS.z_dim,
                          dataset_name=FLAGS.dataset,
                          input_fname_pattern=FLAGS.input_fname_pattern,
                          crop=FLAGS.crop,
                          checkpoint_dir=FLAGS.checkpoint_dir,
                          sample_dir=FLAGS.sample_dir,
                          data_dir=FLAGS.data_dir,
                          output_dir=FLAGS.output_dir,
                          max_to_keep=FLAGS.max_to_keep
                          )

        show_all_variable()

        if FLAGS.train:
            dcgan.train(FLAGS)
        else:
            load_success, load_counter = dcgan.load(FLAGS.checkpoint_dir)
            if not load_success:
                raise Exception("checkpoint not foun in " + FLAGS.checkpoint_dir)

            # visualization
            if FLAGS.export:
                export_dir = os.path.join(FLAGS.checkpoint_dir, 'export_b'+str(FLAGS.batch_size))
                dcgan.save(export_dir, load_counter, ckpt=True, frozen=False)

            if FLAGS.freeze:
                export_dir = os.path.join(FLAGS.checkpoint_dir, 'frozen_b' + str(FLAGS.batch_size))
                dcgan.save(export_dir, load_counter,ckpt=False, frozen=True)

            if FLAGS.visualize:
                OPTION = 1
                visualize(sess, dcgan,FLAGS, OPTION, FLAGS.sample_dir)


if __name__ == '__main__':
    tf.app.run()






