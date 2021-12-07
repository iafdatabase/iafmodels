import imlib as im
import numpy as np
import pylib as py
import tensorflow as tf
import tf2lib as tl
import cv2 
import data
import module
import os

# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--experiment_dir')
py.arg('--batch_size', type=int, default=32)
test_args = py.args()
args = py.args_from_yaml(py.join(test_args.experiment_dir, 'settings.yml'))
args.__dict__.update(test_args.__dict__)


# ==============================================================================
# =                                    test                                    =
# ==============================================================================

# data
A_img_paths_test = py.glob(py.join(args.datasets_dir, args.dataset, 'testA'), '*.png')

A_dataset_test = data.make_dataset(A_img_paths_test, args.batch_size, args.load_size, args.crop_size,
                                   training=False, drop_remainder=False, shuffle=False, repeat=1)

# model
G_A2B = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, 3))
G_B2A = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, 3))

# resotre
tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(args.experiment_dir, 'checkpoints')).restore()


@tf.function
def sample_A2B(A):
    A2B = G_A2B(A, training=False)
    A2B2A = G_B2A(A2B, training=False)
    return A2B, A2B2A


# run
save_dir = py.join(args.experiment_dir, 'samples_testing', 'A2B')
py.mkdir(save_dir)
i = 0
for A in A_dataset_test:
    A2B, A2B2A = sample_A2B(A)
    for A_i, A2B_i, A2B2A_i in zip(A, A2B, A2B2A):
        img = A2B_i.numpy()
        print(py.name_ext(A_img_paths_test[i]))
        name = py.name_ext(A_img_paths_test[i])
        name = name.replace(".png", ".jpg")
        im.imwrite(img, save_dir+"/"+name)
        i += 1

