import tensorflow as tf
from model import *
import os
import cv2
import argparse

import backtrace

backtrace.hook(
    reverse=False,
    align=True,
    strip_path=True,
    enable_on_envvar_only=False,
    on_tty=False,
    conservative=False,
    styles={})

def typeDir(str):
    if(not os.path.isdir(str)):
        raise argparse.ArgumentTypeError("{0} is not a directory.".format(str))
    return str


argParser = argparse.ArgumentParser(description='')
argParser.add_argument('-i', '--inputDir', type=typeDir, required=True, help="folder containing patchs")
argParser.add_argument('-n', '--nbClasses', type=int, required=True, help="number of classes")
args = argParser.parse_args()

images = []

for file in sorted(os.listdir(args.inputDir)):
        if file.endswith(".JPG"):
            im = cv2.imread(args.inputDir+"/"+file)
            images.append([im, file])


os.environ["CUDA_VISIBLE_DEVICES"]='1'


placeholder_shape = [None] + list(images[0][0].shape)
img_placeholder = tf.placeholder(tf.float32, placeholder_shape, name="img_placeholder")
net = tiny_model(img_placeholder, args.nbClasses)

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state("model")
    saver.restore(sess, "model/model.ckpt")

    for i in range(0, len(images)):
        # id of the random image
        im = images[i][0]
        
        res = sess.run(net, feed_dict={img_placeholder:[im]})
        print(images[i][1], res)
