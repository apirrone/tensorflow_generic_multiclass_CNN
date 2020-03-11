import tensorflow as tf
from model import *
import os
import cv2
import argparse
import numpy as np
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
argParser.add_argument('-m', '--modelDir', type=typeDir, required=True, help="folder containing model")
argParser.add_argument('-n', '--nbClasses', type=int, required=True, help="number of classes")
argParser.add_argument('--imWidth', type=int, required=False, help="images width", default=128)
argParser.add_argument('--imHeight', type=int, required=False, help="images height", default=128)
args = argParser.parse_args()

# classes = ["class1", "class2", "class3"] # any number of classes
classes = ["demotic", "greek", "coptic"] # any number of classes
# classes = ["arabic", "greek", "coptic"] # any number of classes
classes = sorted(classes)

images = []

for file in sorted(os.listdir(args.inputDir)):
        if file.endswith(".png"):
            im = cv2.imread(args.inputDir+"/"+file)
            im = cv2.resize(im, (args.imWidth, args.imHeight))
            im = cv2.normalize(im, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)        
            images.append([im, file])


os.environ["CUDA_VISIBLE_DEVICES"]='2'


placeholder_shape = [None] + list(images[0][0].shape)
img_placeholder = tf.placeholder(tf.float32, placeholder_shape, name="img_placeholder")
net = tiny_model(img_placeholder, args.nbClasses)

results = {}

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state("model")
    saver.restore(sess, args.modelDir+"model.ckpt")

    
    print("Processing...")
    for i in range(0, len(images)):
        # id of the random image
        im = images[i][0]
        
        res = sess.run(net, feed_dict={img_placeholder:[im]})

        res = classes[np.argmax(res)]
        if res not in results:
            results[res] = 0

        results[res] += 1
        # print(images[i][1], res)

sum = 0
for key, value in results.items():
    sum += value

print("")
print("PROBABILITIES : ")
    
for key, value in results.items():
    percent = round(results[key]/sum*100, 2)

    print(key, str(percent)+" %")
    
