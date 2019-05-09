import tensorflow as tf
from model import *
import os
import cv2
from tensorflow.python.framework import graph_util
import argparse

def typeDir(str):
    if(not os.path.isdir(str)):
        raise argparse.ArgumentTypeError("{0} is not a directory.".format(str))
    return str


argParser = argparse.ArgumentParser(description='Exports model to .pb and .pbtxt files')
argParser.add_argument('-i', '--inputDir', type=typeDir, required=True, help="folder containing checkpoint files")
argParser.add_argument('-n', '--nbClasses', type=int, required=True, help="number of output classes of the network")
argParser.add_argument('-s', '--imSize', type=int, required=True, help="size of the images")
args = argParser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]='1'

img_placeholder = tf.placeholder(tf.float32, [None, args.imSize, args.imSize, 3], name='img')
net = tiny_model(img_placeholder, args.nbClasses) # don't forget to set the appropriate number of classes

# Restore from checkpoint and calc the features from all of train data
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state("model")
    path = args.inputDir

    while(path[len(path)-1] == '/'):
        path = path[:-1]
    
    saver.restore(sess, path+"/model.ckpt")

    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    output_graph_def = graph_util.convert_variables_to_constants(
        sess,  # The session is used to retrieve the weights
        input_graph_def,  # The graph_def is used to retrieve the nodes
        ["tiny_model/output/output/Softmax"]  # The output node names are used to select the usefull nodes
    )
    # Finally we serialize and dump the output graph to the filesystem
    with tf.gfile.GFile(str(args.inputDir)+"/model.pb", "wb") as f:
        f.write(output_graph_def.SerializeToString())
        tf.train.write_graph(sess.graph_def, '.', str(args.inputDir)+'/model.pbtxt')

    print("Wrote model.pb and model.pbtxt !")
