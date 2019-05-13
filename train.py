import tensorflow as tf
from tensorflow.python.framework import graph_util
import numpy as np
import os
from dataset import *
from model import *
import cv2
import utils
import datetime

now = datetime.datetime.now()

os.environ["CUDA_VISIBLE_DEVICES"]='1'

batch_size = 64
nb_epochs = 100
image_size = [128, 128]
learning_rate = 0.001
train_proportion = 0.8
classes = ["class1", "class2", "class3"] # any number of classes
model = tiny_model
gray_scale = False

model_folder = "model_"+now.strftime("%Y-%m-%d_%H:%M")+"/"
os.system("mkdir "+str(model_folder))
os.system("ln -sf "+str(model_folder)+" lastModel")
logs_dir = str(model_folder)+"/logs/"
data_path = "data/"


if __name__ == "__main__":

        dataset = Dataset(data_path, image_size, train_proportion, classes, gray_scale)        
        nbBatchsInEpoch = dataset.buildBatches(batch_size)

        modelInfos = open(model_folder+"/infos", "a")                                
        modelInfos.write("Model name : tiny_model_1layer2fc\n")
        modelInfos.write("Batch size : "+str(batch_size)+"\n")
        modelInfos.write("Image size : "+str(image_size)+"\n")
        modelInfos.write("Learning rate : "+str(learning_rate)+"\n\n")
        modelInfos.write("Data used : "+str(data_path)+"\n\n")
        
        modelInfos.write(str(len(classes))+" Classes : "+str(classes)+"\n\n")
        for key, value in dataset.pathsByLabel.items():
                modelInfos.write("Number of "+key+" : "+str(len(value))+"\n")
        modelInfos.close()
        

        placeholder_shape = [None] + list(dataset.imShape)
        print("placeholder_shape", placeholder_shape)

        y_true = tf.to_float(tf.placeholder(tf.int32, shape=[None, len(dataset.labels_map)], name="y_true"))
        img_placeholder = tf.placeholder(tf.float32, placeholder_shape, name="img_placeholder")
        y_pred = model(img_placeholder, len(dataset.labels_map))

        # Loss functions
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred))
        sigmoid_cross_entropy_with_logits = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred))
        
        loss = cross_entropy

        # Create a summary to monitor cost tensor
        tf.summary.scalar("loss", loss)
        # Merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()
        
        # Optimizers
        # train_step = tf.train.MomentumOptimizer(learningRate, 0.99, use_nesterov=False).minimize(loss)
        # train_step = tf.train.RMSPropOptimizer(learningRate).minimize(loss)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        currentEpoch = 0
        
        # Start Training
        saver = tf.train.Saver()
        with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                
                # tensorboard stuff
                writer = tf.summary.FileWriter(logs_dir, sess.graph)

                testAcc = []
                trainAcc = []
                losses = []
                
                batch_test_images, batch_test_labels = dataset.getValidationSet()
                
                i = 0
                while(currentEpoch < nb_epochs):

                        currentBatch, batch_train_images, batch_train_labels = dataset.getBatch()
                        
                        _, l, summary = sess.run([train_step, loss, merged_summary_op], feed_dict={img_placeholder:batch_train_images, y_true: batch_train_labels})
                        writer.add_summary(summary, i)

                        losses.append(l)

                        print("Epoch : "+str(currentEpoch)+", i : "+str(i)+", training loss   : "+str(round(l, 4))+", smoothed loss : "+str(round(np.mean(losses[-1000:]), 4)))

                        # End of an epoch
                        if(currentBatch >= nbBatchsInEpoch-1):
                                nbBatchsInEpoch = dataset.buildBatches(batch_size)
                                currentEpoch+=1

                                test_correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(batch_test_labels, 1))
                                test_accuracy = tf.reduce_mean(tf.cast(test_correct_prediction, tf.float32))
                                test_acc = sess.run(test_accuracy, feed_dict={img_placeholder: batch_test_images, y_true: batch_test_labels})
                                testAcc.append(test_acc)
                                print("test accuracy : "+str(round(test_acc, 4))+" smoothed test accuracy : "+str(round(np.mean(testAcc[-10:]), 4)))
                                

                                train_correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(batch_train_labels, 1))
                                train_accuracy = tf.reduce_mean(tf.cast(train_correct_prediction, tf.float32))
                                train_acc = sess.run(train_accuracy, feed_dict={img_placeholder: batch_train_images, y_true: batch_train_labels})
                                trainAcc.append(train_acc)
                                print("train accuracy : "+str(round(train_acc, 4))+" smoothed train accuracy : "+str(round(np.mean(trainAcc[-10:]), 4)))

                                
                                
                                sample = sess.run(y_pred, feed_dict={img_placeholder: batch_test_images})
                                print("=============================================")
                                print("sample :")

                                print(sample[0])
                                print(batch_test_labels[0])
                                print("=============================================")
                                
                                print("")
                                print("Confusion Matrix (cols : prediction, rows : label)")
                                con_mat = tf.confusion_matrix(tf.argmax(batch_test_labels, 1), tf.argmax(sample, 1))
                                confusion_matrix = tf.Tensor.eval(con_mat,feed_dict=None, session=None).astype(np.float64)
                                
                                print("")
                                for j in range(0, len(dataset.inverted_labels_map)):
                                        print(str(dataset.inverted_labels_map[j])+" ", end=' ')
                                
                                print("")
                                print(confusion_matrix.astype(np.int32))
                                normal_confusion_matrix = confusion_matrix.astype(np.int32)
                                confusion_sum = np.sum(confusion_matrix)
                                for a in range(0, len(confusion_matrix[0])):
                                        for b in range(0, len(confusion_matrix[0])):
                                                confusion_matrix[a][b] = round(confusion_matrix[a][b]/confusion_sum, 2)
                                                
                                print("")
                                for j in range(0, len(dataset.inverted_labels_map)):
                                        print(str(dataset.inverted_labels_map[j])+" ", end=' ')
                                        
                                print("")
                                print(confusion_matrix)
                                print("=============================================")

                                
                                print("========= Saving...=========")
                                epoch_folder = str(model_folder)+"/epoch_"+str(currentEpoch)
                                saver.save(sess, epoch_folder+"/model.ckpt")
                                os.system("touch "+epoch_folder+"/infos")

                                utils.visualize_false_predictions(epoch_folder,
                                                                  batch_test_images,
                                                                  batch_test_labels,
                                                                  sample,
                                                                  dataset.image_size,
                                                                  dataset.inverted_labels_map)
                                
                                epochInfos = open(epoch_folder+"/infos", "a")
                                epochInfos.write("Epoch : "+str(currentEpoch)+"/"+str(nb_epochs)+"\n")


                                epochInfos.write("training Loss : "+str(round(np.mean(losses[-1000:]), 4))+"\n")
                                epochInfos.write("train accuracy : "+str(round(trainAcc[len(trainAcc)-1], 4))+"\n")
                                epochInfos.write("test accuracy : "+str(round(testAcc[len(testAcc)-1], 4))+"\n\n")
                                
                                epochInfos.write("Confusion matrix : \n")
                                for j in range(0, len(dataset.inverted_labels_map)):
                                        epochInfos.write(str(dataset.inverted_labels_map[j])+" ")
                                epochInfos.write("\n")
                                epochInfos.write(str(normal_confusion_matrix)+"\n\n")
                                
                                epochInfos.write("Normalized confusion matrix : \n")
                                for j in range(0, len(dataset.inverted_labels_map)):
                                        epochInfos.write(str(dataset.inverted_labels_map[j])+" ")
                                epochInfos.write("\n")
                                epochInfos.write(str(confusion_matrix)+"\n\n")
                                                                
                                epochInfos.close()

                                # exporting .pb file
                                print("========= Exporting .pb and .pbtxt...=========")
                                graph = tf.get_default_graph()
                                input_graph_def = graph.as_graph_def()

                                output_graph_def = graph_util.convert_variables_to_constants(
                                        sess,
                                        input_graph_def,
                                        ["tiny_model/output/output/Softmax"])  # WARNING change this if you change the model
                                
                                with tf.gfile.GFile(str(epoch_folder)+"/model.pb", "wb") as f:
                                        f.write(output_graph_def.SerializeToString())
                                        tf.train.write_graph(sess.graph_def, '.', str(epoch_folder)+'/model.pbtxt')
                                
                                        
                                
                        i+=1





