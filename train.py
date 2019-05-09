import tensorflow as tf
from tensorflow.python.framework import graph_util
import numpy as np
import os
from dataset import *
from model import *
import cv2

os.environ["CUDA_VISIBLE_DEVICES"]='1'

batch_size = 32
train_iter = 5000
step = 50
margin = 0.5
nb_epochs = 100
image_size = 256
model_folder = "model/"
learning_rate = 0.01
dataPath = "data/testDemoticGreek/"
train_proportion = 0.8

classes = ["demotic", "greek"]#, "coptic"]

if __name__ == "__main__":
        
        dataset = Dataset(dataPath, image_size, train_proportion, classes)        
        nbBatchsInEpoch = dataset.buildBatches(batch_size)
        
        model = tiny_model
        print(dataset.imShape)
        placeholder_shape = [None] + list(dataset.imShape)
        print("placeholder_shape", placeholder_shape)

        y_true = tf.to_float(tf.placeholder(tf.int32, shape=[None, len(dataset.labels_map)], name="y_true"))
        img_placeholder = tf.placeholder(tf.float32, placeholder_shape, name="img_placeholder")
        y_pred = model(img_placeholder, len(dataset.labels_map))

        # Loss functions
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred))
        sigmoid_cross_entropy_with_logits = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred))
        
        loss = cross_entropy
        
        # Optimizers
        # train_step = tf.train.MomentumOptimizer(learningRate, 0.99, use_nesterov=False).minimize(loss)
        # train_step = tf.train.RMSPropOptimizer(learningRate).minimize(loss)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        currentEpoch = 0
        
        # Start Training
        saver = tf.train.Saver()
        with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                testAcc = []
                trainAcc = []
                losses = []
                
                batch_test_images, batch_test_labels = dataset.getValidation()
                
                #train iter
                i = 0
                while(currentEpoch < nb_epochs):

                        currentBatch, batch_train_images, batch_train_labels = dataset.getBatch()
                        
                        _, l = sess.run([train_step, loss], feed_dict={img_placeholder:batch_train_images, y_true: batch_train_labels})

                        losses.append(l)

                        print("Epoch : "+str(currentEpoch)+", i : "+str(i)+", training loss   : "+str(round(l, 4))+", mega smoothed loss : "+str(round(np.mean(losses[-1000:]), 4)))
                        
                        if(currentBatch >= nbBatchsInEpoch-1):
                                nbBatchsInEpoch = dataset.buildBatches(batch_size)
                                currentEpoch+=1

                                correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(batch_test_labels, 1))
                                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                                tmp_acc = sess.run(accuracy, feed_dict={img_placeholder: batch_test_images, y_true: batch_test_labels})
                                
                                testAcc.append(tmp_acc)
                                print("test accuracy : "+str(round(tmp_acc, 4))+" smoothed test accuracy : "+str(round(np.mean(testAcc[-10:]), 4)))
                                
                                correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(batch_train_labels, 1))
                                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                                tmp_acc = sess.run(accuracy, feed_dict={img_placeholder: batch_train_images, y_true: batch_train_labels})
                                
                                trainAcc.append(tmp_acc)
                                print("train accuracy : "+str(round(tmp_acc, 4))+" smoothed train accuracy : "+str(round(np.mean(trainAcc[-10:]), 4)))

                                sample = sess.run(y_pred, feed_dict={img_placeholder: batch_test_images})
                                
                                print("=============================================")
                                print("sample :")

                                print(sample[0])
                                print(batch_test_labels[0])
                                
                                print("=============================================")
                                imageIndex = 0
                                os.system("rm -rf visualization_"+model_folder+"/")
                                os.system("mkdir visualization_"+model_folder+"/")
                                for b in range(0, len(sample)):
                                        # visual evaluation
                                        tmpIm = batch_test_images[b]
                                        true = batch_test_labels[b]
                                        pred = sample[b]
                                        if (np.argmax(true) != np.argmax(pred)):
                                                blank_image = np.zeros((dataset.image_size*2, dataset.image_size*2, 3))
                                                blank_image[:, :, 0] = 255
                                                blank_image[:, :, 1] = 255
                                                blank_image[:, :, 2] = 255
                                                blank_image[0:dataset.image_size, 0:dataset.image_size, :] = (tmpIm*255).astype(np.uint8)

                                                str_true = "true : "+dataset.inverted_labels_map[np.argmax(true)]
                                                str_pred = "pred : "+dataset.inverted_labels_map[np.argmax(pred)]
                                        
                                                cv2.putText(blank_image, str_true, (0, dataset.image_size+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
                                                cv2.putText(blank_image, str_pred, (0, dataset.image_size+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
                                                cv2.imwrite("visualization_"+model_folder+"/false_prediction_"+str(imageIndex)+".png", blank_image.astype(np.uint8))
                                                
                                                imageIndex +=1
                                
                                
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
                                saver.save(sess, model_folder+"/model.ckpt")
                                os.system("rm "+model_folder+"/infos")
                                os.system("touch "+model_folder+"/infos")
                                aze = open(model_folder+"/infos", "a")
                                aze.write("Model name : tiny_model_1layer2fc\n")
                                aze.write("Batch size : "+str(batch_size)+"\n")
                                aze.write("Image size : "+str(image_size)+"\n")
                                aze.write("Epoch : "+str(currentEpoch)+"/"+str(nb_epochs)+"\n")
                                aze.write("Learning rate : "+str(learning_rate)+"\n\n")
                                aze.write("Data used : "+str(dataPath)+"\n\n")
                                
                                
                                aze.write(str(len(classes))+" Classes : "+str(classes)+"\n\n")
                                for key, value in dataset.pathsByLabel.items():
                                        aze.write("Number of "+key+" : "+str(len(value))+"\n")
                                aze.write("\n")
                                aze.write("training Loss : "+str(round(np.mean(losses[-1000:]), 4))+"\n")
                                aze.write("train accuracy : "+str(round(trainAcc[len(trainAcc)-1], 4))+"\n")
                                aze.write("test accuracy : "+str(round(testAcc[len(testAcc)-1], 4))+"\n\n")
                                
                                aze.write("Confusion matrix : \n")
                                for j in range(0, len(dataset.inverted_labels_map)):
                                        aze.write(str(dataset.inverted_labels_map[j])+" ")
                                aze.write("\n")
                                aze.write(str(normal_confusion_matrix)+"\n\n")
                                
                                aze.write("Normalized confusion matrix : \n")
                                for j in range(0, len(dataset.inverted_labels_map)):
                                        aze.write(str(dataset.inverted_labels_map[j])+" ")
                                aze.write("\n")
                                aze.write(str(confusion_matrix)+"\n\n")
                                                                
                                aze.close()

                                # exporting .pb file
                                print("========= Exporting .pb and .pbtxt...=========")
                                graph = tf.get_default_graph()
                                input_graph_def = graph.as_graph_def()

                                output_graph_def = graph_util.convert_variables_to_constants(
                                        sess,
                                        input_graph_def,
                                        ["tiny_model/output/output/Softmax"])
                                
                                with tf.gfile.GFile(str(model_folder)+"/model.pb", "wb") as f:
                                        f.write(output_graph_def.SerializeToString())
                                        tf.train.write_graph(sess.graph_def, '.', str(model_folder)+'/model.pbtxt')
                                
                                        
                                
                        i+=1





