import numpy as np
import cv2
import os
import random
import utils

class Dataset():

        def buildLabelsMapAndInvertedLabelsMap(self, classes):
                print("== Building labels_map and inverted_labels_map ==")
                
                classes = sorted(classes) # sort alphabetically
                self.labels_map = {}
                self.inverted_labels_map = {}
                for i in range(0, len(classes)):
                        self.labels_map[classes[i]] = i
                        self.inverted_labels_map[i] = classes[i]

                print(self.labels_map)
                print(self.inverted_labels_map)

        def getLabelVector(self, label):
                labelVector = []
                for i in range(0, len(self.labels_map)):
                        if i == label:
                                labelVector.append(1)
                        else:
                                labelVector.append(0)

                return labelVector                        
                
        def getBatch(self):

                images = []
                labels = []

                for entry in self.batches[self.currentBatch]:
                        im = cv2.imread(entry["image"])
                        im = cv2.resize(im, (self.image_size, self.image_size))
                        im = cv2.normalize(im, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                        
                        tmpIm = np.ndarray(shape=(self.image_size, self.image_size, 3))
                        tmpIm = np.reshape(im, (im.shape[0], im.shape[1], 3))
                        
                        images.append(tmpIm)
                        
                        labelVector = self.getLabelVector(self.labels_map[entry["label"]])
                        labels.append(labelVector)

                self.currentBatch +=1 
                return self.currentBatch, images, labels
        
        
        def getValidation(self):

                images = []
                labels = []
                
                for entry in self.validationSet:
                        images.append(entry["image"])
                        labelVector = self.getLabelVector(self.labels_map[entry["label"]])
                        labels.append(labelVector)

                return images, labels


        def buildBatches(self, batchSize):
                self.currentBatch = 0
                nbBatches = len(self.trainingSet)//batchSize+1
                self.batches = []
                for i in range(0, nbBatches):
                        self.batches.append([])
                        
                currentBatchIdx = 0
                for i in range(0, len(self.trainingSet)):

                        self.batches[currentBatchIdx].append(self.trainingSet[i])
                        
                        if i!=0 and i%batchSize == batchSize-1:
                                currentBatchIdx += 1

                # GROSS sparadrap
                indicesToRemove = []
                for i in range(0, len(self.batches)):
                        if len(self.batches[i]) != batchSize:
                                indicesToRemove.append(i)

                a = 0
                for i in indicesToRemove:
                        self.batches.pop(i-a)
                        a+=1
                        
                return len(self.batches)
                        

        def buildTrainingAndValidationSets(self):
                
                self.pathsByLabel = {}
                for key in self.labels_map.keys():
                        self.pathsByLabel[key] = []
                        currentDataPath = self.dataPath+"/"+key+"/"
                        
                        for file in os.listdir(currentDataPath):
                                if(os.path.isfile(currentDataPath+file)):
                                        if file.endswith('.JPG'):
                                                self.pathsByLabel[key].append(currentDataPath+file)

                        random.shuffle(self.pathsByLabel[key])
                                                
                for key, value in self.pathsByLabel.items():
                        print("Number of "+key+" : "+str(len(value)))

                for key, value in self.pathsByLabel.items():
                        for i in range(0, len(value)):
                                if i < self.train_proportion*len(value): # TRAINING
                                        self.trainingSet.append({"image" : value[i], "label" : key})
                                else:                                    # VALIDATION
                                        imagePath = value[i]
                                        im = cv2.imread(imagePath)
                                        im = cv2.resize(im, (self.image_size, self.image_size))
                                        im = cv2.normalize(im, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                                        
                                        tmpIm = np.ndarray(shape=(self.image_size, self.image_size, 3))
                                        tmpIm = np.reshape(im, (im.shape[0], im.shape[1], 3))
                                        if not self.imShape:
                                                self.imShape = tmpIm.shape
                                        self.validationSet.append({"image" : tmpIm, "label" : key})

                random.shuffle(self.validationSet)
                random.shuffle(self.trainingSet)
        
                                        
        def __init__(self, dataPath, imSize, trainProportion, classes):
                self.image_size = imSize
                self.dataPath = dataPath
                self.train_proportion = trainProportion
                self.currentBatch = 0
                self.imShape = ()
                
                self.trainingSet = [] # [{image, label}]
                self.validationSet = [] # [{image, label}]

                self.buildLabelsMapAndInvertedLabelsMap(classes)
                self.buildTrainingAndValidationSets()

                self.batches = []
                print("Size of training set :" + str(len(self.trainingSet)))
                print("Size of validation set :" + str(len(self.validationSet)*2))

                
if __name__ == "__main__":

        classes = ["demotic", "greek"]#, "coptic"]
        dataPath = "data/testDemoticGreek/"
        dataset = Dataset(dataPath, 256, 0.8, classes)        

        nbBatches = dataset.buildBatches(64)
        for b in dataset.batches:
                print(len(b))

        currentBatch, images, labels = dataset.getBatch()
        print(len(images))
        print(len(labels))
        
