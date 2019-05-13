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
                print("== DONE ==")

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
                        images.append(entry["image"])
                        labelVector = self.getLabelVector(self.labels_map[entry["label"]])
                        labels.append(labelVector)

                self.currentBatch +=1
                return self.currentBatch, images, labels
        
        
        def getValidationSet(self):

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
                totalNumberOfImages = 0
                self.pathsByLabel = {}
                for key in self.labels_map.keys():
                        self.pathsByLabel[key] = []
                        currentDataPath = self.dataPath+"/"+key+"/"
                        
                        for file in os.listdir(currentDataPath):
                                if(os.path.isfile(currentDataPath+file)):
                                        if file.endswith('.png'):
                                                self.pathsByLabel[key].append(currentDataPath+file)
                                                totalNumberOfImages += 1

                        random.shuffle(self.pathsByLabel[key])

                minNb = None
                minNbLabel = ""                                                
                for key, value in self.pathsByLabel.items():
                        # print("Number of "+key+" : "+str(len(value)))
                        if minNb == None or len(value) < minNb:
                                minNb = len(value)
                                minNbLabel = key


                for key, value in self.pathsByLabel.items():
                        if key != minNbLabel:
                                self.pathsByLabel[key] = self.pathsByLabel[key][:-(len(self.pathsByLabel[key])-minNb)]

                for key, value in self.pathsByLabel.items():
                        print("Number of "+key+" : "+str(len(value)))                

                        
                ii = 0
                print("== Loading Images ...  ==")
                for key, value in self.pathsByLabel.items():
                        for i in range(0, len(value)):
                                percentage = int(ii/totalNumberOfImages*100)
                                if i < totalNumberOfImages-1:
                                        print("Loaded "+str(percentage)+"% of the images", end="\r")
                                else:
                                        print("Loaded 100% of the images")
                                        print("== DONE ==")
                                
                                imagePath = value[i]
                                
                                
                                if self.gray_scale:
                                        im = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
                                        nbChannels = 1
                                else:
                                        im = cv2.imread(imagePath)
                                        nbChannels = 3
                                
                                im = cv2.resize(im, (self.image_size[1], self.image_size[0]))
                                im = cv2.normalize(im, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)        
                                
                                tmpIm = np.ndarray(shape=(self.image_size[1], self.image_size[0], nbChannels))
                                tmpIm = np.reshape(im, (self.image_size[1], self.image_size[0], nbChannels))
                                
                                if not self.imShape:
                                        self.imShape = tmpIm.shape
                                        
                                if i < self.train_proportion*len(value): # TRAINING
                                        self.trainingSet.append({"image" : tmpIm, "label" : key})
                                else:                                    # VALIDATION
                                        self.validationSet.append({"image" : tmpIm, "label" : key})
                                ii+=1

                random.shuffle(self.validationSet)
                random.shuffle(self.trainingSet)
        
                                        
        def __init__(self, dataPath, imSize, trainProportion, classes, gray_scale):
                self.image_size = imSize
                self.dataPath = dataPath
                self.train_proportion = trainProportion
                self.currentBatch = 0
                self.imShape = ()
                self.gray_scale = gray_scale
                
                self.trainingSet = [] # [{image, label}]
                self.validationSet = [] # [{image, label}]

                self.buildLabelsMapAndInvertedLabelsMap(classes)
                self.buildTrainingAndValidationSets()

                self.batches = []
                print("Size of training set :" + str(len(self.trainingSet)))
                print("Size of validation set :" + str(len(self.validationSet)))

                
if __name__ == "__main__":

        classes = ["class1", "class2", "class3"] # any number of classes
        dataPath = "path_to_your_data/"
        train_proportion = 0.8
        batch_size = 256
        gray_scale = False

        image_size = [128, 128]
        
        dataset = Dataset(dataPath, image_size, train_proportion, classes, gray_scale)        

        nbBatches = dataset.buildBatches(batch_size)

        currentBatch, images, labels = dataset.getBatch()
        print(len(images))
        print(len(labels))
        
