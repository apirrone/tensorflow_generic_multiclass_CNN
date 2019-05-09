# Tensorflow Generic Multiclass CNN

Inspired by https://github.com/ardiya/siamesenetwork-tensorflow

## Architecture overview

### dataset.py : 

Assumes the layout of your data is as the following :

```
root_data_directory
└───class_1_directory
│   │   image1
│   │   image2
│   │	...
└───class_1_directory
│   │   image1
│   │   image2
│   │	...
|   ...
```

Classes directories must have distinct names and represent the label of the class.

You can create a `Dataset` object this way : 

	dataset = Dataset(data_path, image_size, train_proportion, classes)
	
- `data_path` (string): the path to `root_data_directory`
- `image_size` ([int, int]): the size of the images [width, height]
- `train_proportion` (float between 0 and 1): the proportion of images that will be used for training, the remaining will be used for validation
- `classes` (list of strings): the names of the classes that will be used (should exactly match the names of the subdirectories of `root_data_directory`


