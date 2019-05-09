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

