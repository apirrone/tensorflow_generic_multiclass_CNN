# Tensorflow Generic Multiclass CNN

Inspired by https://github.com/ardiya/siamesenetwork-tensorflow

## Architecture overview

### dataset.py : 

Assumes the layout of your data is as the following :

|
|_<root_data_directory>
	|_<class1>
	|  | <image1>
	|  | <image2>
	|  | ...
	[
	|_<class2>
	|  | <image1>
	|  | <image2>
	|  | ...
	...
