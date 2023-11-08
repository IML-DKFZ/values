# Processing the GTA 5 and Cityscapes datasets

This folder contains the necessary code to process the GTA 5 and the cityscapes dataset. The main idea behind the combination
of those two dataset is to train only on the GTA 5 dataset first and then analyse the uncertainty outputs on both 
datasets and train another cycle including the most uncertain cases, including those from the cityscapes dataset.

## Setup

First, you need to download the datasets. The GTA dataset is available 
[here](https://download.visinf.tu-darmstadt.de/data/from_games/). 
The cityscapes dataset is available [here](https://www.cityscapes-dataset.com). 
Note, that you have to ask for access and create an account in order to download the cityscapes dataset.

To make sure the datasets are compatible with the scripts for preprocessing and the dataloader for later training, 
you should save them in one directory like this:

    <Path to save the dataset e.g. /home/user/GTA>
    ├── OriginalData
    ├── CityScapesOriginalData

Where OriginalData contains the GTA data and CityScapesOriginalData contains the cityscapes data

After downloading you should bring the GTA dataset in the following format:

    <Path to save the dataset e.g. /home/user/GTA>
    ├── OriginalData
        ├── images
            ├── 00001.png
            ├── 00002.png
            ├── ...
        ├── labels
            ├── 00001.png
            ├── 00002.png
            ├── ...

The cityscapes dataset should be in the following format

    <Path to save the dataset e.g. /home/user/GTA>
    ├── CityScapesOriginalData
        ├── images
            ├── leftImg8bit
                ├── train
                    ├── aachen
                        ├── aachen_000000_000019_leftImg8bit.png
                        ├── ...
                    ├── bochum
                    ├── ...
                ├── val
                    ├── frankfurt
                    ├── lindau
                    ├── munster
                ├── test
                    ├── berlin
                    ├── bielefeld
                    ├── ...
            ├── licence.txt
            ├── README
        ├── labels
            ├── gtFine
                ├── train
                    ├── aachen
                        ├── aachen_000000_000019_gtFine_color.png
                        ├── aachen_000000_000019_gtFine_instanceIds.png
                        ├── aachen_000000_000019_gtFine_labelIds.png
                        ├── aachen_000000_000019_gtFine_polygons.json
                        ├── ...
                    ├── bochum
                    ├── ...
                ├── val
                    ├── frankfurt
                    ├── lindau
                    ├── munster
                ├── test
                    ├── berlin
                    ├── bielefeld
                    ├── ...
            ├── licence.txt
            ├── README

## Preprocess the data

To preprocess the data, i.e. crop the images and convert the label maps to only use the training labels, run

```
python preprocess_gta_cityscapes.py -d <path to gta dataset, e.g. /home/user/GTA/OriginalData> --dataset gta
```

and 

```
python preprocess_gta_cityscapes.py -d <path to cityscapes dataset, e.g. /home/user/GTA/CityScapesOriginalData> --dataset cityscapes
```

for the GTA and the cityscapes dataset respectively. This will create a preprocessed folder with images and labels.
For the images and labels, a numpy version is created that is used as input for the model training and a "vis" folder
is created with visualizations of the images as png files and color maps of the labels instead of int labels.

Note, that especially for the gta dataset the preprocessing takes quite a while as 24,904 cases are preprocessed.

## Generating split files for first training cycle

For the initial training with this dataset, only GTA images are used. The cityscapes train images are put in the 
unlabeled pool and the cityscapes validation images will be used as OoD testset. The testset of the GTA images is 
generated randomly. To create the split file for the initial training cycle, run

```
python gta_cs_splits_first_cycle.py -d <path to both datasets with preprocessed data, e.g. /home/user/GTA>
```

You can also specify a different directory for the original dataset if your preprocessed data is stored in a different
directory than the original dataset.

Note, that the dataset has to be [preprocessed](#Preprocess the data) first.


