# Processing the LIDC-IDRI dataset
This folder contains the necessary code to retrieve the LIDC-IDRI dataset, a dataset of lung nodules, as we used it 
in our experiments. This will generate a cropped version of the nodules of size 64x64x64 as .nii.gz (nifti) files.

## Setup

First, you need to download the original image data of the LIDC-IDRI dataset according to the 
[Cancer Imaging Archive](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254).
Next, the data can be preprocessed using [pylidc](https://pylidc.github.io). Note, that unfortunately this library
requires an older version of numpy, which is why you should probably do the processing of the LIDC-IDRI data in a
separate virtual environment. We provide the requirements_lidc.txt file in this folder to install all the necessary
dependencies in this virtual environment. Simply run

```
pip install -r requirements_lidc.txt
```

In your separate virtual environment to install the necessary requirements.

Furthermore, you should provide the .pylidcrc file as described in the
[setup instructions of the pylidc package](https://pylidc.github.io/install.html).

## Create dataset with cropped nodules

To create the dataset that we used to get our experimental results, with cropped nodules of size 64x64x64, run 

```
python save_cropped_nodules.py -s <path to save dir>
```

This will create the subfolders images and labels with the corresponding cropped nodules and annotations in them.
Furthermore, the metadata for each nodule will be stored in a csv file (metadata.csv)