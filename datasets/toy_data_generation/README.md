# Toy Data Generation
This folder contains the necessary code to generate the datasets for the toy experiments using stl files.
3D datasets will be generated with .nii.gz (nifti) files. 

Here, it is described how you can exactly reproduce the datasets that we used in the paper, but also how you can create
your own datasets using custom stl files and blender.

## Setup

To generate the toy datasets based on stl files, you need to clone the forked stl-to-voxel repo.
Note, that the link for this is currently excluded for reasons of anonymity.

## Create the toy datasets used in our experiments

```
python dataset_generation_benchmark.py --base_save_path <path to save dir> --dataset_name <name of the dataset>
```

Here, base_save_path is the directory where you want to store the dataset. A subfolder with the dataset name will be 
created, containing the folders imagesTr, labelsTr, imagesTs, labelsTs. dataset_name is the name of the dataset to 
create, it should be one of the datasets specified in the configs subfolder. For more info about the dataset options, 
run 

```
python dataset_generation_benchmark.py -h
```

### Preprocess the data

To preprocess the data, i.e. normalize the images and save them as numpy arrays for the input for later training, 
you can use the preprocess_datasets_3d script in the datasets folder of this repository. To execute the script, run

```
cd ..
python preprocess_datasets_3d.py -d <base path to toy data dir> -r 3 -i imagesTr imagesTs -l labelsTr labelsTs --dataset toy
```

### Generating split files

To generate the train / val / test split for training, run

```
python create_splits.py -d <base path to toy data dir>
```

Note, that in contrast to the other datasets, no active learning experiments were performed on the toy dataset.
This means that in the split file, only a simple train / val / test split is generated without unlabeled pool.

### Folder structure

If you set up everything like described above, your toy dataset structure should look like this:

    Case_X
    ├── imagesTr
      ├── 0000.nii.gz
      ├── 0001.nii.gz
      ├── ...
    ├── imagesTs
      ├── 0000.nii.gz
      ├── 0001.nii.gz
      ├── ...
    ├── labelsTr
      ├── 0000_00.nii.gz
      ├── ...
    ├── labelsTs
      ├── 0000_00.nii.gz
      ├── ...
    ├── preprocessed
       ├── imagesTr
         ├── 0000.npy
         ├── 0001.npy
         ├── ...
       ├── imagesTs
         ├── 0000.npy
         ├── 0001.npy
         ├── ...
       ├── labelsTr
         ├── 0000_00.npy
         ├── ...
       ├── labelsTs
         ├── 0000_00.npy
         ├── ...
    ├── splits.pkl

## Create your own custom toy datasets

### Current workflow with Blender

1. Install [Blender](https://www.blender.org/)
2. Download and install the [rheo stl export plugin](https://rheologic.net/articles/blender-object-export-separate-stl/#:~:text=To%20use%20the%20addon%2C%20simply,field%20of%20the%20export%20dialog).  
   This is important if you have multiple overlapping objects in Blender and you want them as single object in the nifti file.
3. Create the scene you want as toy data image in Blender
4. Select all objects in Blender (Shortcut A key) and go to File > Export > Rheo STL.  
   This will export every object as individual stl file

Of course, you can create the STL files you want to convert with any other software. 
This is independent of the further steps.

### Convert STL files to nifti

To convert the stl files to a nifti file, run
```
python stl_to_nifti.py --input_files <path to stl 1> <path to stl 2> <...> --save_file_name <path to nifti>
```

This will merge all the input files into a single nifti file with all object having the value 1 and background 0.  
Note, that if you exported multiple overlapping objects as a single STL file instead of one STL file per object, 
the overlapping area will be marked as background. To avoid this, follow the [workflow instructions with Blender](#Current workflow with Blender).  

Per default, the object size is 100 and it will be rendered in an image of size 256x256 pixels with an offset of 50 pixels in all directions.
However, you can change all of these parameters.

There is also the possibility to add noise and / or blur to your final nifti image.

For the other options to convert the STL files to a nifti, run
```
python stl_to_nifti.py -h
```

### Generate a dataset

You also have the possibility to create a whole dataset, where the object in the STL file(s) will be rendered in various random sizes at random positions in the image.
Additionally, you have the option to create segmentations for this dataset with a varying number of raters.
Multiple raters are useful of you include gaussian blur in your data and want to create ambiguous segmentations.

To create a dataset, run
```
python dataset_generation.py --input_files <path to stl 1> <path to stl 2> <...> --save_path <path to save folder> --segmentation
```

This will create a dataset of 10 samples including a segmentation from one rater in the directory you specified in save_path. 
This is just the option without blur and noise, but you again have the possibility to add noise and / or blur.

For the other options to create a dataset, run
```
python dataset_generation.py -h
```