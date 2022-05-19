# Working Pytorch-Lightning Example of a Segmentation Problem

The Goal of this Repository is to showcase the use of Pytorch-Lightning on Medical Image data segmentation problems solved with a U-Net.  
The segmentation tasks that are solved are from the [Medical Segmentation Decathlon](http://medicaldecathlon.com/). Specifically, the repository contains code for the Hippocampus and the Heart Dataset, but you can extend it for other datasets as well.  
A 2D U-Net is trained which gets single slices of 3D medical images as input.  
General knowledge about Pytorch Modules & DataLoaders is advantegous, as well as BatchGenerators.


## Fast start 

1. Clone this repository
2. go into the folder
3. pip install .
4. change unet_defaults.yml or unet_defaults_heart.yml variables to your desired directories
    - save_dir --> logs are saved here
    - data_input_dir --> data is saved and load from here
5. Run light_seg/main.py.  
   If you want to use unet_defaults_heart.yml file as config file, change 
   ```
   hparams, nested_dict = main_cli(config_file="./unet_defaults.yml")
   ```
   to
   ```
   hparams, nested_dict = main_cli(config_file="./unet_defaults_heart.yml")
   ```
   at the bottom of main.py
6. Experiment with settings and code
7. Change test_unet_defaults.yml to your trained checkpoint.
8. Run test.py
9. More information about pytorch-lightning on
    - [Official Docs](https://pytorch-lightning.rtfd.io/en/latest/)
    - [Pytorch-Lightning-Bolts Docs](https://pytorch-lightning-bolts.readthedocs.io/en/latest/)

## General structure of the repository

In the following, the functionality of the individual files will be briefly described.  
In general, there are scripts that are important for training and scripts that are important for testing your trained network.  

The scripts that are important for training are: 


- Config files: These files contain the configuration that is used during training. Specifically, you can set the dataset location, training parametes like the fold or the number of epochs and others.
   - unet_defaults.yml: This is a default configuration which is mainly for training on the Hippocampus dataset (adapt the dataset location to your local paths)
   - unet_defaults_heart.yml: This is a default configuration which is mainly for training on the Heart dataset (adapt the dataset location to your local paths)
- main.py: This is the entry point for the training. Here, the parameters are set, the datamodule, trainer and model are defined and the training loop is started.
- Datamodules: In the datamodules, the dataset is downloaded (this is currently only possible for the Hippocampus dataset, the Heart dataset needs to be downloaded manually), preprocessed and during training time, batches are loaded and augmented.
   - msd_datamodule.py
   - hippocampus_datamodule.py
   - heart_datamodule.py
- unet_lightning.py: This file contains all the necessary parts of the training loop. There, also the U-Net model itself is instantiated.
- unet_module.py: This contains the architecture definition of the U-Net model that is trained.
- loss_modules.py: Here, the soft dice loss is defined which is used as part of the loss function during training.


The scripts that are important for testing are:

- test_unet_defaults.yml: This contains the default configuration for testing. Most importantly, you need to specify the checkpoint from your training that you want to use for your predictions. You can also specify the location of the data and the saving of the test results in case you train and test on different machines. If you train and test on the same machine, this information can also be inferred from the checkpoint.
- test.py: This is the main test loop. The test data will be loaded and processed as 2D slices. In the end, the results are saved in the original 3D format.
- data_carrier.py: This class is responsible for the correct handling of the individual 2D slices that they are correctly saved as 3D images/segmentations in the end. The results will be saved in a specific folder structure (see section below)


### Structure of the test results

The results of the test run are stored as 3D nifti in a specific folder structure, providing the original input to the network, the ground truth segmentation, the predicted segmentation, the softmax predictions and the calculated metrics for the test. Normally, you should have a folder structure like this (either in your specified save_dir from test_unet_defaults.yml if specified or in the save_dir of your training):  
 
    test_results
    ├── <version>
    │   ├── gt_seg
    │   ├── input
    │   ├── pred_prob
    │   ├── pred_seg
    │   ├── metrics.json
    ├── ...



## Pytorch lightning specific code
The relevant code for pytorch-lightning is situated in: 

- unet_lightning.py / UNetExperiment (pl.LightningModule)
- hippocampus_datamodule.py / HippocampusDataModule (pl.LightningDataModule)
- main.py / main_cli (efficient parsing and hparameter handling)
- main.py / main (use of pl.Trainer)

For more complex examples where scores need to be computed over complete datasets it is advised to use the following methods in your pl.LightiningModule:

- train_epoch_end
- validation_epoch_end
- test_epoch_end

These methods overwrite the normal behaviour of pl.EvalResult & pl.TrainResult when they are used.

## Experiment results

In the following, the test results on the two implemented datasets are shown. The networks were trained on 50, 100 and 200 epochs.  
As the results did not differ much for the different number of training epochs, here the results for 200 epochs are shown.  
Furthermore, they are compared with the results of a 2D [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) which was trained and tested on the same data.

### Hippocampus dataset

#### Pytorch-Lightning Example

<table>
  <tr>
   <td>Fold</td>
   <td>Dice</td>
  </tr>
  <tr>
   <td>0</td>
   <td>0.8822</td>
  </tr>
  <tr>
   <td>1</td>
   <td>0.8809</td>
  </tr>
  <tr>
   <td>2</td>
   <td>0.8798</td>
  </tr>
  <tr>
   <td>3</td>
   <td>0.878</td>
  </tr>
  <tr>
   <td>4</td>
   <td>0.8805</td>
  </tr>
  <tr>
   <td>Mean</td>
   <td>0.8803</td>
  </tr>
</table>

#### nnU-Net

<table>
  <tr>
   <td>Fold</td>
   <td>Dice</td>
  </tr>
  <tr>
   <td>0</td>
   <td>0.8748</td>
  </tr>
  <tr>
   <td>1</td>
   <td>0.8764</td>
  </tr>
  <tr>
   <td>2</td>
   <td>0.8768</td>
  </tr>
  <tr>
   <td>3</td>
   <td>0.875</td>
  </tr>
  <tr>
   <td>4</td>
   <td>0.876</td>
  </tr>
  <tr>
   <td>Mean</td>
   <td>0.8758</td>
  </tr>
</table>

### Heart dataset

#### Pytorch-Lightning Example

<table>
  <tr>
   <td>Fold</td>
   <td>Dice</td>
  </tr>
  <tr>
   <td>0</td>
   <td>0.8735</td>
  </tr>
  <tr>
   <td>1</td>
   <td>0.8895</td>
  </tr>
  <tr>
   <td>2</td>
   <td>0.8905</td>
  </tr>
  <tr>
   <td>3</td>
   <td>0.8989</td>
  </tr>
  <tr>
   <td>4</td>
   <td>0.8942</td>
  </tr>
  <tr>
   <td>Mean</td>
   <td>0.8794</td>
  </tr>
</table>

#### nnU-Net

<table>
  <tr>
   <td>Fold</td>
   <td>Dice</td>
  </tr>
  <tr>
   <td>0</td>
   <td>0.9227</td>
  </tr>
  <tr>
   <td>1</td>
   <td>0.9205</td>
  </tr>
  <tr>
   <td>2</td>
   <td>0.9197</td>
  </tr>
  <tr>
   <td>3</td>
   <td>0.9052</td>
  </tr>
  <tr>
   <td>4</td>
   <td>0.9183</td>
  </tr>
  <tr>
   <td>Mean</td>
   <td>0.9173</td>
  </tr>
</table>


## Extend it for your own dataset

Of course you can use this repository as a basis to run experiments with your own datasets.  
If you specifically want to adapt it for another dataset from the Medical Segmentation decathlon, you can derive from the msd_datamodule.py like for the Hippocampus and the Heart dataset.

---

<br>
<p align="left">
  <img src="https://drive.google.com/uc?export=view&id=1RCtBi7LMskVITseelKDgZedPUOTeYXLH" width="350"> &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://avatars.githubusercontent.com/u/31731892?s=280&v=4" width="300"> 
</p>
<p align="left">
  <img src="https://polybox.ethz.ch/index.php/s/kqDrOTTIzPFYPU7/download" width="150"> &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Deutsches_Krebsforschungszentrum_Logo.svg/1200px-Deutsches_Krebsforschungszentrum_Logo.svg.png" width="500"> 
</p>

This library is developed and maintained by the [Medical Image Computing Group](https://www.dkfz.de/en/mic/index.php) of the [DKFZ](https://www.dkfz.de/de/index.html) and the [Interactive Machine Learning Group](https://iml-dkfz.github.io/) of [Helmholtz Imaging](https://www.helmholtz-imaging.de/) and the [DKFZ](https://www.dkfz.de/de/index.html).

