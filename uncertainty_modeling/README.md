# Training and Inference of the models

This part of the repository contains the code for training the models and infering on the test sets.  
The entry point for training the models is thereby ```main.py``` and the enrty points for the inference are ```test_3D.py``` and ```test_2D.py``` for the 3D datasets (Toy data and LIDC data) and the 2D datasets (GTA 5/Cityscapes) repectively.  
The following sections will explain the configs for how to start a training and the settings for starting an inference in more detail. Note that it is assumed that you have preprocessed the datasets as described in the respecive ```datasets``` subfolder at this point.


## Running a training

For running a training, execute ```main.py``` with the appropriate configuration for the setup you want to train. For examples on how to configure a training, see the config files in the ```configs``` subfolder. Generally, there are entry files on the top level of this folder, e.g. ```softmax_config.yaml``` which themselves include different datamodules, models, etc., specified in the corresponding subfolders.  
The structure for the entry configuration is like this:

```yaml
defaults:
    - datamodule: <name of datamodule>
    - model: <name of the model>
    # if data augmentations are used
    - data_augmentations: <used data augmentations>

# Save directory for a specific experiment version is made up of save_dir/exp_name/version
exp_name: <name of experiment, mostly prediction model, e.g. "Softmax">
version: <name of the version, usually made up of seed and fold, e.g. fold0_seed123 or additional properties like pretrain epochs (for SSNs) etc. Basically everything that is unique about the experiment version>
save_dir: <base_path/to/save/experiments>
# datasets should be preprocessed as described in datasets subfolder
data_input_dir: <base_path/to/data>
seed: <seed for experiment>

# training params
max_epochs: <number of epochs to train>
batch_size: <batch size>
learning_rate: <learning rate>
weight_decay: <weight decay>
gpus: <gpus to use>

logger:
  _target_: pytorch_lightning.loggers.TensorBoardLogger
  save_dir: ${save_dir}
  name: ${exp_name}
progress_bar:
  _target_: pytorch_lightning.callbacks.TQDMProgressBar
  refresh_rate: 10

# optional, see gta config for example
optimizer:
  _target_: <optimizer to use>
  lr: ${learning_rate}
  weight_decay: ${weight_decay}
  # ... (additional params)
lr_scheduler:
  _target_: <lr scheduler to use>
  # ... (additional params)
```

The datamodule and model configs are really dependent on the implementations of the datamodule and model they are instantiating. For examples, look in the corresponding subfolders.

## Running inference

For running the inference after you performed a training, execute ```test_3D.py``` for 3D datasets (Toy dataset, LIDC) and ```test_2D.py``` for 2D datasets (GTA5/Cityscapes). 
For the parameters that you can specify, run ```python test_3D.py -h```.  
The only parameter that is not optional, is ```checkpoint_paths``` where you define the checkpoint based on which you want to perform the inference. For ensemble inference, specify all the paths that you want to use for inference.  
Another parameter that you mostly need to specify are the number of predictions that you want to make (```--n_pred```) if you want to sample multiple output segmentations.  
All other parameters like the dataset location etc. can be inferred from the checkpoint itself, although you may want to change them, e.g. if you train and infere on different machines or if your testset is located in a special directory.


After inference, the results are stored in a subfolder called ```test_results```, which has the following structure:


    test_results
    ├── <version>
        ├── aleatoric_uncertainty
        ├── epistemic_uncertainty
        ├── pred_entropy
        ├── gt_seg
        ├── input
        ├── pred_prob
        ├── pred_seg
        ├── metrics.json


This means that during testing, besides the segmentation prediction, also the uncertainty maps are generated and a metrics file which contains metrics regarding the segmentation performance (Dice) and ambiguity modeling (GED). Note that the aleatoric_uncertainty and epistemic_uncertainty directory only exist for methods that sample multiple segmentations, i.e. not for the plain softmax prediction model. Further, for the GTA5/Cityscapes dataset, the input and the predicted probabilities (pred_prob) are not saved for each experiment. For further analyzing and processing the results, see the ```evaluation``` subfolder of this repository.