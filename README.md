# Modeling Uncertainty for Segmentation

This repository contains the code for the paper:

ValUES: A Framework for Systematic Validation of Uncertainty Estimation in Semantic Segmentation

Parts of this repository concerning the model training in uncertainty_modeling are based on the 
[Pytorch Lightning Segmentation Example](https://github.com/IML-DKFZ/lightning-segment).


## Setup

1. Clone this repository
2. Install the requirements
   ```
   pip install -r requirements.txt
   ```

## Structure of this repository

This repository consists of three main parts which are in separate folders:

- datasets: This folder contains all the code to set up the datasets. This includes the used toy dataset, 
  the lung nodule dataset (lidc-idri) and the GTA5 / Cityscapes dataset.
- uncertainty_modeling: This folder contains the main part for training and inference of the various 
  uncertainty methods.
- experiment_analysis: This folder contains the code for analysis of the experiments that is done after the inference, 
  e.g. OoD detection performance, AURC etc.

Each subfolder contains separate README files with instructions how to set everything up.

---

<br>
<p align="left">
  <img src="https://drive.google.com/uc?export=view&id=1RCtBi7LMskVITseelKDgZedPUOTeYXLH" width="350"> &nbsp;&nbsp;&nbsp;&nbsp;
</p>
<p align="left">
  <img src="https://polybox.ethz.ch/index.php/s/kqDrOTTIzPFYPU7/download" width="150"> &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Deutsches_Krebsforschungszentrum_Logo.svg/1200px-Deutsches_Krebsforschungszentrum_Logo.svg.png" width="500"> 
</p>

This library is developed and maintained by the [Interactive Machine Learning Group](https://iml-dkfz.github.io/) of [Helmholtz Imaging](https://www.helmholtz-imaging.de/) and the [DKFZ](https://www.dkfz.de/de/index.html).

