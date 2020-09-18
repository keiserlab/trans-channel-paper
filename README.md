# trans-channel-paper
author: Daniel Wong (daniel.wong2@ucsf.edu)

## Open access image data
(put link here when finished)

## The following python packages are required: 
torch 1.2.0 <br />
torchvision 0.4.0 <br />
pandas 0.24.2 <br />
cv2 4.1.1.26 <br />
numpy 1.16.4 <br />
sklearn 0.21.2 <br />

 
## Hardware Requirements:
All deep learning models were trained using Nvidia Geforce GTX 1080 GPUs

## Content:

**transchannel.py** contains the majority of the necessary code to reproduce the main results of the paper. See the documentation within this script for specifics. 

**transchannel_runner.py** This script pertains to the model training, evaluation, and supplemental analyses of the tauopathy sections of the paper. It can be run with the command "python transchannel_runner.py {FOLD NUMBER}". When specifying a 70/30 split, set FOLD NUMBER to be not in {1,2,3}.

**transchannel_tests.py** This script contains unit tests and integration tests for key functions in transchannel.py.

**osteosarcoma.py** This script pertains to the osteosarcoma sections of the paper. It is the runner code for training and evaluation on the functional genomics dataset. 

**ExampleFigureGenerator.ipynb** contains example code to reproduce the main performance results for the tau dataset.

**models:**
This folder contains the fully trained models, including: <br />
The model we applied to the archival HCS ("raw_1_thru_6_full_Unet_mod_continue_training_2.pt")  <br />
The 3 cross-validation models applied to an expanded dataset with negative controls (see Supplement)  <br />
&nbsp;&nbsp;&nbsp;&nbsp;tau_cross_validate_fold1.pt, tau_cross_validate_fold2.pt, tau_cross_validate_fold3.pt <br />
The 3 models trained with threefold cross-validation on the ablated (95th percentile) osteosarcoma dataset  <br />
&nbsp;&nbsp;&nbsp;&nbsp;d0_to_d1_ablation_cyclin_only_dataset_fold1_continue_training.pt, d0_to_d1_ablation_cyclin_only_dataset_fold2_continue_training.pt, d0_to_d1_ablation_cyclin_only_dataset_fold3_continue_training.pt <br />
The 3 models trained with threefold cross-validation on the raw, unablated osteosarcoma dataset  <br />
&nbsp;&nbsp;&nbsp;&nbsp;d0_to_d1_cyclin_only_dataset_fold1.pt, d0_to_d1_cyclin_only_dataset_fold2.pt, d0_to_d1_cyclin_only_dataset_fold3.pt








