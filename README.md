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

**transchannel_runner.py** The script to run the code found within transchannel.py. This script pertains to the tauopathy sections of the paper, and can be run with a command such as "python transchannel_runner.py {FOLD NUMBER}", or when specifying a 70/30 split, FOLD_NUMBER not in {1,2,3}.

**transchannel_tests.py** This script contains unit tests and integration tests for key functions in transchannel.py

**osteosarcoma.py** This script pertains to the osteosarcoma sections of the paper. It is the runner code for training and evaluation on the functional genomics dataset. 

**ExampleFigureGenerator.ipynb** contains example code to reproduce the main performance results for the tau dataset

**models:**
fully trained models, including the model we trained and applied to the archival HCS ("raw_1_thru_6_full_Unet_mod_continue_training_2.pt"), the 3 cross-validation models applied to an expanded dataset with negative controls (see Supplement), along with the 3 models trained with 3-fold cross-validation on the independent osteosarcoma dataset 








