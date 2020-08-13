# trans-channel-paper
author: Daniel Wong (daniel.wong2@ucsf.edu)

The following python packages are required: 
torch, torchvision, pandas, cv2, numpy, sklearn, pickle, matplotlib

Hardware Requirements:
All deep learning models were trained using Nvidia Geforce GTX 1080 GPUs

Content:

transchannel.py contains the majority of the necessary code to reproduce the main results of the tau portion of the paper. This script can be run with a command such as "python transchannel.py {FOLD NUMBER}". See the documentation within this script for specifics. Be sure to set any parameters as specified in the section of transchannel.py titled "SETTING GLOBAL VARIABLES" prior to running the script. 

osteosarcoma.py contains the code necessary to train and evaluate the model on the independent dataset of the genome-wide functional screen in osteosarcoma cells 

ExampleFigureGenerator.ipynb contains example code to reproduce the main performance results for the tau dataset

models:
fully trained models, including the model we trained and applied to the archival HCS ("raw_1_thru_6_full_Unet_mod_continue_training_2.pt"), the 3 cross-validation models applied to an expanded dataset with negative controls (see Supplement), along with the 3 models trained with 3-fold cross-validation on the independent osteosarcoma dataset 








