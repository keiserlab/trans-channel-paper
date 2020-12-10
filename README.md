# trans-channel-paper
author: Daniel Wong (wongdanr@gmail.com)

## Open access image data
https://osf.io/xntd6/ <br />
Identifier: DOI 10.17605/OSF.IO/XNTD6

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

**transchannel.py** contains the majority of the necessary code to reproduce the main results of the paper. See the docstrings for specifics. 

**transchannel_runner.py** This script pertains to the model training, evaluation, and supplemental analyses of the tauopathy sections of the paper. It can be run with the command "python transchannel_runner.py {FOLD NUMBER}". When specifying a 70/30 split, set FOLD NUMBER to be NOT in {1,2,3}.

**transchannel_tests.py** This script contains unit tests and integration tests for key functions in transchannel.py.

**osteosarcoma.py** This script pertains to the osteosarcoma sections of the paper. It is the runner code for training and evaluation on the functional genomics dataset. 

**figure_generator.py** contains code to reproduce the figures.

**bash_script.sh** convenient bash script to run all of the code, from model training to model evaluation and figure generation

**models:**<br />
This folder contains the fully trained models, including: <br />
The tauopathy model applied to the archival HCS<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;("raw_1_thru_6_full_Unet_mod_continue_training_2.pt")  <br />
The 3 models trained with threefold cross-validation on the ablated (95th percentile) osteosarcoma dataset  <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;d0_to_d1_ablation_cyclin_only_dataset_fold1_continue_training.pt <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;d0_to_d1_ablation_cyclin_only_dataset_fold2_continue_training.pt <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;d0_to_d1_ablation_cyclin_only_dataset_fold3_continue_training.pt <br />
The 3 models trained with threefold cross-validation on the raw, unablated osteosarcoma dataset<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;d0_to_d1_cyclin_only_dataset_fold1.pt  <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;d0_to_d1_cyclin_only_dataset_fold2.pt <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;d0_to_d1_cyclin_only_dataset_fold3.pt <br />
The two single channel models (i.e. one channel input to one channel output) for the supplemental analysis of the tauopathy study <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;DAPI_only_to_AT8.pt<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;YFP_only_to_AT8.pt<br />

**csvs:**<br />
This folder contains the CSVs with matching string pointers for the image datasets <br />

**stats:**<br />
This folder contains .npy files with stats used for normalization <br />

**pickles:**<br />
This folder is used for saving different results as pickles <br />

**matplotlib_figures:**<br />
This folder is where figure_generator.py saves its files <br />

**outputs:**<br />
This folder is a temporary directory used for saving images <br />






