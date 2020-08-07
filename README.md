# trans-channel-paper
author: Daniel Wong (daniel.wong2@ucsf.edu)

The following python packages are required: 
torch 
torchvision
pandas
cv2
math
numpy
time 
shutil
os
socket
sklearn 
random
sys
pickle
matplotlib

Hardware Requirements:
All deep learning models were trained using NVIDIA 1080 GPUs

Content:
transchannel.py contains the majority of the necessary code to reproduce the main results of the tau portion of the paper
osteosarcoma.py contains the code necessary to train and evaluate the model on the independent osteosarcoma dataset
models:
fully trained models, including the model we trained and applied to the archival HCS, the 3 cross-validation models applied to an expanded dataset with negative controls (see Supplement), along with the 3 models trained with 3-fold cross-validation on the independent osteosarcoma dataset 



