"""
This is the core of the necessary code to reproduce the main training and model evaluation results from the tauopathy dataset and the osteosarcoma dataset
This script is divided into three parts:
1) class and method definitions for main results
2) helper functions
3) supplemental code
"""
import torch 
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.sampler import SequentialSampler
import pandas as pd
import cv2
import math
import numpy as np
import time 
import shutil
import os
import socket
import sklearn 
import random
import sys
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
hostname = socket.gethostname() 

#============================================================================
#============================================================================
## CLASS AND METHOD DEFINITIONS
#============================================================================
#============================================================================
class ImageDataset(Dataset):
    """
    Dataset of tau images. INPUTMIN, INPUTMAX, DAPIMIN, DAPIMAX, LABELMIN, and LABELMAX all correspond to float values to normalize our images 
    DATA_DIR specifies where to read the raw images from 
    getitem returns (YFP image name, YFP + DAPI concatenated image, AT8 pTau image)
    CSV_FILE should be a csv of x,t pairs corresponding to image names such that 
    x is the YFP-tau image name, and t is the AT8-pTau image name
    """
    def __init__(self, csv_file, inputMin, inputMax, DAPIMin, DAPIMax, labelMin, labelMax, DATA_DIR):
        self.data = pd.read_csv(csv_file).values
        self.inputMin = inputMin
        self.inputMax = inputMax
        self.DAPIMin = DAPIMin
        self.DAPIMax = DAPIMax
        self.labelMin = labelMin
        self.labelMax = labelMax
        self.DATA_DIR = DATA_DIR
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x, t = self.data[idx] 
        x = self.DATA_DIR + x
        t = self.DATA_DIR + t
        x_img = cv2.imread(x, cv2.IMREAD_UNCHANGED) 
        x_img = x_img.astype(np.float32)
        x_img = ((x_img - self.inputMin) / (self.inputMax - self.inputMin)) * 255
        DAPI_img = cv2.imread(getDAPI(x), cv2.IMREAD_UNCHANGED)
        DAPI_img = DAPI_img.astype(np.float32)
        DAPI_img = ((DAPI_img - self.DAPIMin) / (self.DAPIMax - self.DAPIMin)) * 255
        t_img = cv2.imread(t, cv2.IMREAD_UNCHANGED)
        t_img = t_img.astype(np.float32)
        t_img = ((t_img - self.labelMin) / (self.labelMax - self.labelMin)) * 255
        img_dim = x_img.shape[-1]
        dest = np.zeros((2,img_dim,img_dim), dtype=np.float32)
        dest[0,:] = x_img
        dest[1,:] = DAPI_img
        return x, dest, t_img

class DAPIDataset(Dataset):
    """
    Dataset of DAPI/AT8 pTau images. getitem returns DAPI name, DAPI image, and AT8 pTau image
    Used for supplemental analysis, and to assess whether or not we can learn the AT8 pTau channel solely from the DAPI channel
    CSV_FILE should be a csv of x,t pairs corresponding to image names
    x is the YFP-tau image name, and t is the AT8-pTau image name. 
    """
    def __init__(self, csv_file, DATA_DIR):
        self.data = pd.read_csv(csv_file).values
        self.DATA_DIR = DATA_DIR
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        x, t = self.data[idx] 
        x = self.DATA_DIR + x
        t = self.DATA_DIR + t        
        DAPI_img = cv2.imread(getDAPI(x), cv2.IMREAD_UNCHANGED)
        DAPI_img = DAPI_img.astype(np.float32)
        DAPI_img = (DAPI_img / 65535.0) * 255
        t_img = cv2.imread(t, cv2.IMREAD_UNCHANGED)
        t_img = t_img.astype(np.float32)
        t_img = (t_img / 65535.0) * 255
        return getDAPI(x), DAPI_img.reshape(1, DAPI_img.shape[-1], DAPI_img.shape[-1]), t_img

class YFPDataset(Dataset):
    """
    Dataset of YFP-tau/AT8-pTau images. getitem returns YFP-tau name, YFP-tau image, and AT8 pTau image
    Used for supplemental analysis, and to assess learning the AT8 pTau channel solely from the YFP-tau channel
    CSV_FILE should be a csv of x,t pairs corresponding to image names
    x is the YFP-tau image name, and t is the AT8-pTau image name. 
    """
    def __init__(self, csv_file, DATA_DIR):
        self.data = pd.read_csv(csv_file).values
        self.DATA_DIR = DATA_DIR
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        x, t = self.data[idx] 
        x = self.DATA_DIR + x
        t = self.DATA_DIR + t
        x_img = cv2.imread(x, cv2.IMREAD_UNCHANGED) 
        x_img = x_img.astype(np.float32)
        x_img = (x_img / 65535.0) * 255          
        t_img = cv2.imread(t, cv2.IMREAD_UNCHANGED)
        t_img = t_img.astype(np.float32)
        t_img = (t_img / 65535.0) * 255
        return x, x_img.reshape(1, x_img.shape[-1], x_img.shape[-1]), t_img 

class OsteosarcomaDataset(Dataset):
    """
    Image dataset for the osteosarcoma system, uses raw, unablated images
    csv should be in the form d1, d0, such that
    d1 will be the cyclin-B1 label image name and d0 will be the Hoechst input image name
    """
    def __init__(self, csv_file, DATA_DIR):
        self.data = pd.read_csv(csv_file).values
        self.DATA_DIR = DATA_DIR
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        d1, d0 = self.data[idx]
        d1 = self.DATA_DIR + d1
        d0 = self.DATA_DIR + d0
        d0_img = cv2.imread(d0, cv2.IMREAD_UNCHANGED)
        d0_img = d0_img.astype(np.float32) 
        d0_img = (d0_img / 65535.0) * 255
        d1_img = cv2.imread(d1, cv2.IMREAD_UNCHANGED)
        d1_img = d1_img.astype(np.float32)
        d1_img = (d1_img / 65535.0) * 255
        return d0, d0_img.reshape(1,d0_img.shape[-1],d0_img.shape[-1]), d1_img

class OsteosarcomaAblationDataset(Dataset):
    """
    Image Dataset for ablation of the the Hoechst channel in the osteosarcoma dataset 
    csv should be in the form d1, d0, such that
    d1 will be the cyclin-B1 label image name and d0 will be the Hoechst input image name
    """
    def __init__(self, csv_file, DATA_DIR, thresh_percent):
        self.data = pd.read_csv(csv_file).values
        self.DATA_DIR = DATA_DIR
        self.thresh_percent = thresh_percent
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        d1, d0 = self.data[idx]
        d1 = self.DATA_DIR + d1
        d0 = self.DATA_DIR + d0
        d0_img = cv2.imread(d0, cv2.IMREAD_UNCHANGED)
        d0_img = d0_img.astype(np.float32) #breaks here 
        d0_img = (d0_img / 65535.0) * 255
        d1_img = cv2.imread(d1, cv2.IMREAD_UNCHANGED)
        d1_img = d1_img.astype(np.float32)
        d1_img = (d1_img / 65535.0) * 255
        ##ablate bottom x% of pixels to 0
        thresh_index = int(1104 * 1104 * self.thresh_percent)
        sorted_d0 = np.sort(d0_img, axis=None)
        if self.thresh_percent == 1:
            threshold = 10000000
        else:
            threshold = sorted_d0[thresh_index]
        d0_img[d0_img < threshold] = 0
        img_dim = d0_img.shape[-1]
        return d0, d0_img.reshape(1,img_dim,img_dim), d1_img

class Unet_mod(nn.Module):
    """
    The deep learning architecture inspired by Unet (Ronneberger et al. 2015), Motif: upsample, follow by 2x2 conv, concatenate with early layers (skip connections), transpose convolution 
    """
    def __init__(self, inputChannels=2):
        super(Unet_mod, self).__init__()
        self.conv1 = nn.Conv2d(inputChannels, 32, 3) 
        self.conv2 = nn.Conv2d(32, 64, 3)  
        self.conv3 = nn.Conv2d(64, 128, 3)  
        self.conv4 = nn.Conv2d(128, 256, 3)  
        self.conv5 = nn.Conv2d(256, 512, 3) 
        self.conv6 = nn.Conv2d(512, 256, 2) 
        self.conv7 = nn.ConvTranspose2d(256+256, 256, 3)  
        self.conv8 = nn.Conv2d(256, 128, 2) 
        self.conv9 = nn.ConvTranspose2d(128+128, 128, 3) 
        self.conv10 = nn.Conv2d(128, 64, 2)
        self.conv11 = nn.ConvTranspose2d(64+64, 64, 3) 
        self.conv12 = nn.Conv2d(64, 32, 2)
        self.conv13 = nn.ConvTranspose2d(32+32, 1, 3)  
        self.maxp1 = nn.MaxPool2d(2, stride=2, ceil_mode=True, return_indices=True)
        self.maxp2 = nn.MaxPool2d(2, stride=2, ceil_mode=True, return_indices=True)
        self.maxp3 = nn.MaxPool2d(2, stride=2, ceil_mode=True, return_indices=True)
        self.maxp4 = nn.MaxPool2d(2, stride=2, ceil_mode=True, return_indices=True)
        self.upsamp1 = nn.Upsample(size=(254,254),mode='bilinear', align_corners=True) #size = h5 - 1
        self.upsamp2 = nn.Upsample(size=(510,510), mode='bilinear', align_corners=True)#size = h3 - 1
        self.upsamp3 = nn.Upsample(size=(1022,1022), mode='bilinear', align_corners=True) #size = h1 - 1
        self.upsamp4 = nn.Upsample(size=(2047,2047), mode='bilinear', align_corners=True)
        self.transpose1 = nn.ConvTranspose2d(512,512,2) #not used, but needs to be here to load previously saved state dictionaries of older models that left this unused attribute as part of the state dictionary, legacy artifact
        
    def forward(self, x):
        h0 = x.view(x.shape[0], x.shape[1], x.shape[-1], x.shape[-1]) 
        h0 = nn.functional.relu(self.conv1(h0))
        h1, p1indices = self.maxp1(h0) 
        h2 = nn.functional.relu(self.conv2(h1))
        h3, p2indices = self.maxp2(h2) 
        h4 = nn.functional.relu(self.conv3(h3))
        h5, p3indices = self.maxp3(h4) 
        h6 = nn.functional.relu(self.conv4(h5))
        h7, p4indices = self.maxp4(h6) 
        h8 = nn.functional.relu(self.conv5(h7))
        #upsamp, 2x2 conv, 3x3 transposed_conv with skip connections 
        h9 = self.upsamp1(h8) 
        h10 = nn.functional.relu(self.conv6(h9))
        h11 = nn.functional.relu(self.conv7(torch.cat((h6,h10), dim=1)))
        h12 = self.upsamp2(h11)
        h13 = nn.functional.relu(self.conv8(h12))
        h14 = nn.functional.relu(self.conv9(torch.cat((h4,h13), dim=1)))
        h15 = self.upsamp3(h14)
        h16 = nn.functional.relu(self.conv10(h15))
        h17 = nn.functional.relu(self.conv11(torch.cat((h2,h16), dim=1)))
        h18 = self.upsamp4(h17)
        h19 = nn.functional.relu(self.conv12(h18))
        h20 = nn.functional.relu(self.conv13(torch.cat((h0,h19), dim=1))) 
        return h20

class Unet_mod_osteo(nn.Module):
    """
    Same architecture as Unet_mod, but the osteosarcoma training procedure occurred much later than the training for the tauopathy dataset 
    As a result, the architecture was updated to not include the self.transpose attribute that was included in class Unet_mod (included but also not used and had no function - legacy artifact),
    Models for the osteosarcoma dataset were saved without this attribute, so we need this class to load such models
    """
    def __init__(self, inputChannels=1):
        super(Unet_mod_osteo, self).__init__()
        self.conv1 = nn.Conv2d(inputChannels, 32, 3) 
        self.conv2 = nn.Conv2d(32, 64, 3)  
        self.conv3 = nn.Conv2d(64, 128, 3)  
        self.conv4 = nn.Conv2d(128, 256, 3)  
        self.conv5 = nn.Conv2d(256, 512, 3) 
        self.conv6 = nn.Conv2d(512, 256, 2) #
        self.conv7 = nn.ConvTranspose2d(256+256, 256, 3)  
        self.conv8 = nn.Conv2d(256, 128, 2) #
        self.conv9 = nn.ConvTranspose2d(128+128, 128, 3) 
        self.conv10 = nn.Conv2d(128, 64, 2) #
        self.conv11 = nn.ConvTranspose2d(64+64, 64, 3) 
        self.conv12 = nn.Conv2d(64, 32, 2) #
        self.conv13 = nn.ConvTranspose2d(32+32, 1, 3)  
        self.maxp1 = nn.MaxPool2d(2, stride=2, ceil_mode=True, return_indices=True)
        self.maxp2 = nn.MaxPool2d(2, stride=2, ceil_mode=True, return_indices=True)
        self.maxp3 = nn.MaxPool2d(2, stride=2, ceil_mode=True, return_indices=True)
        self.maxp4 = nn.MaxPool2d(2, stride=2, ceil_mode=True, return_indices=True)
        self.upsamp1 = nn.Upsample(size=(136,136),mode='bilinear', align_corners=True) #size = h5 - 1
        self.upsamp2 = nn.Upsample(size=(274,274), mode='bilinear', align_corners=True)#size = h3 - 1
        self.upsamp3 = nn.Upsample(size=(550,550), mode='bilinear', align_corners=True) #size = h1 - 1
        self.upsamp4 = nn.Upsample(size=(1103,1103), mode='bilinear', align_corners=True)

    def forward(self, x):
        h0 = x.view(x.shape[0], x.shape[1], x.shape[-1], x.shape[-1])
        h0 = nn.functional.relu(self.conv1(h0))
        h1, p1indices = self.maxp1(h0) 
        h2 = nn.functional.relu(self.conv2(h1))
        h3, p2indices = self.maxp2(h2)
        h4 = nn.functional.relu(self.conv3(h3))
        h5, p3indices = self.maxp3(h4) 
        h6 = nn.functional.relu(self.conv4(h5))
        h7, p4indices = self.maxp4(h6) 
        h8 = nn.functional.relu(self.conv5(h7))
        #upsamp, 2x2 conv, 3x3 transposed_conv with stitch 
        h9 = self.upsamp1(h8) 
        h10 = nn.functional.relu(self.conv6(h9))
        h11 = nn.functional.relu(self.conv7(torch.cat((h6,h10), dim=1)))
        h12 = self.upsamp2(h11)
        h13 = nn.functional.relu(self.conv8(h12))
        h14 = nn.functional.relu(self.conv9(torch.cat((h4,h13), dim=1)))
        h15 = self.upsamp3(h14)
        h16 = nn.functional.relu(self.conv10(h15))
        h17 = nn.functional.relu(self.conv11(torch.cat((h2,h16), dim=1)))
        h18 = self.upsamp4(h17)
        h19 = nn.functional.relu(self.conv12(h18))
        h20 = nn.functional.relu(self.conv13(torch.cat((h0,h19), dim=1))) #127 x 127
        return h20

def train(continue_training=False, load_training_name="None", model=None, max_epochs=20, training_generator=None, validation_generator=None, lossfn=None, optimizer=None, plotName=None, device=None):
    """
    Method to train and save a model as a .pt object
    Saves the model every epoch
    CONTINUE_TRAINING indicates whether or not we should load a MODEL as a warm start for training
    MAX_EPOCHS indicates the number of epochs we should use for training
    TRAINING_GENERATOR and VALIDATION_GENERATOR are the training and validation generators
    LOSSFN is our loss function
    OPTIMIZER is the optimizer function
    PLOTNAME is the string that we will save our model name with 
    DEVICE is the cuda device to allocate  
    Saves the trained model to directory "models/"
    """
    if continue_training: ##flag to use if we are continuing to train a previously trained model
        checkpoint = torch.load(load_training_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        model.train()
    start_time = time.time()
    total_step = len(training_generator)
    for epoch in range(max_epochs):
        model.train()
        i = 0 
        running_loss = 0
        for names, local_batch, local_labels in training_generator:
            i += 1
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            outputs = model(local_batch)
            loss = lossfn(outputs, local_labels)
            running_loss += loss             
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("epoch: ", epoch + 1, "avg training loss over all batches: ", running_loss / float(i), ", time elapsed: ", time.time() - start_time)
        ##save model and check validation loss every epoch
        saveTraining(model, epoch, optimizer, running_loss / float(i), "models/" + plotName + ".pt") 
        running_val_loss = 0
        j, k = 0, 0
        with torch.set_grad_enabled(False):
            for names, local_batch, local_labels in validation_generator:
                j += 1
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                outputs = model(local_batch)
                loss =  lossfn(outputs, local_labels)
                running_val_loss += loss 
        print("epoch: ", epoch + 1, " global validation loss: ", running_val_loss / float(j))
    saveTraining(model, epoch, optimizer, running_loss / float(j), "models/" + plotName + ".pt")

def test(sample_size=1000000, model=None, loadName=None, validation_generator=None, lossfn=None, device=None):
    """
    Method to run the MODEL on the test set over SAMPLE_SIZE number of images
    MODEL should be of type nn.Module, and specifies the architecture to use
    LOADNAME is the name of the model to load 
    VALIDATION_GENERATOR is the generator that iterates over validation/test images
    LOSSFN is the loss function to use
    DEVICE is the cuda device to allocate to  
    Returns and prints the pearson performance and mse performance of both the ML Model and also the Null Model (0th channel)
    """
    start_time = time.time()
    checkpoint = torch.load(loadName)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    model.eval()
    performance, null_performance, mse_performance, null_mse_performance, pearson_inp_predicted = [], [], [], [], []
    total_step = len(validation_generator)
    j = 0
    running_loss = 0
    with torch.set_grad_enabled(False):
        for names, local_batch, local_labels in validation_generator:
            j += 1
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            outputs = model(local_batch)
            loss = lossfn(outputs, local_labels)
            running_loss += loss 
            pearson = getPearson(outputs, local_labels)
            performance.append(pearson) 
            mse_performance.append(calculateMSE(outputs, local_labels))
            null_performance.append(getPearson(local_batch[:, 0, :, :], local_labels))
            null_mse_performance.append(calculateMSE(local_batch[:, 0, :, :], local_labels))
            pearson_inp_predicted.append(getPearson(local_batch[:, 0, :, :], outputs))
            if j == sample_size:
                break
    ml_model_perf = np.mean(performance), np.std(performance)
    null_model_perf = np.mean(null_performance), np.std(null_performance)
    ml_model_mse_perf = np.mean(mse_performance), np.std(mse_performance)
    null_model_mse_perf = np.mean(null_mse_performance), np.std(null_mse_performance)
    print("time elapsed: ", time.time() - start_time)
    print("global ML pearson performance: ", ml_model_perf)
    print("global null pearson performance (0th channel): ", null_model_perf) 
    print("global MSE performance: ", ml_model_mse_perf)
    print("global null MSE performance (0th channel): ", null_model_mse_perf)
    print("pearson correlation between input and predicted image: ", np.mean(pearson_inp_predicted), np.std(pearson_inp_predicted))
    print("global loss: ", running_loss / float(j))
    return ml_model_perf, null_model_perf, ml_model_mse_perf, null_model_mse_perf

def testOnSeparatePlates(sample_size, model=None, loadName=None, validation_generator=None, lossfn=None, device=None):
    """
    Runs model evaluation and calculates pearson by plate (each plate had a unique drug condition), given SAMPLE_SIZE number of images 
    Prints and pickles the results
    """
    plate_pearsons = {i: [] for i in range(1,7)} #key plate number, value: list of pearson correlations
    null_YFP_pearsons = {i: [] for i in range(1,7)}
    null_DAPI_pearsons = {i: [] for i in range(1,7)}
    checkpoint = torch.load(loadName)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    model.eval()
    performance = []
    total_step = len(validation_generator)
    j = 0
    running_loss = 0
    with torch.set_grad_enabled(False):
        for names, local_batch, local_labels in validation_generator:
            plateNumber = int(names[0][names[0].find("plate") + 5 :names[0].find("plate") + 6]) ##given plate number <= 9! our study used plates 1-6
            assert(plateNumber < 7)
            j += 1
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            outputs = model(local_batch)       
            loss =  lossfn(outputs, local_labels)
            running_loss += loss 
            pearson = getPearson(outputs, local_labels)
            performance.append(pearson) 
            plate_pearsons[plateNumber].append(pearson)
            null_YFP_pearson = getPearson(local_batch[:,0,:,:], local_labels)
            null_YFP_pearsons[plateNumber].append(null_YFP_pearson)
            null_DAPI_pearson = getPearson(local_batch[:,1,:,:], local_labels)
            null_DAPI_pearsons[plateNumber].append(null_DAPI_pearson)
            if j > sample_size:
                break
    ##pickle and print results
    model_performances, null_YFP_performances, null_DAPI_performances = [], [], []
    model_stds, null_YFP_stds, null_DAPI_stds = [], [], []
    for i in plate_pearsons:
        print("plate {} average pearson test performance: {}, std: {}".format(i, np.mean(plate_pearsons[i]), np.std(plate_pearsons[i])))
        print("    plate {} YFP performance: {}, std: {}; DAPI performance: {}, std: {}".format(i, np.mean(null_YFP_pearsons[i]), np.std(null_YFP_pearsons[i]), np.mean(null_DAPI_pearsons[i]), np.std(null_DAPI_pearsons[i])))
        model_performances.append(np.mean(plate_pearsons[i]))
        model_stds.append(np.std(plate_pearsons[i]))
        null_YFP_performances.append(np.mean(null_YFP_pearsons[i]))
        null_YFP_stds.append(np.std(null_YFP_pearsons[i]))
        null_DAPI_performances.append(np.mean(null_DAPI_pearsons[i]))
        null_DAPI_stds.append(np.std(null_DAPI_pearsons[i]))
    pickle.dump(model_performances, open("pickles/separatePlateTestModelPerformances.pkl", "wb"))
    pickle.dump(model_stds, open("pickles/separatePlateTestModelStds.pkl", "wb"))
    pickle.dump(null_YFP_performances, open("pickles/separatePlateTestYFPPerformances.pkl", "wb"))
    pickle.dump(null_YFP_stds, open("pickles/separatePlateTestYFPStds.pkl", "wb"))
    pickle.dump(null_DAPI_performances, open("pickles/separatePlateTestDAPIPerformances.pkl", "wb"))
    pickle.dump(null_DAPI_stds, open("pickles/separatePlateTestDAPIStds.pkl", "wb"))

def getROC(lab_thresh, sample_size, model=None, loadName=None, validation_generator=None, fold=-1, device=None):
    """
    LAB_THRESH is the pixel threshold that we use to binarize our label image 
    SAMPLE_SIZE specifies how many images we want to include in this analysis
    MODEL is the model of type NN.Module (a trained model specified by the name within this function will be loaded)
    LOADNAME is the name of the model to load
    VALIDATION_GENERATOR is the generator that iterates over validation/test images to evaluate
    FOLD specifies the fold of the cross validation (or if no cross validation, then specify fold != 1,2, or 3)
    DEVICE is the cuda device to allocate to 
    Saves ROC curves for YFP Null Model, DAPI Null Model, and the actual ML model, 
    Pickles and returns the coordinates of the different curves, and saves to pickles/
    """
    checkpoint = torch.load(loadName)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    mapp = {} #will correspond to ML model results
    null_mapp = {} #will correspond to Null YFP model results
    null_DAPI_mapp = {} #will correspond to Null DAPI model results
    threshs = list(np.arange(-1, -.1, .1)) #the various thresholds that we will use to binarize our predicted label images
    threshs += list(np.arange(-.1, .1, .01))
    threshs +=  list(np.arange(.1, 1, .1)) 
    threshs +=  list(np.arange(1, 4, 1)) 
    threshs.append(30) 
    threshs.append(1000)
    for t in threshs:
        mapp[t] = [] #thresh to list of (FPR, TPR) points
        null_mapp[t] = []
        null_DAPI_mapp[t] = []
    with torch.set_grad_enabled(False):
        j = 0
        for names, local_batch, local_labels in validation_generator:
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            outputs = model(local_batch)
            for t in threshs:
                TPR, TNR, PPV, NPV, FNR, FPR = getMetrics(outputs, local_labels, lab_thresh=lab_thresh, pred_thresh=t)
                null_TPR, null_TNR, null_PPV, null_NPV, null_FNR, null_FPR = getMetrics(local_batch[:,0,:,:], local_labels, lab_thresh=lab_thresh, pred_thresh=t)
                null_DAPI_TPR, null_DAPI_TNR, null_DAPI_PPV, null_DAPI_NPV, null_DAPI_FNR, null_DAPI_FPR = getMetrics(local_batch[:,1,:,:], local_labels, lab_thresh=lab_thresh, pred_thresh=t) #for NULL AUC
                mapp[t].append((FPR, TPR))
                null_mapp[t].append((null_FPR, null_TPR))
                null_DAPI_mapp[t].append((null_DAPI_FPR, null_DAPI_TPR))
            j += 1
            if j > sample_size:
                break
            print(j)
        ##generate ROC curves data, one per model
        i = 0
        for m in [null_mapp, null_DAPI_mapp, mapp]:
            coordinates = [] #list of tuples (thresh, FPR, TPR) to plot
            for key in m:
                FPRs = [l[0] for l in m[key]]
                TPRs = [l[1] for l in m[key]]
                coordinates.append((key, np.mean(FPRs), np.mean(TPRs)))
            coordinates = sorted(coordinates, key=lambda x: x[0])
            labels = [t[0] for t in coordinates] 
            x = [t[1] for t in coordinates]
            y = [t[2] for t in coordinates]
            if i == 0:
                pickle.dump(x, open("pickles/null_YFP_mapp_x_values_fold_{}.pk".format(fold), "wb"))
                pickle.dump(y, open("pickles/null_YFP_mapp_y_values_fold_{}.pk".format(fold),  "wb"))
                null_YFP_x, null_YFP_y = x, y
            if i == 1:
                pickle.dump(x, open("pickles/null_DAPI_mapp_x_values_fold_{}.pk".format(fold), "wb"))
                pickle.dump(y, open("pickles/null_DAPI_mapp_y_values_fold_{}.pk".format(fold), "wb"))
                null_DAPI_x, null_DAPI_y = x, y
            if i == 2:
                pickle.dump(x, open("pickles/mapp_x_values_fold_{}.pk".format(fold), "wb"))
                pickle.dump(y, open("pickles/mapp_y_values_fold_{}.pk".format(fold), "wb"))
                ML_x, ML_y = x, y
            i += 1
            labels, x, y = labels[::-1], x[::-1], y[::-1]
            auc = np.trapz(y,x)
            print("AUC: ", auc)
        return ML_x, ML_y, null_YFP_x, null_YFP_y, null_DAPI_x, null_DAPI_y

def osteosarcomaAblatedAndNonAblated(sample_size=100, validation_generator=None, model=None, fold=None, device=None):
    """
    Method to load the model trained on ablated images and the model trained on raw images
    Runs through the test set, and saves example images to outputs/ directory 
    SAMPLE_SIZE specifies the number of images to run through, VALIDATION_GENERATOR is the validation generator, MODEL is the architeture to use, FOLD is the cross-validation fold, DEVICE is the cuda device 
    """
    ablated_model = Unet_mod_osteo(inputChannels=1)
    ablated_model = ablated_model.to(device)
    checkpoint  = torch.load("models/d0_to_d1_ablation_cyclin_only_dataset_fold{}_continue_training.pt".format(fold), map_location='cuda:0') 
    ablated_model.load_state_dict(checkpoint['model_state_dict'])
    unablated_model = Unet_mod_osteo(inputChannels=1)
    unablated_model = unablated_model.to(device)
    checkpoint2  = torch.load("models/d0_to_d1_cyclin_only_dataset_fold{}.pt".format(fold), map_location='cuda:0') 
    unablated_model.load_state_dict(checkpoint2['model_state_dict'])
    i = 0
    with torch.set_grad_enabled(False):
        for names, local_batch, local_labels in validation_generator:
            rand = np.random.randint(0,100000000)
            img_dim = local_batch.shape[-1]
            ##plot raw Hoechst, raw cyclinB1 first
            cv2.imwrite("outputs/" + str(rand) + "_a_raw_Hoechst.tif", ((local_batch.cpu().numpy().reshape(img_dim, img_dim) / 255) * 65535).astype(np.uint16))
            cv2.imwrite("outputs/" + str(rand) + "_c_cyclinB1_label.tif", ((local_labels.cpu().numpy().reshape(img_dim, img_dim) / 255) * 65535).astype(np.uint16))
            ##use unablated model to predict with raw image 
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            unablated_outputs = unablated_model(local_batch)
            pearson = getPearson(unablated_outputs, local_labels)
            ##plot raw prediction of cyclinB1
            cv2.imwrite("outputs/" + str(rand) + "_d_predicted_raw_pearson={}.tif".format(pearson), ((unablated_outputs.cpu().numpy().reshape(img_dim, img_dim) / 255) * 65535).astype(np.uint16))
            ##ablate bottom 95% of pixels to 0
            a = .95
            local_batch = local_batch.cpu().numpy()
            img_dim = local_batch.shape[-1]
            local_batch = local_batch.reshape((img_dim, img_dim))
            thresh_index = int(1104 * 1104 * a)
            sorted_local_batch = np.sort(local_batch, axis=None)
            if a == 1:
                threshold = 10000000
            else:
                threshold = sorted_local_batch[thresh_index]
            local_batch[local_batch < threshold] = 0
            ##plot ablated Hoechst
            cv2.imwrite("outputs/" + str(rand) + "_b_ablated_Hoechst.tif", ((local_batch / 255) * 65535).astype(np.uint16))
            ##convert back to tensor
            local_batch = local_batch.reshape((1, 1, img_dim, img_dim))
            local_batch = torch.from_numpy(local_batch).float().to(device)
            ablated_outputs = ablated_model(local_batch)
            pearson = getPearson(ablated_outputs, local_labels)
            ##plot ablated prediction of cyclinB1
            cv2.imwrite("outputs/" + str(rand) + "_e_predicted_ablated_pearson={}.tif".format(pearson), ((ablated_outputs.cpu().numpy().reshape(img_dim, img_dim) / 255) * 65535).astype(np.uint16))
            i += 1
            if i >= sample_size:
                break

#============================================================================
#============================================================================
## HELPER FUNCTIONS
#============================================================================
#============================================================================

def calculateMSE(predicted, actual):
    """
    Helper function that calculates the MSE between PREDICTED and ACTUAL, performing a local normalization to 0 mean and unit variance
    returns the MSE
    """
    predicted = predicted.cpu().numpy()
    actual = actual.cpu().numpy()
    img_dim = actual.shape[-1]
    predicted = predicted.reshape((img_dim, img_dim))
    actual = actual.reshape((img_dim, img_dim))
    lmean, lstd = np.mean(actual), np.std(actual)
    pmean, pstd = np.mean(predicted), np.std(predicted)
    actual = ((actual - lmean) /float(lstd))
    predicted = ((predicted - pmean) /float(pstd))
    mse = np.average(np.square(predicted - actual))
    return mse

def pearsonCorrLoss(outputs, targets):
    """
    Calculates and returns the negative pearson correlation loss
    """
    vx = outputs - torch.mean(outputs)
    vy = targets - torch.mean(targets)
    cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return -1 * cost

def saveTraining(model, epoch, optimizer, loss, PATH):
    """
    Helper function to save MODEL, EPOCH, OPTIMIZER, and LOSS, as pytorch file to PATH
    """
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }, PATH)

def getPearson(predicted, labels):
    """
    Calculates and returns the average pearson correlation between tensors PREDICTED and LABELS (potentially containing multiple images)
    """
    labels = labels.cpu().numpy()
    predicted = predicted.cpu().numpy()
    pearsons = []
    for i in range(0, len(predicted)):
        corr = np.corrcoef(labels[i].flatten(), predicted[i].flatten())[0][1]
        pearsons.append(corr)
    return np.mean(pearsons) 

def getDAPI(filename):
    """
    Simple string manipulation helper method to replace "FITC" with "DAPI" and "Blue" with "UV" of input FILENAME,
    Returns the modified string
    """
    DAPI = filename.replace("FITC", "DAPI")
    DAPI = DAPI.replace("Blue", "UV")
    return DAPI

def plotInputs(inputs, labels, predicted, directory, rand=None):
    """
    Helper method to visually plot the INPUTS, LABELS, and PREDICTED (all of type CUDA), and plot in images in DIRECTORY
    Writes corresponding images with a random identifier
    """ 
    inputs = inputs.cpu().numpy()
    labels = labels.cpu().numpy()
    predicted = predicted.cpu().numpy()
    print(inputs.shape)
    pearsons = []
    for i in range(0, len(inputs)):
        if rand is None:
            rand = np.random.randint(0,10000000000)
        img_dim = inputs.shape[-1]
        inp = inputs[i][0].reshape((img_dim,img_dim))       
        inp = (inp / 255) * 65535.0
        inp = inp.astype(np.uint16)
        cv2.imwrite(directory + str(rand) + "input_0.tif", inp)
        if inputs.shape[1] == 2: 
            dapi = inputs[i][1].reshape((img_dim, img_dim))
            dapi = (dapi / 255) * inputMax
            dapi = dapi.astype(np.uint16)
            cv2.imwrite(directory + str(rand) + "input_1.tif", dapi)
        lab = labels[i].reshape((img_dim,img_dim))
        lab = (lab / 255) * 65535.0
        lab = lab.astype(np.uint16)
        cv2.imwrite(directory + str(rand) + "label_AT8.tif", lab)
        pred = predicted[i].reshape((img_dim,img_dim))
        pred = (pred / 255) * 65535.0
        pred = pred.astype(np.uint16)
        corr = np.corrcoef(lab.flatten(), pred.flatten())[0][1] #pearson correlation after transformations (pearson correlations of raw images are reported in study)
        cv2.imwrite(directory + str(rand) + "predicted_AT8_pearson=" + str(corr)[0:5] + ".tif", pred)

def getMetrics(predicted, labels, lab_thresh=None, pred_thresh=None):
    """
    Given image tensors PREDICTED and LABELS, a LAB_THRESH threshold to binarize LABELS, and a PRED_THRESH threshold to binarize the predicted image
    Returns TPR, TNR, PPV, NPV, FNR, FPR for batch of images
    """
    predicted = predicted.cpu().numpy()
    labels = labels.cpu().numpy()
    img_dim = labels.shape[-1]
    ##put images back in original space (0 to 65535 pixel value)
    lab = labels[0].reshape((img_dim,img_dim))
    pred = predicted[0].reshape((img_dim,img_dim))
    lab = (lab / 255) * 65535.0  
    pred = (pred / 255) * 65535.0
    ##normalize image to have mean = 0, variance = 1
    lmean, lstd = np.mean(lab), np.std(lab)
    pmean, pstd = np.mean(pred), np.std(pred)
    lab = ((lab - lmean) /float(lstd))
    pred = ((pred - pmean) /float(pstd))
    true_positive = np.sum(np.where((pred >= pred_thresh) & (lab >= lab_thresh), 1, 0))
    true_negative = np.sum(np.where((pred < pred_thresh) & (lab < lab_thresh), 1, 0))
    false_positive = np.sum(np.where((pred >= pred_thresh) & (lab < lab_thresh), 1, 0))
    false_negative = np.sum(np.where((pred < pred_thresh) & (lab >= lab_thresh), 1, 0))  
    TPR = true_positive / float(true_positive + false_negative )
    TNR = true_negative / float(true_negative + false_positive )
    PPV = true_positive / float(true_positive + false_positive )
    NPV = true_negative / float(true_negative + false_negative )
    FNR = false_negative / float(false_negative + true_positive )
    FPR = false_positive / float(false_positive + true_negative ) 
    return TPR, TNR, PPV, NPV, FNR, FPR

def findStats(validation_generator, device):
    """
    finds the average mean and std of inputs (first channel) and labels over the test set of images for the tauopathy dataset
    """
    start_time = time.time()
    total_step = len(validation_generator)
    print("test size: ", total_step)
    j = 0
    input_stats = [] #list of (mean, std)
    label_stats = [] 
    with torch.set_grad_enabled(False):
        for names, local_batch, local_labels in validation_generator:
            j += 1
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            local_batch = local_batch.cpu().numpy()
            local_labels = local_labels.cpu().numpy()
            for i in range(0, len(local_batch)):
                img = local_batch[i][0]
                img = (img / float(255)) * 65535.0
                label = local_labels[i]
                label = (label / float(255)) * 65535.0
                img = img.reshape((2048,2048))
                label = label.reshape((2048,2048))
                input_stats.append((np.mean(img), np.std(img)))
                label_stats.append((np.mean(label), np.std(label)))
        input_stats = [sum(y) / len(y) for y in zip(*input_stats)]
        label_stats = [sum(y) / len(y) for y in zip(*label_stats)]
        print("input stats: ", input_stats)
        print("label stats: ", label_stats)
        print("time elapsed: ", time.time() - start_time)

#============================================================================
#============================================================================
## SUPPLEMENTAL CODE
#============================================================================
#============================================================================

def getOverlap(sample_size=1000000, generator=None):
    """
    SAMPLE_SIZE indicates how many image example pairs to calculate overlap, GENERATOR specifies the data generator
    Finds the percentage of signal that is present in both YFP input and pTau output
    Divided by the total signal of pTau
    Method to show that essentially all of pTau signal is also present in YFP channel (to some permissive threshold) 
    Pickles the results of overlap at each threshold to pickles/ directory  
    """
    overlaps = []
    start_time = time.time()
    label_thresh = 1 
    input_threshs = [1, .75, .5, .25, 0, -.25, -.5, -.75, -1] #anything over input_thresh will be considered positive signal
    for input_thresh in input_threshs:
        fractions = []
        j = 0
        with torch.set_grad_enabled(False):
            for names, local_batch, local_labels in generator:
                j += 1
                local_batch = local_batch.cpu().numpy()
                local_labels = local_labels.cpu().numpy()
                for i in range(0, len(local_batch)):
                    img = local_batch[i][0]
                    img = (img / float(255)) * 65535.0
                    label = local_labels[i]
                    label = (label / float(255)) * 65535.0
                    img = img.reshape((2048,2048))
                    label = label.reshape((2048,2048))
                    lmean, lstd = np.mean(label), np.std(label)
                    imean, istd = np.mean(img), np.std(img)
                    label = ((label - lmean) /float(lstd))
                    img = ((img - imean) /float(istd))
                    img_count = np.sum(np.where((img >= input_thresh), 1, 0))
                    label_count = np.sum(np.where((label >= label_thresh), 1, 0))
                    overlap = np.sum(np.where((label >= label_thresh) & (img >= input_thresh), 1, 0))
                    fraction = overlap / float(label_count)
                    fractions.append(fraction)
                if j == sample_size:
                    break
        overlap_avg = np.mean(fractions)
        print("inp thresh: ", input_thresh, "label thresh: ", label_thresh, "average overlap: ",overlap_avg)
        overlaps.append(overlap_avg)
    pickle.dump(input_threshs, open("pickles/input_threshs_for_overlap.pkl", "wb"))
    pickle.dump(overlaps, open("pickles/overlaps.pkl", "wb"))

class TauAblationDataset(Dataset):
    """
    Dataset in which we ablate the bottom x% of pixels to be 0 valued
    This is to test if image bleed through is an issue
    csv_file should be a csv of x,t pairs of full path strings, pointing to images
    x is the YFP-tau image name, and t is the AT8-pTau image name
    """
    def __init__(self, csv_file, thresh_percent):
        self.data = pd.read_csv(csv_file).values
        self.thresh_percent = thresh_percent

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, t = self.data[idx]
        x_img = cv2.imread(x, cv2.IMREAD_UNCHANGED) 
        x_img = x_img.astype(np.float32)
        x_img = ((x_img - inputMin) / (inputMax - inputMin)) * 255
        DAPI_img = cv2.imread(getDAPI(x), cv2.IMREAD_UNCHANGED)
        DAPI_img = DAPI_img.astype(np.float32)
        DAPI_img = ((DAPI_img - DAPIMin) / (DAPIMax - DAPIMin)) * 255
        t_img = cv2.imread(t, cv2.IMREAD_UNCHANGED)
        t_img = t_img.astype(np.float32)
        t_img = ((t_img - labelMin) / (labelMax - labelMin)) * 255
        ##ablate bottom x% of pixels to 0
        thresh_index = int(2048 * 2048 * self.thresh_percent)
        sorted_x = np.sort(x_img, axis=None) #sorted in increasing order
        sorted_dapi = np.sort(DAPI_img, axis=None)
        if self.thresh_percent == 1:
            threshold_x = 10000000
            threshold_dapi = 10000000
        else:
            threshold_x = sorted_x[thresh_index]
            threshold_dapi = sorted_dapi[thresh_index]
        #threshold to 0
        x_img[x_img < threshold_x] = 0 
        DAPI_img[DAPI_img < threshold_dapi] = 0
        dest = np.zeros((2,img_dim,img_dim), dtype=np.float32)
        dest[0,:] = x_img
        dest[1,:] = DAPI_img
        return dest, t_img

def ablationTestTau(sample_size=1000000, validation_generator=None, ablate_DAPI_only=False, model=None, loadName=None, device=None):
    """
    SAMPLE_SIZE is the number of images to evaluate from the VALIDATION_GENERATOR
    VALIDATION_GENERATOR is the generator that iterates over validation/test images to evaluate
    If ABLATE_DAPI_ONLY, then will only ablate the DAPI input and leave YFP-tau alone 
    MODEL specifies which architecture to use
    LOADNAME is the name of the model to load and evaluate 
    DEVICE is the cuda device to allocate to
    Method to run the model on the tauopathy test set and print pearson performance under a range of ablation conditions
    pickles the results
    """
    ablations = list(np.arange(0, .1, .02))
    ablations += list(np.arange(.1, 1.1, .1))
    x = []
    y = []
    stds = []
    start_time = time.time()
    checkpoint = torch.load(loadName)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    for a in ablations:
        performance = []
        total_step = len(validation_generator)
        j = 0
        with torch.set_grad_enabled(False):
            for names, local_batch, local_labels in validation_generator:
                j += 1   
                local_batch = local_batch.cpu().numpy()
                img_dim = local_batch.shape[-1]
                local_batch = local_batch.reshape((local_batch.shape[1],img_dim, img_dim)) ##local_batch.shape[1] is number of channels
                YFP = local_batch[0]
                DAPI = local_batch[1]
                thresh_index = int(img_dim * img_dim * a)
                sorted_local_YFP = np.sort(YFP, axis=None)
                sorted_local_DAPI = np.sort(DAPI, axis=None)
                if a == 1:
                    YFP_threshold = 10000000
                    DAPI_threshold = 10000000
                else:
                    YFP_threshold = sorted_local_YFP[thresh_index]
                    DAPI_threshold = sorted_local_DAPI[thresh_index]
                ##ablate both DAPI and YFP tau
                if ablate_DAPI_only == False: 
                    YFP[YFP < YFP_threshold] = 0
                DAPI[DAPI < DAPI_threshold] = 0
                local_batch = np.zeros((2,img_dim,img_dim), dtype=np.float32)
                local_batch[0,:] = YFP
                local_batch[1,:] = DAPI
                local_batch = torch.from_numpy(local_batch).float()
                local_batch = local_batch.reshape((1, 2, img_dim, img_dim))
                local_batch = local_batch.to(device)
                local_labels = local_labels.to(device)
                outputs = model(local_batch)
                pearson = getPearson(outputs, local_labels)            
                if j == sample_size:
                    break
                performance.append(pearson) 
        x.append(a)
        y.append(np.mean(performance))
        stds.append(np.std(performance))
        pickle.dump(x, open("pickles/ablation_tau_x.pkl", "wb"))
        pickle.dump(y, open("pickles/ablation_tau_y.pkl", "wb"))
        pickle.dump(stds, open("pickles/ablation_tau_stds.pkl", "wb"))
        print("ablation: {}, global performance: avg={}, std= {}".format(a, np.mean(performance), np.std(performance)))
    

