"""
We include here all of the necessary code to reproduce the main training and model evaluation results from the tau dataset
This script is divided into five parts:
1) class and method definitions
2) helper functions
3) supplemental code for ablation analysis
Any questions should be directed to daniel.wong2@ucsf.edu. Thank you!
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
    Dataset of tau images. CSV_FILE should be a csv of x,t pairs of full-path strings corresponding to image names.
    x is the YFP-tau image name, and t is the AT8-pTau image name. 
    """
    def __init__(self, csv_file, inputMin, inputMax, DAPIMin, DAPIMax, labelMin, labelMax,img_dim):
        self.data = pd.read_csv(csv_file).values
        self.inputMin = inputMin
        self.inputMax = inputMax
        self.DAPIMin = DAPIMin
        self.DAPIMax = DAPIMax
        self.labelMin = labelMin
        self.labelMax = labelMax
        self.img_dim = img_dim
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x, t = self.data[idx] 
        if "keiserlab" not in hostname: #if on butte lab server 
            x = x.replace("/fast/disk0/dwong/", "/data1/wongd/")
            t = t.replace("/fast/disk0/dwong/", "/data1/wongd/")
            x = x.replace("/srv/nas/mk3/users/dwong/", "/data1/wongd/") 
            t = t.replace("/srv/nas/mk3/users/dwong/", "/data1/wongd/")
        if "keiserlab" in hostname: 
            x = x.replace("/data1/wongd/", "/srv/nas/mk3/users/dwong/")
            t = t.replace("/data1/wongd/", "/srv/nas/mk3/users/dwong/")
            x = x.replace("/fast/disk0/dwong/", "/srv/nas/mk3/users/dwong/")
            t = t.replace("/fast/disk0/dwong/", "/srv/nas/mk3/users/dwong/")
        x_img = cv2.imread(x, cv2.IMREAD_UNCHANGED) 
        x_img = x_img.astype(np.float32)
        x_img = ((x_img - self.inputMin) / (self.inputMax - self.inputMin)) * 255
        DAPI_img = cv2.imread(getDAPI(x), cv2.IMREAD_UNCHANGED)
        DAPI_img = DAPI_img.astype(np.float32)
        DAPI_img = ((DAPI_img - self.DAPIMin) / (self.DAPIMax - self.DAPIMin)) * 255
        t_img = cv2.imread(t, cv2.IMREAD_UNCHANGED)
        t_img = t_img.astype(np.float32)
        t_img = ((t_img - self.labelMin) / (self.labelMax - self.labelMin)) * 255
        dest = np.zeros((2,self.img_dim,self.img_dim), dtype=np.float32)
        dest[0,:] = x_img
        dest[1,:] = DAPI_img
        return dest, t_img

class Unet_mod(nn.Module):
    """
    The deep learning architecture inspired by Unet (Ronneberger et al. 2015), Motif: upsample, follow by 2x2 conv, concatenate with early layers (skip connections), transpose convolution 
    """
    def __init__(self, img_dim):
        super(Unet_mod, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, 3) 
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
        self.transpose1 = nn.ConvTranspose2d(512,512,2)
        self.img_dim=img_dim
        
    def forward(self, x):
        h0 = x.view(x.shape[0], x.shape[1], self.img_dim, self.img_dim)
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

def train(continue_training=False, model=None, max_epochs=20, training_generator=None, validation_generator=None, lossfn=None, optimizer=None, plotName=None, device=None):
    """
    Method to train and save a model as a .pt object, saves the model every 5 epochs
    CONTINUE_TRAINING indicates whether or not we should load a MODEL as a warm start for training
    MAX_EPOCHS indicates the number of epochs we should use for training
    TRAINING_GENERATOR and VALIDATION_GENERATOR are the training and validation generators to use
    LOSSFN is our loss function to use
    OPTIMIZER is the optimizer function to use 
    PLOTNAME is the string that we will save our model name with 
    DEVICE is the cuda device to allocate to  
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
        for local_batch, local_labels in training_generator:
            i += 1
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            outputs = model(local_batch)
            loss = lossfn(outputs, local_labels)
            running_loss += loss 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("epoch: ", epoch + 1, "avg training loss over all batches: ", running_loss / float(i), ", time elapsed: ", time.time() - start_time)
        ##save model and check validation loss every 5 epochs
        if epoch % 5 == 0:
            saveTraining(model, epoch, optimizer, running_loss / float(i), "models/" + plotName + ".pt") 
            running_val_loss = 0
            j = 0
            with torch.set_grad_enabled(False):
                for local_batch, local_labels in validation_generator:
                    j += 1
                    local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                    outputs = model(local_batch)
                    loss =  lossfn(outputs, local_labels)
                    running_val_loss += loss 
            print("epoch: ", epoch + 1, " global validation loss: ", running_val_loss / float(j))
    saveTraining(model, epoch, optimizer, running_loss / float(j), "models/" + plotName + ".pt")

def test(sample_size=1000000, model=None, loadName=None, validation_generator=None, lossfn=None, device=None):
    """
    Method to run the model on the test set over SAMPLE_SIZE number of images
    MODEL should be of type nn.Module, and specifies the architecture to use
    LOADNAME is the name of the model to load 
    CROSS_VAL indicates if we're testing under a cross-validation scheme
    VALIDATION_GENERATOR is the generator that iterates over validation/test images
    LOSSFN is the loss function to use
    DEVICE is the cuda device to allocate to  
    Print the pearson performance, and pickle the result to pickles/
    """
    start_time = time.time()
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
        for local_batch, local_labels in validation_generator:
            j += 1
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            outputs = model(local_batch)
            loss = lossfn(outputs, local_labels)
            running_loss += loss 
            pearson = getPearson(outputs, local_labels)
            print(str(j) + "/" + str(total_step) + ", batch pearson: ", pearson)
            if j == sample_size:
                break
            performance.append(pearson) 
    print("time elapsed: ", time.time() - start_time)
    print("global performance: ", np.mean(performance), np.std(performance))
    print("global loss: ", running_loss / float(j))
    pickle.dump((np.mean(performance), np.std(performance)), open("pickles/ML_model_pearson_performance.pkl", "wb"))

def getMSE(loadName=None, model=None, validation_generator=None, device=None):
    """
    LOADNAME specifies the name of the model to load
    MODEL should be of type nn.Module, and specifies the architecture to use 
    VALIDATION_GENERATOR is the generator that iterates over validation/test images
    calculates the MSE of ML predictions, pickles the results to pickles/
    DEVICE is the cuda device to allocate to 
    """
    checkpoint = torch.load(loadName)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    performance = []
    total_step = len(validation_generator)
    with torch.set_grad_enabled(False):
        for local_batch, local_labels in validation_generator:
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            outputs = model(local_batch)
            mse = calculateMSE(outputs, local_labels)
            performance.append(mse)
            print(mse)
    print("global peformance (mse): ", np.mean(performance), np.std(performance))
    pickle.dump((np.mean(performance), np.std(performance)), open("pickles/ML_model_MSE_performance.pkl", "wb"))

def getNull(validation_generator=None, device=None):
    """
    VALIDATION_GENERATOR is the generator that iterates over validation/test images
    DEVICE is the cuda device to allocate to
    Calculates the null Pearson and MSE performance of the dataset (i.e. pearson(input, label)) comparing YFP to AT8, and comparing DAPI to AT8
    Pickles the results to pickles/
    """
    j = 0
    YFP_pearson_performance = []
    YFP_MSE_performance = []
    DAPI_pearson_performance = []
    DAPI_MSE_performance = []
    total_step = len(validation_generator)
    with torch.set_grad_enabled(False):
        for local_batch, local_labels in validation_generator:
            j += 1
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            YFP = local_batch[:, 0, :, :]
            DAPI = local_batch[:, 1, :, :]
            YFP_pearson_performance.append(getPearson(YFP, local_labels)) #NULL autoencode model 
            DAPI_pearson_performance.append(getPearson(DAPI, local_labels))
            YFP_MSE_performance.append(calculateMSE(YFP, local_labels))
            DAPI_MSE_performance.append(calculateMSE(DAPI, local_labels))
    print("global null YFP pearson performance: ", np.mean(YFP_pearson_performance), np.std(YFP_pearson_performance))
    print("global null DAPI pearson performance: ", np.mean(DAPI_pearson_performance), np.std(DAPI_pearson_performance))
    print("global null YFP MSE performance: ", np.mean(YFP_MSE_performance), np.std(YFP_MSE_performance))
    print("global null DAPI MSE performance: ", np.mean(DAPI_MSE_performance), np.std(DAPI_MSE_performance))
    pickle.dump((np.mean(YFP_pearson_performance), np.std(YFP_pearson_performance)), open("pickles/YFP_model_pearson_performance.pkl", "wb"))
    pickle.dump((np.mean(YFP_MSE_performance), np.std(YFP_MSE_performance)), open("pickles/YFP_model_MSE_performance.pkl", "wb"))
    pickle.dump((np.mean(DAPI_pearson_performance), np.std(DAPI_pearson_performance)), open("pickles/DAPI_model_pearson_performance.pkl", "wb"))
    pickle.dump((np.mean(DAPI_MSE_performance), np.std(DAPI_MSE_performance)), open("pickles/DAPI_model_MSE_performance.pkl", "wb"))

def getROC(lab_thresh, sample_size, model=None, loadName=None, validation_generator=None, device=None):
    """
    LAB_THRESH is the pixel threshold that we use to binarize our label image 
    SAMPLE_SIZE specifies how many images we want to include in this analysis
    MODEL is the model of type NN.Module (a trained model specified by the name within this function will be loaded)
    LOADNAME is the name of the model to load
    VALIDATION_GENERATOR is the generator that iterates over validation/test images to evaluate
    DEVICE is the cuda device to allocate to 
    Saves ROC curves for YFP Null Model, DAPI Null Model, and the actual ML model, 
    Pickles the coordinates of the different curves, and saves to pickles/
    """
    checkpoint = torch.load(loadName)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    mapp = {} #will correspond to ML model results
    null_mapp = {} #will correspond to Null YFP model results
    null_DAPI_mapp = {} #will correspond to Null DAPI model results
    threshs = list(np.arange(0, .9, .1)) #threshs are the various thresholds that we will use to binarize our predicted label images
    threshs += list(np.arange(.90, 1.1, .01))
    threshs +=  list(np.arange(1.1, 2, .1)) 
    threshs +=  list(np.arange(2, 5, 1)) 
    threshs.append(30) 
    threshs.append(1000)
    for t in threshs:
        mapp[t] = [] #thresh to list of (FPR, TPR) points
        null_mapp[t] = []
        null_DAPI_mapp[t] = []
    with torch.set_grad_enabled(False):
        j = 0
        for local_batch, local_labels in validation_generator:
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
                pickle.dump(x, open("pickles/null_mapp_x_values_fold_{}.pk".format(fold), "wb"))
                pickle.dump(y, open("pickles/null_mapp_y_values_fold_{}.pk".format(fold),  "wb"))
            if i == 1:
                pickle.dump(x, open("pickles/null_DAPI_mapp_x_values_fold_{}.pk".format(fold), "wb"))
                pickle.dump(y, open("pickles/null_DAPI_mapp_y_values_fold_{}.pk".format(fold), "wb"))
            if i == 2:
                print("dumping pickle")
                pickle.dump(x, open("pickles/mapp_x_values_fold_{}.pk".format(fold), "wb"))
                pickle.dump(y, open("pickles/mapp_y_values_fold_{}.pk".format(fold), "wb"))
            i += 1
            labels, x, y = labels[::-1], x[::-1], y[::-1]
            auc = np.trapz(y,x)
            print("AUC: ", auc)
       
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
    helper function to save MODEL, EPOCH, OPTIMIZER, and LOSS, as pytorch file to PATH
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
    pearsons = []
    for i in range(0, len(inputs)):
        if rand is None:
            rand = np.random.randint(0,10000000000)
        inp = inputs[i][0].reshape((img_dim,img_dim))
        if normalize == "unit":
            inp = (inp * inputSTD) + inputMean
            inp = inp.astype(np.uint8)
        if normalize == "scale":
            inp = (inp / 255) * inputMax
            inp = inp.astype(np.uint16)
        cv2.imwrite(directory + str(rand) + "input_yfp.tif", inp)
        dapi = inputs[i][1].reshape((img_dim, img_dim))
        dapi = (dapi / 255) * inputMax
        dapi = dapi.astype(np.uint16)
        cv2.imwrite(directory + str(rand) + "input_dapi.tif", dapi)
        lab = labels[i].reshape((img_dim,img_dim))
        if normalize == "unit":
            lab = (lab * labelSTD) + labelMean
            lab = lab.astype(np.uint8)
        if normalize == "scale":
            lab = (lab / 255) * labelMax
            lab = lab.astype(np.uint16)
        cv2.imwrite(directory + str(rand) + "label_AT8.tif", lab)
        pred = predicted[i].reshape((img_dim,img_dim))
        if normalize == "unit":
            pred = (pred * labelSTD) + labelMean
            pred = pred.astype(np.uint8)
        if normalize == "scale":
            pred = (pred / 255) * labelMax 
            pred = pred.astype(np.uint16)
        corr = np.corrcoef(lab.flatten(), pred.flatten())[0][1] #pearson correlation after transformations
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

#============================================================================
#============================================================================
## SUPPLEMENTAL CODE
#============================================================================
#============================================================================

class AblationDataset(Dataset):
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

def ablationTest(sample_size=1000000, validation_generator=None, ablate_DAPI_only=False, model=None, loadName=None, device=None):
    """
    SAMPLE_SIZE is the number of images to evaluate from the VALIDATION_GENERATOR
    VALIDATION_GENERATOR is the generator that iterates over validation/test images to evaluate
    If ABLATE_DAPI_ONLY, then will only ablate the DAPI input and leave YFP-tau alone 
    LOADNAME is the name of the model to load and evaluate 
    DEVICE is the cuda device to allocate to
    Method to run the model on the test set and print pearson performance under a range of ablation conditions
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
            for local_batch, local_labels in validation_generator:
                j += 1   
                local_batch = local_batch.cpu().numpy()
                img_dim = local_batch.shape[-1]
                local_batch = local_batch.reshape((2,img_dim, img_dim))
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
                # print(str(j) + "/" + str(total_step) + ", batch pearson: ", pearson)
                if j == sample_size:
                    break
                performance.append(pearson) 
        x.append(a)
        y.append(np.mean(performance))
        stds.append(np.std(performance))
        print("ablation: {}, global performance: avg={}, std= {}".format(a, np.mean(performance), np.std(performance)))
    

