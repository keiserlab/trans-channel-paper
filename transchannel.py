"""
We include here all of the necessary code to reproduce the main training and model evaluation results from the tau dataset.
This script can be run with the command "python transchannel.py {FOLD NUMBER}". 
We used a 3-fold cross-validation scheme over an extended dataset for the supplemental analysis (specify this by setting FOLD NUMBER to either 1,2, or 3).
For the main paper analysis, we performed a 70% train, 30% test split presented. To specify this setup, give an integer argument that is not in [1,2,3] for FOLD NUMBER. 

This script is divided into five parts:
1) class and method definitions
2) helper functions
3) supplemental code for ablation analysis
4) setting global variables 
5) method calls to run specific parts of the code

Any questions should be directed to daniel.wong2@ucsf.edu. Thank you!
"""
import torch 
import torch.nn as nn
import torchvision.transforms as transforms
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
    def __init__(self, csv_file, transform=None):
        self.transform = transform
        self.data = pd.read_csv(csv_file).values
    
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
        if normalize == "scale":
            x_img = cv2.imread(x, cv2.IMREAD_UNCHANGED) 
            x_img = x_img.astype(np.float32)
            x_img = ((x_img - inputMin) / (inputMax - inputMin)) * 255
            DAPI_img = cv2.imread(getDAPI(x), cv2.IMREAD_UNCHANGED)
            DAPI_img = DAPI_img.astype(np.float32)
            DAPI_img = ((DAPI_img - DAPIMin) / (DAPIMax - DAPIMin)) * 255
            t_img = cv2.imread(t, cv2.IMREAD_UNCHANGED)
            t_img = t_img.astype(np.float32)
            t_img = ((t_img - labelMin) / (labelMax - labelMin)) * 255
            dest = np.zeros((2,img_dim,img_dim), dtype=np.float32)
            dest[0,:] = x_img
            dest[1,:] = DAPI_img
            return dest, t_img

class Unet_mod(nn.Module):
    """
    The deep learning architecture inspired by Unet (Ronneberger et al. 2015), Motif: upsample, follow by 2x2 conv, concatenate with early layers (skip connections), transpose convolution 
    """
    def __init__(self):
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
        
    def forward(self, x):
        h0 = x.view(x.shape[0], x.shape[1], img_dim, img_dim)
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

def train():
    """
    Method to train and save a model as a .pt object, saves the model every 5 epochs
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
            if lossfn == "pearson":
                loss = pearsonCorrLoss(outputs, local_labels)
                running_loss += loss 
            else: #other pytorch defined loss functions in case we wanted to use a different loss function (didn't use in study)
                loss = criterion(outputs, local_labels)
                running_loss += loss.item()
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
                    if lossfn == "pearson":
                        loss =  pearsonCorrLoss(outputs, local_labels)
                    else:
                        loss = criterion(outputs, local_labels)
                    running_val_loss += loss 
            print("epoch: ", epoch + 1, " global validation loss: ", running_val_loss / float(j))
    saveTraining(model, epoch, optimizer, running_loss / float(j), "models/" + plotName + ".pt")

def test(sample_size):
    """
    Method to run the model on the test set, print the pearson performance, and pickle the result to pickles/
    """
    start_time = time.time()
    ##specifies which model to load
    if cross_val:
        loadName = "models/cross_validate_fold{}cross_validating_Unet_mod_continue_training.pt".format(fold)
    else:
        loadName = "models/raw_1_thru_6_full_Unet_mod_continue_training_2.pt" #model used for historical image validation
    checkpoint = torch.load(loadName)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
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
            if lossfn == "pearson":
                loss =  pearsonCorrLoss(outputs, local_labels)
            else:
                loss = criterion(outputs, local_labels)
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

def getROC(lab_thresh, sample_size):
    """
    LAB_THRESH is the pixel threshold that we use to binarize our label image 
    SAMPLE_SIZE specifies how many images we want to include in this analysis
    Saves ROC curves for YFP Null Model, DAPI Null Model, and the actual ML model, 
    Pickles the coordinates of the different curves, and saves to pickles/
    """
    if fold in [1,2,3]:
        loadName = "models/cross_validate_fold{}cross_validating_Unet_mod_continue_training.pt".format(fold)
    else:
        loadName = "models/raw_1_thru_6_full_Unet_mod_continue_training_2.pt" #model used for archival HCS image validation
    checkpoint = torch.load(loadName)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
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
        ##generate both ROC curves, one for model, one for null 
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
            plt.plot(x,y,linewidth=2.0)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.plot([0, .5, 1], [0,.5, 1], linewidth=1.0)
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.title("Reciever Operating Characteristic AUC = " + str(auc))
        plt.savefig('ROC_cross_validate_fold_{}_thresh='.format(fold) + str(lab_thresh) + "_sample_size" + str(sample_size) +"_" + loadName.replace("models/", "") + '.png')

def getMSE():
    """
    calculates the MSE of ML predictions, pickles the results to pickles/
    """
    if cross_val:
        loadName = "models/cross_validate_fold{}cross_validating_Unet_mod_continue_training.pt".format(fold)
    else:
        loadName = "models/raw_1_thru_6_full_Unet_mod_continue_training_2.pt" #model used for historical image validation
    checkpoint = torch.load(loadName)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()
    performance = []
    total_step = len(validation_generator)
    with torch.set_grad_enabled(False):
        for local_batch, local_labels in validation_generator:
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            outputs = model(local_batch)
            mse = calculateMSE(outputs, local_labels)
            performance.append(mse)
    print("global peformance (mse): ", np.mean(performance), np.std(performance))
    pickle.dump((np.mean(performance), np.std(performance)), open("pickles/ML_model_MSE_performance.pkl", "wb"))

def getNull():
    """
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


#============================================================================
#============================================================================
##HELPER FUNCTIONS
#============================================================================
#============================================================================
def calculateMSE(predicted, actual):
    """
    Helper function that calculates the MSE between PREDICTED and ACTUAL, performing a local normalization to 0 mean and unit variance
    returns the MSE
    """
    predicted = predicted.cpu().numpy()
    actual = actual.cpu().numpy()
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
    def __init__(self, csv_file, thresh_percent, transform=None):
        self.transform = transform
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

def ablationTest(sample_size, ablate_DAPI_only=False):
    """
    Method to run the model on the test set and print pearson performance under ablation conditions
    if ABLATE_DAPI_ONLY, then will only ablate the DAPI input and leave YFP-tau alone 
    calculates over SAMPLE_SIZE number of images
    """
    ablations = list(np.arange(0, .1, .02))
    ablations += list(np.arange(.1, 1.1, .1))
    x = []
    y = []
    stds = []
    start_time = time.time()
    if fold in [1,2,3]:
        loadName = "models/cross_validate_fold{}cross_validating_Unet_mod_continue_training.pt".format(fold)
    else:
        loadName = "models/raw_1_thru_6_full_Unet_mod_continue_training_2.pt" #model used for historical image validation
    checkpoint = torch.load(loadName)
    shutil.rmtree("outputs/")
    os.mkdir("outputs/")
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
                ##plot originals before ablation
                rand = np.random.randint(0,10000000000)
                cv2.imwrite("outputs/" + str(rand) + "_aainput_yfp_og.tif", ((YFP / 255) * 65535).astype(np.uint16))
                cv2.imwrite("outputs/" + str(rand) + "_aainput_dapi_og.tif", ((DAPI / 255) * 65535).astype(np.uint16))
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
                plotInputs(local_batch, local_labels, outputs, directory="outputs/", rand=rand)
                print(str(j) + "/" + str(total_step) + ", batch pearson: ", pearson)
                if j == sample_size:
                    break
                performance.append(pearson) 
        x.append(a)
        y.append(np.mean(performance))
        stds.append(np.std(performance))
        print("global performance: ", np.mean(performance), np.std(performance))
    plt.xlabel("% Ablated")
    plt.ylabel("Pearson Accuracy")
    plt.plot(x,y,linewidth=2.0)
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.title("Ablation Testing")
    plt.savefig('ablation_sample_size=' + str(sample_size) + 'model=' + loadName.replace("models/", "") + '.png')

#============================================================================
#============================================================================
##SETTING GLOBAL VARIABLES
#============================================================================
#============================================================================

hostname = socket.gethostname() 
if torch.cuda.device_count() > 1:
  print("# available GPUs", torch.cuda.device_count(), "GPUs!")
use_cuda = torch.cuda.is_available()

##fold of the 3-fold cross-validation to use (1,2, or 3), else any other integer will specify no cross-validation and use a random 70% training, 30% test split
fold = int(sys.argv[1]) 
if fold in [1,2,3]:
    cross_val = True
else:
    cross_val = False
img_dim = 2048
plotName = "MODEL_NAME".format(fold) #name used to save model 
if "keiserlab" in hostname: ##if on keiser lab server, else butte lab server 
    if cross_val:
        csv_name = "/srv/home/dwong/AIInCell/datasets/butte_server_raw_plates_1_thru_16.csv"
    else:
        csv_name = "/srv/home/dwong/AIInCell/datasets/raw_dataset_1_thru_6_full_images_gpu2.csv" #plates 1 through 6 on gpu2's fast drive, used to train model for historical validation 
    meanSTDStats = "/srv/home/dwong/AIInCell/models/raw_dataset_1_thru_6_stats.npy"
    minMaxStats = "/srv/home/dwong/AIInCell/models/raw_1_thru_6_min_max.npy" #stats for min max values 
    train_params = {'batch_size': 1, 'num_workers': 6} #batch size 32 for tiles, 4 seems ok for full image
    test_params = {'batch_size': 1, 'num_workers': 6} 
else:
    if cross_val:
        csv_name = "butte_server_raw_plates_1_thru_16.csv"
    else:
        csv_name = "raw_dataset_1_thru_6_full_images_gpu2.csv"
    meanSTDStats = "raw_dataset_1_thru_6_stats.npy"
    minMaxStats = "raw_1_thru_6_min_max.npy" #stats for min max values 
    train_params = {'batch_size': 1, 'num_workers': 2} #batch size 32 for tiles, 4 seems ok for full image
    test_params = {'batch_size': 1, 'num_workers': 2} 
max_epochs = 20
learning_rate = .001
continue_training = False ##if we want to train from a pre-trained model
if continue_training:
    load_training_name = "LOAD_MODEL_NAME.pt" #model to use if we're training from a pre-trained model
gpu_list = [0,1] ##gpu ids to use
normalize = "scale" #scale for scaling values 0 to 255, or "unit" for subtracting mean and dividing by std 
lossfn = "pearson"
architecture = "unet mod"
device = torch.device("cuda:" + str(gpu_list[0]) if use_cuda else "cpu")
dataset = ImageDataset(csv_name, transform=transforms.Compose([transforms.ToTensor()]))
## Random seed for data split 
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(.3 * dataset_size)) #30% test 
seed = 42
np.random.seed(seed)
np.random.shuffle(indices)
if cross_val: 
    split1 = int(np.floor(.3333 * dataset_size)) #1/3 test 
    split2 = int(np.floor(.6666667 * dataset_size)) #2/3 train
    if fold == 1:
        train_indices, test_indices = indices[0:split2], indices[split2:] #[train, train, test]
    if fold == 2:
        train_indices, test_indices = indices[split1:], indices[0:split1] #[test, train, train]
    if fold == 3:
        train_indices, test_indices = indices[0:split1] + indices[split2:], indices[split1:split2] #[train, test, train]
    train_sampler = SubsetRandomSampler(train_indices) 
    test_sampler = SubsetRandomSampler(test_indices) #randomize indices 
else: #if we're specifying a 70% train, 30% test split 
    train_indices, test_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices) #radom indices 
training_generator = data.DataLoader(dataset, sampler=train_sampler, **train_params)
validation_generator = data.DataLoader(dataset,sampler=test_sampler, **test_params)
##Initialize stats, model, loss function, and optimizer 
stats = np.load(meanSTDStats)
inputMean, inputSTD, labelMean, labelSTD, DAPIMean, DAPISTD = stats
stats = np.load(minMaxStats)
inputMin, inputMax, labelMin, labelMax, DAPIMin, DAPIMax = stats
model = Unet_mod()
if len(gpu_list) > 1:
    model = nn.DataParallel(model, device_ids=gpu_list).cuda()
model = model.to(device)
#Other loss functions are included here in case we wanted to use ones other than a pearson loss (not performed in study however)
if lossfn == "MSE":
    criterion = nn.MSELoss()
if lossfn == "L1Loss":
    criterion = nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=.9)


#============================================================================
#============================================================================
##METHOD CALLS
#============================================================================
#============================================================================
train()
test(100000)
getMSE() 
getNull()
getROC(lab_thresh=1.0, sample_size=1000000)
ablationTest(10, ablate_DAPI_only=False)











