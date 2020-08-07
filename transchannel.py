"""
All necessary code to reproduce the results for the paper are included here
This script is divided into three parts:
1) class and method definitions
2) global variables that can be changed and tweeked
3) a section to call different methods depending on what you would like done (training, testing, statistical analyses, etc.)
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
from PIL import Image
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
    AI in Cell dataset of images. csv_file should be a csv of x,t pairs of full path strings, pointing to images.
    x is the YFP-tau image name, and t is the AT8-pTau image name
    """
    def __init__(self, csv_file, transform=None):
        self.transform = transform
        self.data = pd.read_csv(csv_file).values
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x, t = self.data[idx] 
        if normalize == "scale":
            x_img = cv2.imread(x, cv2.IMREAD_UNCHANGED) #for enhanced images, will be 3 channel; for raw images will be 1 channel 
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

class AblationDataset(Dataset):
    """
    dataset in which we ablate the bottom x% of pixels to be 0 valued
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

def getDAPI(filename):
    """
    simple string manipulation to replace "FITC" with "DAPI" and "Blue" with "UV" of input FILENAME
    returns the modified string
    """
    DAPI = filename.replace("FITC", "DAPI")
    DAPI = DAPI.replace("Blue", "UV")
    return DAPI

class Unet_mod(nn.Module):
    """
    motif: upsamp, follow by 2x2 conv, concatenate this with old layer (skip connections), transpose conv 
    """
    def __init__(self):
        super(Unet_mod, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, 3) 
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
        if img_dim == 256:
            self.upsamp1 = nn.Upsample(size=(30,30),mode='bilinear', align_corners=True) #size = h5 - 1
            self.upsamp2 = nn.Upsample(size=(62,62), mode='bilinear', align_corners=True)#size = h3 - 1
            self.upsamp3 = nn.Upsample(size=(126,126), mode='bilinear', align_corners=True) #size = h1 - 1
            self.upsamp4 = nn.Upsample(size=(255,255), mode='bilinear', align_corners=True) #size = input - 1
        if img_dim == 2048:
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
        h20 = nn.functional.relu(self.conv13(torch.cat((h0,h19), dim=1))) #127 x 127
        return h20

def plotInputs(inputs, labels, predicted, directory, rand=None):
    """
    method to plot the INPUTS, LABELS, and PREDICTED images in DIRECTORY, writes corresponding images with a random identifier
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
        cv2.imwrite(directory + str(rand) + "_aainput_yfp.tif", inp)
        dapi = inputs[i][1].reshape((img_dim, img_dim))
        dapi = (dapi / 255) * inputMax
        dapi = dapi.astype(np.uint16)
        cv2.imwrite(directory + str(rand) + "_aainput_dapi.tif", dapi)
        lab = labels[i].reshape((img_dim,img_dim))
        if normalize == "unit":
            lab = (lab * labelSTD) + labelMean
            lab = lab.astype(np.uint8)
        if normalize == "scale":
            lab = (lab / 255) * labelMax
            lab = lab.astype(np.uint16)
        cv2.imwrite(directory + str(rand) + "_bbactual.tif", lab)
        pred = predicted[i].reshape((img_dim,img_dim))
        if normalize == "unit":
            pred = (pred * labelSTD) + labelMean
            pred = pred.astype(np.uint8)
        if normalize == "scale":
            pred = (pred / 255) * labelMax 
            pred = pred.astype(np.uint16)
        corr = np.corrcoef(lab.flatten(), pred.flatten())[0][1] #correlation after transformations
        cv2.imwrite(directory + str(rand) + "cc_predicted" + "_pearson=" + str(corr)[0:5] + ".tif",pred)

def getPearson(predicted, labels):
    """
    returns the average Pearson correlation coefficient between PREDICTED and LABELS lists of images
    """
    labels = labels.cpu().numpy()
    predicted = predicted.cpu().numpy()
    pearsons = []
    for i in range(0, len(predicted)):
        corr = np.corrcoef(labels[i].flatten(), predicted[i].flatten())[0][1]
        if np.isnan(corr):
            print("nan pearson")
        else:
            pearsons.append(corr)
    return np.mean(pearsons) 

def plotThresholds(inputs=None, predicted=None, labels=None):
    """
    plots threhsolded images of INPUTS, PREDICTED, and LABELS to outputs/
    threshtype = "std" for investigating different thresholds = mean + multiplier * std
    threshtype = "absolute" for investigating different thresholds = specified raw pixel values
    """
    inputs = inputs.cpu().numpy()
    labels = labels.cpu().numpy()
    if predicted != None:
        predicted = predicted.cpu().numpy()
    for i in range(0, len(inputs)):
        rand = np.random.randint(0,10000)
        inp = (inputs[i][0] / 255) * 65535.0
        lab = (labels[i] / 255) * 65535.0
        inp = inp.reshape((img_dim,img_dim))
        lab = lab.reshape((img_dim,img_dim))
        if predicted != None:
            pred = (predicted[i] / 255) * 65535.0
            pred = pred.reshape((img_dim,img_dim))
        cv2.imwrite("outputs/" + str(rand) + "aaainp" + ".tif", inp.astype(np.uint16))
        cv2.imwrite("outputs/" + str(rand) + "bbblabel" + ".tif", lab.astype(np.uint16))
        if predicted != None:
            cv2.imwrite("outputs/" + str(rand) + "cccpred" + ".tif", pred.astype(np.uint16))  
            print("original mean, std: ", np.mean(inp), np.std(inp), np.mean(lab), np.std(lab), np.mean(pred), np.std(pred))
        else:
            print("original mean, std: ", np.mean(inp), np.std(inp), np.mean(lab), np.std(lab))
        ##normalize image to have mean = 0, variance = 1
        imean, istd = np.mean(inp), np.std(inp)
        lmean, lstd = np.mean(lab), np.std(lab)
        if predicted != None:
            pmean, pstd = np.mean(pred), np.std(pred)
            pred = ((pred - pmean) /float(pstd))
        inp = ((inp - imean) / float(istd))
        lab = ((lab - lmean) /float(lstd))
        threshs = [-1, -.75, -.5, -.25, 0, .25, .5, .75, 1]
        for thresh in threshs:
            mask = np.where(inp >= thresh, 65535, 0)
            mask = mask.astype(np.uint16)
            cv2.imwrite("outputs/" + str(rand) + "aainp_thresh=" +str(thresh) + ".tif",mask)
            mask = np.where(lab >= thresh, 65535, 0)
            mask = mask.astype(np.uint16)
            cv2.imwrite("outputs/" + str(rand) + "bblab_thresh=" +str(thresh) + ".tif",mask)
            if predicted != None:
                pred = pred.reshape((img_dim,img_dim))
                mask = np.where(pred >= thresh, 65535, 0)
                mask = mask.astype(np.uint16)
                cv2.imwrite("outputs/" + str(rand) + "ccpred_thresh=" +str(thresh) + ".tif",mask)
   
def getROC(lab_thresh, sample_size):
    """
    plots and saves ROC curves for both Null models and the actual ML model, saves to current directory,
    pickles the coordinates of the curve, also prints the plot coordinates
    """
    if fold in [1,2,3]:
        loadName = "models/cross_validate_fold{}cross_validating_Unet_mod_continue_training.pt".format(fold)
    else:
        loadName = "models/raw_1_thru_6_full_Unet_mod_continue_training_2.pt" #model used for archival HCS image validation
    checkpoint = torch.load(loadName)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()
    mapp = {}
    null_mapp = {}
    null_DAPI_mapp = {}
    threshs = list(np.arange(0, .9, .1))
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
                null_TPR, null_TNR, null_PPV, null_NPV, null_FNR, null_FPR = getMetrics(local_batch[:,0,:,:], local_labels, lab_thresh=lab_thresh, pred_thresh=t, isNull=True)
                null_DAPI_TPR, null_DAPI_TNR, null_DAPI_PPV, null_DAPI_NPV, null_DAPI_FNR, null_DAPI_FPR = getMetrics(local_batch[:,1,:,:], local_labels, lab_thresh=lab_thresh, pred_thresh=t, isNull=True) #for NULL AUC
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
            print("labels: ", labels)
            print("x: ", len(x),  x)
            print("y: ", len(y), y)
            print("coordinates: ", coordinates)
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
        rankings = []
        for coord in coordinates:
            rankings.append((coord[0], coord[2] - coord[1]))
        rankings = sorted(rankings, key=lambda x: x[1])
        print("ranked: ", rankings)

def getMetrics(predicted, labels, lab_thresh=None, pred_thresh=None):
    """
    returns TPR, TNR, PPV, NPV, FNR, FPR for batch of images PREDICTED and LABELS
    LAB_THRESH and PRED_THRESH are the pixel thresholds at which signal >= thresh is called positive, else negative
    """
    predicted = predicted.cpu().numpy()
    labels = labels.cpu().numpy()

    for i in range(0, len(labels)):
        ##put images back in original space
        lab = labels[i].reshape((img_dim,img_dim))
        pred = predicted[i].reshape((img_dim,img_dim))
        lab = (lab / 255) * 65535.0  
        pred = (pred / 255) * 65535.0      
        ##normalize image to have mean = 1, variance = 1
        lmean, lstd = np.mean(lab), np.std(lab)
        pmean, pstd = np.mean(pred), np.std(pred)
        lab = ((lab - lmean) /float(lstd)) + 1
        pred = ((pred - pmean) /float(pstd)) + 1
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
        if math.isnan(TPR) or math.isnan(TNR) or math.isnan(PPV) or math.isnan(NPV) or math.isnan(FNR) or math.isnan(FPR):
            TPR = 0 if math.isnan(TPR) else TPR
            TNR = 0 if math.isnan(TNR) else TNR
            PPV = 0 if math.isnan(PPV) else PPV
            NPV = 0 if math.isnan(NPV) else NPV
            FNR = 0 if math.isnan(FNR) else FNR
            FPR = 0 if math.isnan(FPR) else FPR
            return TPR, TNR, PPV, NPV, FNR, FPR
        if math.isnan(TPR) or math.isnan(FPR):
            print("TPR or FPR is nan with lab/pred thresh: ", lab_thresh, pred_thresh)
    return TPR, TNR, PPV, NPV, FNR, FPR

def saveTraining(model, epoch, optimizer, loss, PATH):
    """
    function to save MODEL, EPOCH, OPTIMIZER, and LOSS, as pytorch file to PATH
    """
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }, PATH)

def pearsonCorrLoss(outputs, targets):
    """
    customized loss function to compute the negative pearson correlation loss
    """
    vx = outputs - torch.mean(outputs)
    vy = targets - torch.mean(targets)
    cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return -1 * cost

def train():
    """
    method to train the model, saves the model every few epochs
    """
    if continue_training:
        checkpoint = torch.load(load_training_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        model.train()
    start_time = time.time()
    total_step = len(training_generator)
    for epoch in range(max_epochs):
        model.train() #added later set to train mode and keep track of gradient operations  
        i = 0 
        running_loss = 0
        for local_batch, local_labels in training_generator:
            i += 1
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            outputs = model(local_batch)
            if lossfn == "pearson":
                loss = pearsonCorrLoss(outputs, local_labels)
                if math.isnan(loss):
                    i -= 1
                    print("nan loss at epoch: ", epoch)
                    continue 
                else:
                    running_loss += loss 
            else: #other pytorch defined loss functions
                loss = criterion(outputs, local_labels)
                running_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("epoch: ", epoch + 1, "avg training loss over all batches: ", running_loss / float(i), ", time elapsed: ", time.time() - start_time)
        ##check validation loss every 5 epochs, save model every 5 epochs
        if epoch % 5 == 0:
            saveTraining(model, epoch, optimizer, running_loss / float(i), "models/" + plotName + ".pt") #or should criterion be loss
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
    method to run the model on the test set and print pearson performance
    """
    start_time = time.time()
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

def calculateMSE(predicted, actual):
    """
    calculates the MSE between PREDICTED and ACTUAL, performing a local normalization to 0 mean and unit variance
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

def getNull():
    """
    finds the null pearson performance of the dataset, i.e. pearson(input, label)
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

def getNullMSE(sample_size):
    """
    calculates the null MSE of the dataset, given SAMPLE_SIZE number of images
    """
    j = 0
    null_YFP_performance = []
    null_DAPI_performance = []
    total_step = len(validation_generator)
    print("length of validation generator: ", total_step)
    with torch.set_grad_enabled(False):
        for local_batch, local_labels in validation_generator:
            j += 1
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            YFP = local_batch[:, 0, :, :]
            DAPI = local_batch[:, 1, :, :]
            null_YFP_performance.append(calculateMSE(YFP, local_labels))
            null_DAPI_performance.append(calculateMSE(DAPI, local_labels))
            if j == sample_size:
                break
            print(j)
    print("null YFP peformance (mse): ", np.mean(null_YFP_performance), np.std(null_YFP_performance))
    print("null DAPI peformance (mse): ", np.mean(null_DAPI_performance), np.std(null_DAPI_performance))


def plotPixelDistributions(sample_size, exclude_0_valued_pixels=False, display_pixel_max=255):
    """
    gets post normalized pixel distributions of YFP, actual AT8 pTau, and predicted AT8 pTau
    we will exclude pixels without signal (pixel value = 0) if exclude_0_valued_pixels is True
    display_pixel_max will be the highest pixel value that we'll include in our display histogram
    dumps results to pickle
    """
    classes = ["YFP", "AT8", "predicted", "DAPI"]
    pixels_dict = {c: {px : 0 for px in range(0, 65536)} for c in classes} ##predicted range is unbounded, but YFP, DAPI, and AT8 are bounded in range (0, 255)
    start_time = time.time()
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
    full_sampler = SubsetRandomSampler(indices) #radom indices 
    full_generator = data.DataLoader(dataset,sampler=full_sampler, **test_params)
    with torch.set_grad_enabled(False):
        for local_batch, local_labels in full_generator:
            j += 1
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            outputs = model(local_batch)
            if lossfn == "pearson":
                loss =  pearsonCorrLoss(outputs, local_labels)
            else:
                loss = criterion(outputs, local_labels)
            running_loss += loss 
            pearson = getPearson(outputs, local_labels)
            performance.append(pearson) 
            local_batch = local_batch.cpu().numpy()
            local_labels = local_labels.cpu().numpy()
            outputs = outputs.cpu().numpy()
            batches = [local_batch[:,0,:,:], local_labels, outputs, local_batch[:,1,:,:]]
            for i in range(0, len(batches)):
                if i == 0:
                    key = "YFP"
                if i == 1:
                    key = "AT8"
                if i == 2:
                    key = "predicted"
                if i == 3:
                    key = "DAPI"
                for img in batches[i]:            
                    img = img.astype(np.uint16).flatten().tolist()
                    for el in img:
                        pixels_dict[key][el] += 1
            if j == sample_size:
                break
    print("time elapsed: ", time.time() - start_time)
    print("global performance: ", np.mean(performance))
    print("global loss: ", running_loss / float(j))
    pickle.dump(pixels_dict, open("pickles/pixels_dict_HCS_prep_full_set.plk", "wb"))

def getStats(sample_size):
    """
    method to get the average and std of the labels and predicted
    """
    sum_matrix_label = np.zeros(img_dim * img_dim) #flattened array to hold the sum of each label image
    sum_matrix_predicted = np.zeros(img_dim * img_dim) #flattened array to hold the sum of each predicted image
    loadName = "models/raw_1_thru_6_full_Unet_mod_continue_training_2.pt" #model used for historical image validation
    checkpoint = torch.load(loadName)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    j = 0 
    full_data_sampler = SubsetRandomSampler(indices) 
    full_data_generator = data.DataLoader(dataset, sampler=full_data_sampler, **train_params)
    with torch.set_grad_enabled(False):
        for local_batch, local_labels in full_data_generator:
            j += 1
            local_batch = local_batch.to(device)
            outputs = model(local_batch)
            local_labels = local_labels.cpu().numpy()
            outputs = outputs.cpu().numpy()
            sum_matrix_label += local_labels.flatten() 
            sum_matrix_predicted += outputs.flatten()
            if j == sample_size:
                break
    label_average = np.sum(sum_matrix_label) / float(j * img_dim * img_dim)
    predicted_average = np.sum(sum_matrix_predicted) / float(j * img_dim * img_dim)
    print("averages: ", label_average, predicted_average)

    #find std
    label_numerator_sum = 0 #scalar value to hold the ongoing summation of the numerator for the label
    predicted_numerator_sum = 0 #scalar value to hold the ongoing summation of the numerator for the predicted
    j = 0
    with torch.set_grad_enabled(False):
        for local_batch, local_labels in full_data_generator:
            j += 1
            local_batch = local_batch.to(device)
            outputs = model(local_batch)
            local_labels = local_labels.cpu().numpy()
            outputs = outputs.cpu().numpy()
            local_labels = local_labels.reshape((img_dim, img_dim))
            outputs = outputs.reshape((img_dim, img_dim))
            label_numerator_sum += np.sum(np.square(local_labels - label_average))
            predicted_numerator_sum += np.sum(np.square(outputs - predicted_average))
            if j == sample_size:
                break 
    std_label = np.sqrt(label_numerator_sum / float((j * img_dim * img_dim) - 1))
    std_predicted = np.sqrt(predicted_numerator_sum / float((j * img_dim * img_dim) - 1))
    print("stds: ", std_label, std_predicted)

def getMSE(sample_size):
    """
    calculates MSE of predictions + null MSE and prints 
    """
    if cross_val:
        loadName = "models/cross_validate_fold{}cross_validating_Unet_mod_continue_training.pt".format(fold)
    else:
        loadName = "models/raw_1_thru_6_full_Unet_mod_continue_training_2.pt" #model used for historical image validation
    checkpoint = torch.load(loadName)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()
    j = 0
    performance = []
    null_performance = []
    total_step = len(validation_generator)
    with torch.set_grad_enabled(False):
        for local_batch, local_labels in validation_generator:
            j += 1
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            outputs = model(local_batch)
            mse = calculateMSE(outputs, local_labels)
            performance.append(mse)
            null_mse = calculateMSE(local_batch[0][0], local_labels)
            null_performance.append(null_mse)
            if j == sample_size:
                break
    print("global peformance (mse): ", np.mean(performance), np.std(performance))
    print("null peformance (mse): ", np.mean(null_performance), np.std(null_performance))

def getOverlap(sample_size):
    """
    finds the percentage of signal that is present in both YFP input and pTau output
    divided by the total signal of pTau
    method to show that essentially all of pTau signal is also present in YFP channel (to some permissive threshold) 
    """
    start_time = time.time()
    label_thresh = 2 #thresh = 1 lets in background artifacts
    input_threshs = [2, 1.75, 1.5, 1.25, 1, .75, .5, .25, 0] #anything over input_thresh will be considered positive signal
    for input_thresh in input_threshs:
        fractions = []
        full_data_sampler = SubsetRandomSampler(indices) 
        full_data_generator = data.DataLoader(dataset, sampler=full_data_sampler, **train_params)
        j = 0
        with torch.set_grad_enabled(False):
            for local_batch, local_labels in full_data_generator:
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
                    label = ((label - lmean) /float(lstd)) + 1
                    img = ((img - imean) /float(istd)) + 1
                    img_count = np.sum(np.where((img >= input_thresh), 1, 0))
                    label_count = np.sum(np.where((label >= label_thresh), 1, 0))
                    overlap = np.sum(np.where((label >= label_thresh) & (img >= input_thresh), 1, 0))
                    fraction = overlap / float(label_count)
                    fractions.append(fraction)
                if j == sample_size:
                    break
        print("inp thresh: ", input_thresh, "label thresh: ", label_thresh, "average overlap: ",np.mean(fractions))

def getFractionalAT8InYFP(sample_size):
    """
    apply same thresh to both YFP and AT8 images after normalization, calculate fraction of positive AT8 pixels that are also positive in corresponding YFP 
    """
    start_time = time.time()
    threshs = [1, .75, .5, .25, 0, -.25, -.50, -.75, -1] #anything over input_thresh will be considered positive signal
    for thresh in threshs:
        fractions = []
        full_data_sampler = SubsetRandomSampler(indices) 
        full_data_generator = data.DataLoader(dataset, sampler=full_data_sampler, **train_params)
        j = 0
        with torch.set_grad_enabled(False):
            for local_batch, local_labels in full_data_generator:
                if j == 0 or j == 1: #plot two representative thresholding procedures to outputs/ 
                    plotThresholds(inputs=local_batch, predicted=None, labels=local_labels)
                j += 1
                local_batch = local_batch.cpu().numpy()
                local_labels = local_labels.cpu().numpy()
                for i in range(0, len(local_batch)):
                    img = local_batch[i][0]
                    img = img.reshape((2048,2048))
                    label = local_labels[i]
                    label = label.reshape((2048,2048))
                    lmean, lstd = np.mean(label), np.std(label)
                    imean, istd = np.mean(img), np.std(img)
                    label = ((label - lmean) /float(lstd))
                    img = ((img - imean) /float(istd))
                    label_count = np.sum(np.where((label >= thresh), 1, 0))
                    overlap = np.sum(np.where((label >= thresh) & (img >= thresh), 1, 0))
                    fraction = overlap / float(label_count)
                    fractions.append(fraction)
                if j == sample_size:
                    break
        print("thresh: ", thresh,"average overlap: ",np.mean(fractions))

def findStats():
    """
    given a model specified in this function, finds the average mean and std over the test set of images 
    for input images, label images, and predicted
    """
    start_time = time.time()
    loadName = "models/raw_1_thru_6_full_Unet_mod_continue_training_2.pt" #model used for historical image validation
    checkpoint = torch.load(loadName)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    total_step = len(validation_generator)
    j = 0
    input_stats = [] #list of (mean, std)
    label_stats = [] 
    predicted_stats = []
    with torch.set_grad_enabled(False):
        for local_batch, local_labels in validation_generator:
            j += 1
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            outputs = model(local_batch)
        print("input stats: ", input_stats)
        print("label stats: ", label_stats)
        print("predicted stats: ", predicted_stats)
        print("time elapsed: ", time.time() - start_time)
            
def ablationTest(sample_size, ablate_DAPI_only=False):
    """
    method to run the model on the test set and print pearson performance under ablation conditions
    if ablate_DAPI_only, then will only ablate the DAPI input and leave YFP-tau alone 
    calculates over sample_size number of images
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
    print("x: ", x)
    print("y: ", y)
    print("stds: ", stds)

#============================================================================
#============================================================================
## GLOBAL VARIABLES 
#============================================================================
#============================================================================

hostname = socket.gethostname() 
if torch.cuda.device_count() > 1:
  print("# available GPUs", torch.cuda.device_count(), "GPUs!")
use_cuda = torch.cuda.is_available()
## Parameters
fold = int(sys.argv[1]) ##1,2,3 for fold of the cross validation, else no cross validation and will do random train test split 
if fold in [1,2,3]:
    cross_val = True
else:
    cross_val = False
img_dim = 2048
plotName = "cross_validating_fold_{}_Unet_mod_continue_training".format(fold) #name to save model 
if "keiserlab" in hostname:
    if cross_val:
        csv_name = "/srv/home/dwong/AIInCell/datasets/butte_server_raw_plates_1_thru_16.csv" ##if cross validating, use extended dataset with negative controls included
    else:
        csv_name = "/srv/home/dwong/AIInCell/datasets/raw_dataset_1_thru_6_full_images_gpu2.csv" #else use training drug dataset plates 1 through 6
    meanSTDStats = "/srv/home/dwong/AIInCell/models/raw_dataset_1_thru_6_stats.npy" 
    minMaxStats = "/srv/home/dwong/AIInCell/models/raw_1_thru_6_min_max.npy" #stats for min max values 
    train_params = {'batch_size': 1, 'num_workers': 6}
    test_params = {'batch_size': 1, 'num_workers': 6} 
##for butte server
else:
    if cross_val:
        csv_name = "butte_server_raw_plates_1_thru_16.csv"
    else:
        csv_name = "raw_dataset_1_thru_6_full_images_gpu2.csv"
    meanSTDStats = "raw_dataset_1_thru_6_stats.npy"
    minMaxStats = "raw_1_thru_6_min_max.npy" 
    train_params = {'batch_size': 1, 'num_workers': 2} 
    test_params = {'batch_size': 1, 'num_workers': 2} 
max_epochs = 20
learning_rate = .001 #.001 og
continue_training = False ##if we want to train from a pre-trained model
if continue_training:
    load_training_name = "models/cross_validate_fold3cross_validating_Unet_mod.pt" #model to use if we're training from a pre-trained model
if "keiserlab" in hostname:
    gpu_list = [0,3] ##gpu ids to use
else:
    gpu_list = [1,0]
normalize = "scale" #scale for scaling values 0 to 255, or "unit" for subtracting mean and dividing by std 
lossfn = "pearson"
architecture = "unet mod"
dataset_type = "images"
##print some parameters to console
print("plotname: ", plotName)
print("dataset: ", csv_name)
print("cross_val: ", cross_val)
print("stats: ", meanSTDStats)
print("learning_rate: ", learning_rate, "epochs: ", max_epochs, "gpu list: ", gpu_list)
print("train params: ", train_params)
print("test params: ", test_params)
print("lossfn: ", lossfn, "normalization: ", normalize)
print("dataset_type: ", dataset_type)
print("architecture: ", architecture)
device = torch.device("cuda:" + str(gpu_list[0]) if use_cuda else "cpu")
if dataset_type == "images":
    dataset = ImageDataset(csv_name, transform=transforms.Compose([transforms.ToTensor()]))
## Random seed for data split 
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(.3 * dataset_size)) #30% test 
seed = 42
print("seed: ", seed)
np.random.seed(seed)
np.random.shuffle(indices)
if cross_val:
    split1 = int(np.floor(.3333 * dataset_size)) #30% test 
    split2 = int(np.floor(.6666667 * dataset_size))
    if fold == 1:
        train_indices, test_indices = indices[0:split2], indices[split2:] #[train, train, test]
    if fold == 2:
        train_indices, test_indices = indices[split1:], indices[0:split1] #[test, train, train]
    if fold == 3:
        train_indices, test_indices = indices[0:split1] + indices[split2:], indices[split1:split2] #[train, test, train]
    train_sampler = SubsetRandomSampler(train_indices) 
    test_sampler = SubsetRandomSampler(test_indices) #randomize indices 
else:
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
if architecture == "unet mod":
    model = Unet_mod()
if len(gpu_list) > 1:
    model = nn.DataParallel(model, device_ids=gpu_list).cuda()
model = model.to(device)
# Loss and optimizer
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
getMSE(100000) 
getNull()
getNullMSE(100000)
getROC(lab_thresh=2.0, sample_size=1000000)
ablationTest(10, ablate_DAPI_only=False)

# plotPixelDistributions(sample_size=1000000000, exclude_0_valued_pixels=False, display_pixel_max=20)
# getStats(1000000)
# getOverlap(1000000)
# getFractionalAT8InYFP(1)
# findStats()












