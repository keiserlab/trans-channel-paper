"""
Script to train and evaluate models on the independent dataset of the genome-wide functional screen in osteosarcoma cells 

This script is divided into three parts:
1) class and method definitions
2) global variables
3) method calls to run specific parts of the code

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
import cupy
import math
import numpy as np
import time 
import shutil
import os
import socket
import random
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#=======================================================================================
#=======================================================================================
#CLASS AND METHOD DEFINITIONS
#=======================================================================================
#=======================================================================================

class ImageDataset(Dataset):
    """
    Image dataset for the osteosarcoma system,
    csv should be in the form d2, d1, d0, such that
    d0 will be the Hoechst input image name, and d1 will be the cyclin-B1 label image name, d2 is a marker not used in this study, and can be ignored
    """
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file).values
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        d2, d1, d0 = self.data[idx]
        if "keiserlab" not in hostname: #if on butte lab server 
            d2 = d2.replace("/srv/nas/mk1/users/dwong/", "/data1/wongd/") 
            d1 = d1.replace("/srv/nas/mk1/users/dwong/", "/data1/wongd/")
            d0 = d0.replace("/srv/nas/mk1/users/dwong/", "/data1/wongd/")
        else: #if on keiser server
            d2 = d2.replace("/data1/wongd/", "/srv/nas/mk1/users/dwong/") 
            d1 = d1.replace("/data1/wongd/", "/srv/nas/mk1/users/dwong/")
            d0 = d0.replace("/data1/wongd/", "/srv/nas/mk1/users/dwong/")
        d0_img = cv2.imread(d0, cv2.IMREAD_UNCHANGED)
        d0_img = d0_img.astype(np.float32) 
        d0_img = (d0_img / 65535.0) * 255
        d1_img = cv2.imread(d1, cv2.IMREAD_UNCHANGED)
        d1_img = d1_img.astype(np.float32)
        d1_img = (d1_img / 65535.0) * 255
        return d0_img.reshape(1,img_dim,img_dim), d1_img

class AblationDataset(Dataset):
    """
    Image Dataset for ablation testing
    """
    def __init__(self, csv_file, thresh_percent):
        self.data = pd.read_csv(csv_file).values
        self.thresh_percent = thresh_percent
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        d2, d1, d0 = self.data[idx]
        if "keiserlab" not in hostname: #if on butte lab server 
            d2 = d2.replace("/srv/nas/mk1/users/dwong/", "/data1/wongd/") 
            d1 = d1.replace("/srv/nas/mk1/users/dwong/", "/data1/wongd/")
            d0 = d0.replace("/srv/nas/mk1/users/dwong/", "/data1/wongd/")
        else: #if on keiser server
            d2 = d2.replace("/data1/wongd/", "/srv/nas/mk1/users/dwong/") 
            d1 = d1.replace("/data1/wongd/", "/srv/nas/mk1/users/dwong/")
            d0 = d0.replace("/data1/wongd/", "/srv/nas/mk1/users/dwong/")
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
        return d0_img.reshape(1,img_dim,img_dim), d1_img

class Unet_mod(nn.Module):
    """
    Motif: upsamp, follow by 2x2 conv, concatenate this with old layer, transpose conv this final 
    """
    def __init__(self):
        super(Unet_mod, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3) 
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
        self.upsamp1 = nn.Upsample(size=(136,136),mode='bilinear', align_corners=True) #size = h5 - 1
        self.upsamp2 = nn.Upsample(size=(274,274), mode='bilinear', align_corners=True)#size = h3 - 1
        self.upsamp3 = nn.Upsample(size=(550,550), mode='bilinear', align_corners=True) #size = h1 - 1
        self.upsamp4 = nn.Upsample(size=(1103,1103), mode='bilinear', align_corners=True)

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
        h20 = nn.functional.relu(self.conv13(torch.cat((h0,h19), dim=1)))
        return h20

def pearsonCorrLoss(outputs, targets):
    """
    Calculates and returns the negative pearson correlation loss
    """
    vx = outputs - torch.mean(outputs)
    vy = targets - torch.mean(targets)
    cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return -1 * cost

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
    return np.mean(pearsons)  #avg pearson for batch of predicted 

def saveTraining(model, epoch, optimizer, loss, PATH):
    """
    Function to save MODEL, EPOCH, OPTIMIZER, and LOSS, as pytorch file to PATH
    """
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }, PATH)

def plotInputs(inputs, labels, predicted, directory, rand=None):
    """
    Method to plot the INPUTS, LABELS, and PREDICTED images in DIRECTORY, writes corresponding images with a random identifier
    """
    inputs = inputs.cpu().numpy()
    labels = labels.cpu().numpy()
    predicted = predicted.cpu().numpy()
    pearsons = []
    for i in range(0, len(inputs)):
        if rand is None:
            rand = np.random.randint(0,10000000000)
        inp = inputs[i][0].reshape((img_dim,img_dim))
        inp = (inp / 255) * 65535.0
        inp = inp.astype(np.uint16)
        cv2.imwrite(directory + str(rand) + "_aainput.tif", inp)
        lab = labels[i].reshape((img_dim,img_dim))        
        lab = (lab / 255) * 65535.0
        lab = lab.astype(np.uint16)
        cv2.imwrite(directory + str(rand) + "_bbactual.tif", lab)
        pred = predicted[i].reshape((img_dim,img_dim))
        pred = (pred / 255) * 65535.0
        pred = pred.astype(np.uint16)
        corr = np.corrcoef(lab.flatten(), pred.flatten())[0][1] #correlation after transformations
        cv2.imwrite(directory + str(rand) + "cc_predicted" + "_pearson=" + str(corr)[0:5] + ".tif",pred)

def train():
    """
    Method to train the model, saves the model every two epochs
    """
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
            else: 
                loss = criterion(outputs, local_labels)
                running_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("epoch: ", epoch + 1, "avg training loss over all batches: ", running_loss / float(i), ", time elapsed: ", time.time() - start_time)
        if epoch % 2 == 0:
            print("saving model")
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
    saveTraining(model, epoch, optimizer, running_loss / float(j), "models/" + plotName + ".pt") #save training loss as well

def test(sample_size):
    """
    Runs the model through the test size for SAMPLE_SIZE number of images, calculates and prints the average pearson 
    """
    loadName = "models/cross_validate_fold1cross_validating_sourav_set.pt"
    checkpoint = torch.load(loadName, map_location='cuda:0') #why is this last flag needed on butte server? 
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
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
            if lossfn == "MSE":
                loss = criterion(outputs, local_labels)
            if lossfn == "pearson":
                loss =  pearsonCorrLoss(outputs, local_labels)
            running_loss += loss 
            pearson = getPearson(local_labels, outputs)
            performance.append(pearson)
    print("global pearson performance: ", np.mean(performance), np.std(performance))
    print("global loss: ", running_loss / float(j))

def getNull():
    """
    Prints the null performance of the dataset, i.e. pearson(input, label)
    """
    j = 0
    performance = []
    full_data_sampler = SubsetRandomSampler(indices) 
    full_data_generator = data.DataLoader(dataset, sampler=full_data_sampler, **train_params)
    total_step = len(full_data_generator)
    with torch.set_grad_enabled(False):
        for local_batch, local_labels in full_data_generator:
            j += 1
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            pearson = getPearson(local_batch, local_labels) #NULL autoencode model 
            performance.append(pearson)
    print("global null performance: ", np.mean(performance), np.std(performance))

def calculateMSE(predicted, actual):
    """
    Helper function to calculate the MSE between PREDICTED and ACTUAL, will normalize to 0 mean and unit variance
    Returns the MSE
    """
    predicted = predicted.cpu().numpy()
    actual = actual.cpu().numpy()
    predicted = predicted.reshape((img_dim, img_dim))
    actual = actual.reshape((img_dim, img_dim))
    ##normalize image to have mean = 0, variance = 1 locally
    lmean, lstd = np.mean(actual), np.std(actual)
    pmean, pstd = np.mean(predicted), np.std(predicted)
    actual = ((actual - lmean) /float(lstd)) 
    predicted = ((predicted - pmean) /float(pstd)) 
    mse = np.average(np.square(predicted - actual))
    return mse

def getMSE(sample_size):
    """
    Prints the MSE with SAMPLE_SIZE number of images selected
    """
    loadName = "models/d0_to_d1_cyclin_only_dataset_fold3.pt"
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
            null_mse = calculateMSE(local_batch, local_labels)
            performance.append(mse)
            null_performance.append(null_mse)
            if j == sample_size:
                break
    print("global peformance (mse): ", np.mean(performance), np.std(performance))
    print("null peformance (mse): ", np.mean(null_performance), np.std(null_performance))

def ablationTest(sample_size):
    """
    Method to run the model on the test set and print pearson performance over SAMPLE_SIZE number of images
    """
    ablations = list(np.arange(0, .1, .02))
    ablations += list(np.arange(.1, .9, .1))
    ablations += list(np.arange(.9, 1.02, .02))
    x = []
    y = []
    start_time = time.time()
    loadName = "models/cross_validate_fold1cross_validating_sourav_set.pt"
    checkpoint = torch.load(loadName, map_location='cuda:0')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    for a in ablations:
        print("ablation", a)
        performance = []
        total_step = len(validation_generator)
        j = 0
        with torch.set_grad_enabled(False):
            for local_batch, local_labels in validation_generator:
                j += 1
                ##ablate bottom x% of pixels to 0
                local_batch = local_batch.cpu().numpy()
                local_batch = local_batch.reshape((img_dim, img_dim))
                thresh_index = int(1104 * 1104 * a)
                sorted_local_batch = np.sort(local_batch, axis=None)
                if a == 1:
                    threshold = 10000000
                else:
                    threshold = sorted_local_batch[thresh_index]
                rand = np.random.randint(0,100000000)
                local_batch[local_batch < threshold] = 0
                local_batch = local_batch.reshape((1, 1, img_dim, img_dim))
                local_batch = torch.from_numpy(local_batch).float().to(device)
                local_labels = local_labels.to(device)
                outputs = model(local_batch)
                pearson = getPearson(outputs, local_labels)
                if j == sample_size:
                    break
                performance.append(pearson) 
        x.append(a)
        y.append(np.mean(performance))
        print("global performance: ", np.mean(performance))
    plt.xlabel("% Ablated")
    plt.ylabel("Pearson Accuracy")
    plt.plot(x,y,linewidth=2.0)
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.title("Ablation Testing")
    print("x: ", x)
    print("y: ", y)


#=======================================================================================
#=======================================================================================
#GLOBAL VARIABLES
#=======================================================================================
#=======================================================================================

hostname = socket.gethostname() 
if "keiserlab.org" not in hostname:
    prefix = "/data1/wongd/"
else:
    prefix = "/srv/nas/mk1/users/dwong/"
task = "transchannel" #for training the learner on unablated images
# task = "ablation" ##for training the learner on ablated images at the 95th percentile intensity
short = False #param for truncating training and testing process for quick training dev
img_dim = 1104
train_params = {'batch_size': 1, 'num_workers': 5}
test_params = {'batch_size': 1, 'num_workers': 5}
plotName = "MODEL_NAME" #name of model to save
csvName = "datasets/cyclin_dataset.csv"
gpu_list = [1] 
max_epochs = 5
learning_rate = .001
continue_training = False
if continue_training:
    load_training_name = "LOAD_MODEL_NAME.pt"
lossfn = "pearson"
architecture = "unet mod"
fold = int(sys.argv[1]) #specify cross validation fold to use in [1,2,3], else any other integer will do a 70% train, 30% test split
if fold in [1,2,3]:
    cross_val = True
else:
    cross_val = False
device = torch.device("cuda:" + str(gpu_list[0]))
## Generators
if task == "transchannel":
    dataset = ImageDataset(csvName)
if task == "ablation":
    dataset = AblationDataset(csvName, .95)
## Random seed for data split 
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(.3 * dataset_size)) #30% test 
np.random.seed(42)
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
    test_sampler = SubsetRandomSampler(test_indices) 
else:
    train_indices, test_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices) 
training_generator = data.DataLoader(dataset, sampler=train_sampler, **train_params)
validation_generator = data.DataLoader(dataset,sampler=test_sampler, **test_params)
if architecture == "unet mod":
    model = Unet_mod()
if len(gpu_list) > 1:
    model = nn.DataParallel(model, device_ids=gpu_list).cuda()
model = model.to(device)
if lossfn == "MSE": #in case we want to use a different loss function than negative pearson corr loss
    criterion = nn.MSELoss()
if lossfn == "MAE":
    criterion = nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=.9)
if continue_training:
    checkpoint = torch.load(load_training_name,  map_location='cuda:0')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    model.train()


#=======================================================================================
#=======================================================================================
#METHOD CALLS 
#=======================================================================================
#=======================================================================================

train()
test(1000000)
getNull()
ablationTest(1000000)
getMSE(1000000)



