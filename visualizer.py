import pandas as pd
import cv2
import math
import numpy as np
import time 
import shutil
import os
import socket
print(socket.gethostname())
import matplotlib
import matplotlib.pyplot as plt
# import colormaps as cmaps
# plt.register_cmap(name='viridis', cmap=cmaps.viridis)
# plt.set_cmap(cmaps.viridis)

def colorImage(num_images):
    #first tile images 
    # directory = "Unet_cont_training_2_enhanced"
    # directory = "Figure 1 Pictures/Unet mod 1 candidates"
    directory = "outputs_enhanced" 
    # directory = "Unet_cont_training_enhanced"
    for filename in os.listdir(directory):
        if ".tif" not in filename: 
            continue
        img = cv2.imread(directory + "/" + filename, cv2.IMREAD_UNCHANGED)
        tile(img, num_images, filename, "temp_tiles/") #

    #then iterate over tiles and color them
    for filename in os.listdir("temp_tiles"):
        img = plt.imread("temp_tiles/" + filename)
        print(filename)
        filename = filename.replace(".tif", ".tiff")
        if "inp" in filename or "YFP" in filename or "FITC" in filename:
            print("saving!")
            plt.imsave('temp_colored_tiles/' + filename, img)
        if "predicted" in filename or "actual" in filename or "lab" in filename or "dsRed" in filename:
            plt.imsave('temp_colored_tiles/' + filename, img, cmap="magma")
        if "dapi" in filename or "DAPI" in filename:
            plt.imsave('temp_colored_tiles/' + filename, img, cmap="cividis")
            


def tile(image, length, name, directory):
    """
    takes IMAGE of type cv2 image, and the number of tiles wanted along one side LENGTH, also retains the NAME
    to integrate into the new image name. writes tiles to DIRECTORY.
    """
    print(name)
    # if "input" in name:
    #   subdir = "inputs/"
    # if "actual" in name:
    #   subdir = "labels/"
    # if "predicted" in name:
    #   subdir = "predicted/"
    numrows, numcols = length, length
    height = int(image.shape[0] / numrows)
    width = int(image.shape[1] / numcols)
    for row in range(numrows):
        for col in range(numcols):
            y0 = row * height
            y1 = y0 + height
            x0 = col * width
            x1 = x0 + width
            cv2.imwrite(directory + name.replace('.tif', '') + 'tile_%d%d.tif' % (row, col), image[y0:y1, x0:x1])

def reset():
    if os.path.exists("temp_tiles/"):
        shutil.rmtree("temp_tiles/")
    os.mkdir("temp_tiles")
    if os.path.exists("temp_colored_tiles/"):
        shutil.rmtree("temp_colored_tiles/")
    os.mkdir("temp_colored_tiles")


reset()
colorImage(1)




