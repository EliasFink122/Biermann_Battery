"""
Created on Tue Aug 20 2024

@author: Elias Fink (elias.fink22@imperial.ac.uk)

Fetch and use experimental data.

Methods:
    read_in:
        read in tif image
"""
import os
import numpy as np
import scipy.ndimage as ndi
from PIL import Image
import matplotlib.pyplot as plt

def read_in(path: str) -> np.ndarray:
    '''
    Read in data from tif image

    Args:
        path: absolute path to image

    Returns:
        picture as array
    '''

    img = Image.open(path)
    imarray = np.array(img)

    return imarray

def lineout(img: np.ndarray) -> np.ndarray:
    '''
    Take lineout of image

    Args:
        img: image to take lineout of

    Returns:
        lineout
    '''
    com = ndi.center_of_mass(img)
    l = np.zeros(len(img))
    for i, row in enumerate(img):
        for j, val in enumerate(row):
            x = i - com[0]
            y = j - com[1]
            if x**2 + y**2 == 0:
                continue
            index = int(np.sqrt(x**2 + y**2))
            if index < len(l):
                l[index] += val/(2*np.pi*np.sqrt(x**2 + y**2))
    return np.array(l)

def show(arr: np.ndarray):
    '''
    Plot array

    Args:
        array to show

    Raises:
        Value Error: if input array has wrong dimensions
    '''
    if len(np.shape(arr)) == 1:
        plt.figure()
        plt.title("Lineout")
        plt.plot(range(len(arr)), arr)
        plt.xlabel("R [A.U]")
        plt.ylabel("Intensity [A.U]")
        plt.show()
    elif len(np.shape(arr)) == 2:
        plt.figure()
        plt.title("Focal spot image")
        plt.imshow(arr)
        plt.xlabel("x [A.U]")
        plt.ylabel("y [A.U]")
        plt.colorbar(label = "Intensity [A.U]")
        plt.show()
    else:
        raise ValueError("Input array needs to be one or two dimensional.")

def loop_directory(dname: str):
    '''
    Loop through directory and save intensities and lineouts

    Args:
        dname: path to directory
    '''
    for file in os.listdir(os.fsencode(dname)):
        fname = os.fsdecode(file)
        if fname.endswith(".tif") or fname.endswith(".tiff"):
            print(f"Processing {fname} ...")
            img = read_in(dname + "/" + fname)
            np.savetxt("Outputs/intensity_" + fname.split('.')[0] + ".txt", img)
            lout = lineout(img)
            np.savetxt("Outputs/lineout_" + fname.split('.')[0] + ".txt", lout)

if __name__ == '__main__':
    filenames = ["AB_EF_avg_bg.tif", "GH_IJ_avg_bg.tif", "Run_5_evt_6_alvium_0.tiff",
                 "Run_7_evt_6_alvium_0.tiff", "Run_8_evt_6_alvium_0.tiff",
                 "Run_10_evt_6_alvium_0.tiff", "Run_11_evt_6_alvium_0.tiff",
                 "Run_13_evt_6_alvium_0.tiff"]
    DIRECTORY = "Phase_Plate_Data"
    PATHNAME = DIRECTORY + "/" + filenames[7]
    # image = read_in(PATHNAME)
    # line = lineout(image)
    # show(line)
    loop_directory(DIRECTORY)
