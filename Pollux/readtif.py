"""
Created on Tue Aug 20 2024

@author: Elias Fink (elias.fink22@imperial.ac.uk)

Fetch and use experimental data.

Methods:
    read_in:
        read in tif image
"""
import numpy as np
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
    l = np.zeros(int(len(img)/2)+1)
    for i, row in enumerate(img):
        for j, val in enumerate(row):
            x = i - len(img)/2
            y = j - len(row)/2
            index = int(np.sqrt(x**2 + y**2))
            if index < len(l):
                l[index] += val
    return np.array(l)/len(l)

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

if __name__ == '__main__':
    filenames = ["AB_EF_avg_bg.tif", "GH_IJ_avg_bg.tif", "Run_5_evt_6_alvium_0.tiff",
                 "Run_6_evt_6_alvium_0.tiff", "Run_7_evt_6_alvium_0.tiff",
                 "Run_8_evt_6_alvium_0.tiff", "Run_10_evt_6_alvium_0.tiff",
                 "Run_11_evt_6_alvium_0.tiff", "Run_13_evt_6_alvium_0.tiff"]
    PATHNAME = "Phase_Plate_Data/" + filenames[2]
    image = read_in(PATHNAME)
    # line = lineout(image)
    show(image)
