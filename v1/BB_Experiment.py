"""
Created on Mon Jul 22 2024

@author: Elias Fink (elias.fink22@imperial.ac.uk)

Fetch and use experimental data.

Methods:
    read_in:
        read in tif image
    remove_white_space:
        trim image
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def read_in(path: str) -> tuple[np.ndarray, np.ndarray]:
    '''
    Read in data from tif image

    Args:
        path: absolute path to image

    Returns:
        pictures
    '''
    thresh = 150

    img = Image.open(path)
    imarray = np.array(img)

    imarray = imarray[int(len(imarray)/3):int(4*len(imarray)/5),
                      int(len(imarray[0])/5):int(4*len(imarray[0])/5)]
    for i, row in enumerate(imarray):
        for j, val in enumerate(row):
            if val < 150:
                imarray[i, j] = 0

    imarray = remove_white_space(imarray, thresh)

    for i, column in enumerate(np.transpose(imarray)):
        if np.mean(column) < thresh:
            beam_arr = imarray[:, :i]
            target_arr = imarray[:, i:]
            break

    beam_arr = remove_white_space(beam_arr, thresh+200)
    target_arr = remove_white_space(target_arr, thresh)
    return beam_arr, target_arr

def remove_white_space(imarray: np.ndarray, thresh: int) -> np.ndarray:
    '''
    Remove all white space from image

    Args:
        imarray: image array
        thresh: threshold for cutting

    Returns:
        trimmed image
    '''
    for i, row in enumerate(imarray):
        if np.mean(row) >= thresh:
            imarray = imarray[i:]
            break
    for i, row in enumerate(reversed(imarray)):
        if np.mean(row) >= thresh:
            imarray = imarray[:(len(imarray)-i)]
            break
    for i, column in enumerate(np.transpose(imarray)):
        if np.mean(column) >= thresh:
            imarray = imarray[:, i:]
            break
    for i, column in enumerate(reversed(np.transpose(imarray))):
        if np.mean(row) >= thresh:
            imarray = imarray[:, :(len(imarray)-i)]
            break
    return imarray

if __name__ == "__main__":
    beam, _ = read_in("shot018_180723.tiff")
    plt.imshow(beam)
    plt.show()
