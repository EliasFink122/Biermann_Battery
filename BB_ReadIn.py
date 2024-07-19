"""
Created on Fri Jul 19 2024

@author: Elias Fink (elias.fink22@imperial.ac.uk)

Read in proton position files

Methods:
    read_in:
        read in text results file
    plot_data:
        plot all data
"""
import numpy as np
import matplotlib.pyplot as plt

def read_in(path = "results.txt"):
    '''
    Read in txt file

    Args:
        path: full path to file

    Returns:
        data array
    '''
    data = np.loadtxt(path)
    return data

def plot_data(data):
    '''
    Plot text file data on histogram
    '''
    plt.figure()
    plt.title("Simulated RCF")
    plt.hist2d(data[:, 0]*1000, data[:, 1]*1000, bins = 1000)
    plt.xlabel("x [mm]")
    plt.ylabel("y [mm]")
    plt.show()

if __name__ == "__main__":
    positions = read_in()
    plot_data(positions)
