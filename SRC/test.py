# xdf file importation
import os
import pyxdf
import numpy as np
from scipy.signal import welch


# Ensure that the root directory is the project directory

# if pwd is SRC  change to root
print("Current working directory: ", os.getcwd())
if os.path.basename(os.getcwd()) == "SRC":
    os.chdir("..")
print("Current working directory: ", os.getcwd())
print(os.path.basename(os.getcwd()))

# =============================================================================
################################## initialization #############################
# =============================================================================
# Define the xdf file path
FILENAME = "001_MolLud_20201112_1_c.xdf"
# FILENAME = "020_DesMar_20211129_1_c.xdf"
path = os.path.normpath("./DAT/INPUT/"+FILENAME)


# Load only streams of interest (EEG signal and Mouse task Markers) from the xdf data file
# data, header = pyxdf.load_xdf(path, select_streams=
# [{'type': 'EEG', 'name': 'LSLOutletStreamName-EEG'},{'type': 'Markers', 'name': 'MouseToNIC'}] )
data, header = pyxdf.load_xdf(path)