import cv2
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

from os import path

def hmread(path):
    hm = cv2.imread(path)
    hm = cv2.cvtColor(hm, cv2.COLOR_BGR2GRAY)
    return hm

def read_csv_from_root(root):
    dfs = [pd.read_csv(path) for path in glob.glob(root)]
    return dfs
