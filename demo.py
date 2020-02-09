import numpy as np
import csv
import pandas
import matplotlib.rcsetup
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
from matplotlib.ticker import PercentFormatter
from matplotlib.pyplot import figure
import matplotlib.ticker

def plot_FAR_FRR():
    path = 'D:\\Keystroke\\mobile_pace-sensor_features_MIN_MAX.csv'

    FAR_FRR_DF = pandas.read_csv(path)

    x = FAR_FRR_DF.values(columns=FAR_FRR_DF.columns[3:621])

    plt.plot(x)

plot_FAR_FRR()