# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 19:52:25 2019

@author: Biomedia4n6
"""

import numpy as np
import csv
import pandas
import random
from joblib.numpy_pickle_utils import xrange

np.set_printoptions(suppress=True)

def gen_thersholds(n):
    """ Generate a list of  n thresholds between 0.0 and 1.0"""
    thersholds = []
    for x in xrange(1, n+1):
        thersholds.append(float("{:.9f}".format(random.uniform(0.0, 1))))

    return sorted(thersholds)

def evaluateEERGMM(user_scores, imposter_scores):
    """
    1. FOR CMU DATASET: threshold range: (20,51)
    2. FOR COAKLEY DATASET: threshold range: (-3500,-500)
    3. FOR ANTAL DATASET: threshold range: (-1000,1000)
    4. FOR TEH DATASET: threshold range: (-2500,250)
    """
    # FAR = []
    # FRR = []
    counter = 1
    row = []
    global thresh
    # array_size = 0
    # thresholds = gen_thersholds(100)
    thresholds = range(-1500, 250)
    array = np.zeros((len(thresholds), 3))
    # array_new = np.zeros((len(thresholds), 3))
    i = 0
    for th in thresholds:
        g_i = 0
        i_g = 0
        for score in user_scores:
            if score < th:
                g_i = g_i + 1
        for score in imposter_scores:
            if score > th:
                i_g = i_g + 1

        FA = float(i_g) / len(imposter_scores)
        FR = float(g_i) / len(user_scores)
        array[i, 0] = th
        array[i, 1] = FA
        array[i, 2] = FR
        # array_new[i, 0] = th
        # array_new[i, 1] = FA
        # array_new[i, 2] = FR
        row.append(th)
        row.append(FA)
        row.append(FR)
        # with open("D:\\Keystroke\\FAR_FRR_VALUES\\FAR_FRR_TEH_MIN_MAX_ALL_TIME.csv", "a", newline='') as fp:
        #     wr = csv.writer(fp, dialect='excel')
        #     wr.writerow(row)
        #     row.clear()
        with open("D:\\Keystroke\\FAR_FRR_VALUES\\FAR_FRR_TEH_Z_SCORE_FA.csv", "a", newline='') as fp:
            wr = csv.writer(fp, dialect='excel')
            wr.writerow(row)
            row.clear()
        # with open("D:\\Keystroke\\FAR_FRR_VALUES\\FAR_FRR_ANTAL_MIN_MAX_KDT.csv", "a", newline='') as fp:
        #     wr = csv.writer(fp, dialect='excel')
        #     wr.writerow(row)
        #     row.clear()
        # with open("D:\\Keystroke\\FAR_FRR_VALUES\\FAR_FRR_ANTAL_Z_SCORE_KDT.csv", "a", newline='') as fp:
        #     wr = csv.writer(fp, dialect='excel')
        #     wr.writerow(row)
        #     row.clear()
        # with open("D:\\Keystroke\\FAR_FRR_VALUES\\FAR_FRR_COAKLEY_STDDEV.csv", "a", newline='') as fp:
        #     wr = csv.writer(fp, dialect='excel')
        #     wr.writerow(row)
        #     row.clear()
        # with open("D:\\Keystroke\\FAR_FRR_VALUES\\FAR_FRR_COAKLEY_MIN_MAX_ACCEL.csv", "a", newline='') as fp:
        #     wr = csv.writer(fp, dialect='excel')
        #     wr.writerow(row)
        #     row.clear()
        # with open("D:\\Keystroke\\FAR_FRR_VALUES\\FAR_FRR_COAKLEY_Z_SCORE_FA.csv", "a", newline='') as fp:
        #     wr = csv.writer(fp, dialect='excel')
        #     wr.writerow(row)
        #     row.clear()
        i = i + 1
    # with open("D:\\Keystroke\\FAR_FRR_VALUES\\FAR_FRR_TEH_MIN_MAX.csv", "a", newline='') as fp:
    #     wr = csv.writer(fp, dialect='excel')
    #     wr.writerow(row)
    #     row.clear()

    # with open("D:\\Keystroke\\FAR_FRR_VALUES\\FAR_FRR_ANTAL_MIN_MAX.csv", "a", newline='') as fp:
    #     wr = csv.writer(fp, dialect='excel')
    #     wr.writerow(row)
    #     row.clear()
    # with open("D:\\Keystroke\\FAR_FRR_VALUES\\FAR_FRR_COAKLEY_MIN_MAX.csv", "a", newline='') as fp:
    #     wr = csv.writer(fp, dialect='excel')
    #     wr.writerow(row)
    #     row.clear()
    # with open("D:\\Keystroke\\FAR_FRR_VALUES\\FAR_FRR_TEH_MIN_MAX.csv", "a", newline='') as fp:
    #     wr = csv.writer(fp, dialect='excel')
    #     # wr.writerow(row)
    #     for j in range(array_new.shape[0]):
    #         if (array_new[j, 0] or array_new[j, 1] or array_new[j, 2]) != 0:
    #             row.append(array_new[j, 0])
    #             row.append(array_new[j, 1])
    #             row.append(array_new[j, 2])
    #             wr.writerow(row)
    #         row.clear()

    # with open("D:\\Keystroke\\FAR_FRR_VALUES\\FAR_FRR_TEH_Z_SCORE.csv", "a", newline='') as fp:
    #     wr = csv.writer(fp, dialect='excel')
    #     for j in range(array_new.shape[0]):
    #         if (array_new[j, 0] or array_new[j, 1] or array_new[j, 2]) != 0:
    #             row.append(array_new[j, 0])
    #             row.append(array_new[j, 1])
    #             row.append(array_new[j, 2])
    #             wr.writerow(row)
    #         row.clear()
    #
    for j in range(array.shape[0]):
        if array[j, 1] < array[j, 2]:
            thresh = (array[j, 0] + array[j - 1, 0]) / 2
            break
    g_i = 0
    i_g = 0

    for score in user_scores:
        if score < thresh:
            g_i = g_i + 1
    for score in imposter_scores:
        if score > thresh:
            i_g = i_g + 1

    FA = float(i_g) / len(imposter_scores)
    FR = float(g_i) / len(user_scores)
    print("USER SCORE:", user_scores)
    print("IMPOSTER SCORE:", imposter_scores)
    print("FAR SCORE:", FA)
    print("FRR SCORE:", FR)

    # row.append(counter)
    # row.append(FA)
    # row.append(FR)
    # with open("D:\\Keystroke\\FAR_FRR_VALUES\\FAR_FRR_TEH_MIN_MAX.csv", "a", newline='') as fp:
    #     wr = csv.writer(fp, dialect='excel')
    #     wr.writerow(row)

    # with open("D:\\Keystroke\\FAR_FRR_VALUES\\FAR_FRR_TEH_Z_SCORE.csv", "a", newline='') as fp:
    #     wr = csv.writer(fp, dialect='excel')
    #     wr.writerow(row)

    # with open("D:\\Keystroke\\FAR_FRR_VALUES\\FAR_FRR_ANTAL_MIN_MAX.csv", "a", newline='') as fp:
    #     wr = csv.writer(fp, dialect='excel')
    #     wr.writerow(row)

    # with open("D:\\Keystroke\\FAR_FRR_VALUES\\FAR_FRR_CMU_MIN_MAX.csv", "a", newline='') as fp:
    #     wr = csv.writer(fp, dialect='excel')
    #     wr.writerow(row)

    # with open("D:\\Keystroke\\FAR_FRR_VALUES\\FAR_FRR_ANTAL_Z_SCORE.csv", "a", newline='') as fp:
    #     wr = csv.writer(fp, dialect='excel')
    #     wr.writerow(row)
    return (FA + FR) / 2
