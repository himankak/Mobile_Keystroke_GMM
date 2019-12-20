# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 19:52:25 2019

@author: Biomedia4n6
"""

import numpy as np
np.set_printoptions(suppress = True)

def evaluateEERGMM(user_scores, imposter_scores):
    """
    1. FOR CMU DATASET: threshold range: (20,51)
    2. FOR COAKLEY DATASET: threshold range: (20,100)
    3. FOR ANTAL DATASET: threshold range: (0,6)
    4. FOR TEH DATASET: threshold range: (20,51)
    """
    thresholds = range(-150, 150)
    array = np.zeros((len(thresholds), 3))
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
        i = i + 1
    
    print("USER SCORE:", user_scores)
    print("IMPOSTER SCORE:", imposter_scores)
    
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
    print("FAR SCORE:", FA)
    print("FRR SCORE:", FR)
    return (FA + FR) /2