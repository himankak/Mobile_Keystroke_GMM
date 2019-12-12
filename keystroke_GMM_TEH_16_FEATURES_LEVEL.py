# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 09:51:33 2019

@author: himan
"""

#keystroke_GMM.py

from sklearn.mixture import GaussianMixture
from sklearn import preprocessing
from EER_GMM import evaluateEERGMM
from numpy import array
import pandas
import numpy as np
import warnings
#from matplotlib.patches import Ellipse
#import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

class GMMDetector:
#the training(), testing() and evaluateEER() function change, rest all is same.            
    def __init__(self, subjects):
        self.user_scores = []
        self.imposter_scores = []
        self.mean_vector = []
        # self.user_scores_new = []
        # self.imposter_scores_new = []
        self.subjects = subjects
        
    def training(self):
        self.gmm = GaussianMixture(n_components = 3, covariance_type = 'diag', verbose = False, random_state = 150)
#        self.plot_gmm(self.gmm, data)
        self.gmm.fit(self.train)
 
    def testing(self):
        for i in range(self.test_genuine.shape[0]):
            j = self.test_genuine.iloc[i].values.reshape(1, -1)
            cur_score = self.gmm.score(j)
            # cur_score = preprocessing.normalize(cur_score, norm='l2')
            self.user_scores.append(cur_score)

        # self.user_scores_new = array(self.user_scores)
        # self.user_scores_new = self.user_scores_new.reshape(-1, 1)
        # self.user_scores_new = preprocessing.normalize(self.user_scores_new, norm='l1')
 
        for i in range(self.test_imposter.shape[0]):
            j = self.test_imposter.iloc[i].values.reshape(1, -1)
            cur_score = self.gmm.score(j)
            # cur_score = preprocessing.normalize(cur_score, norm='l2')
            self.imposter_scores.append(cur_score)

        # self.imposter_scores_new = array(self.imposter_scores)
        # self.imposter_scores_new = self.imposter_scores_new.reshape(-1, 1)
        # self.imposter_scores_new = preprocessing.normalize(self.imposter_scores_new, norm='l1')
    
    def evaluate(self):
        eers = []
 
        for subject in subjects:        
            genuine_user_data = data.loc[data.subject == subject, \
                                         "KDT.1":"FA(n+1).17"]
#            data.columns.str.startswith('RRT')
            """DATA NORMALIZATION OF GENUINE DATA"""
            """1. MAX"""
            # genuine_user_data = genuine_user_data/genuine_user_data.max()
            """2. MIN_MAX"""
            # genuine_user_data = (genuine_user_data-genuine_user_data.min())/(genuine_user_data.max()-genuine_user_data.min())
            """3. Z_SCORE"""
            # genuine_user_data = (genuine_user_data-genuine_user_data.mean())/genuine_user_data.std()

            """DATA NORMALIZATION OF IMPOSTOR DATA"""
            imposter_data = data.loc[data.subject != subject, :]
            """1. MAX"""
            # imposter_data = imposter_data/imposter_data.max()
            """2. MIN_MAX"""
            # imposter_data = (imposter_data-imposter_data.min())/(imposter_data.max()-imposter_data.min())
            """3. Z_SCORE"""
            # imposter_data = (imposter_data-imposter_data.mean())/imposter_data.std()

            self.train = genuine_user_data[:7]
            self.test_genuine = genuine_user_data[3:]
            self.test_imposter = imposter_data.groupby("subject"). \
                                 head(3).loc[:, "KDT.1":"FA(n+1).17"]
# data.columns.str.startswith('RRT')
            self.training()
            self.testing()
            eers.append(evaluateEERGMM(self.user_scores, self.imposter_scores))
        print(eers)
        return np.mean(eers)
    
#    def draw_ellipse(position, covariance, ax=None, **kwargs):
##        """Draw an ellipse with a given position and covariance"""
#        ax = ax or plt.gca()
#        # Convert covariance to principal axes
#        if covariance.shape == (2, 2):
#            U, s, Vt = np.linalg.svd(covariance)
#            angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
#            width, height = 2 * np.sqrt(s)
#        else:
#            angle = 0
#            width, height = 2 * np.sqrt(covariance)
#    
#        # Draw the Ellipse
#        for nsig in range(1, 4):
#            ax.add_patch(Ellipse(position, nsig * width, nsig * height,
#                             angle, **kwargs))
#    
#    def plot_gmm(gmm, X, label=True, ax=None):
#        ax = ax or plt.gca()
#        labels = gmm.fit(data).predict(X)
#        if label:
#            ax.scatter(data[:, 0], data[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
#        else:
#            ax.scatter(data[:, 0], data[:, 1], s=40, zorder=2)
#            ax.axis('equal')
#    
#        w_factor = 0.2 / gmm.weights_.max()
#        for pos, covar, w in zip(gmm.means_, gmm.covars_, gmm.weights_):
#            self.draw_ellipse(pos, covar, alpha=w * w_factor)


path = "D:\\Keystroke\\TEH_FEATURES.csv" 
data = pandas.read_csv(path)
subjects = data["subject"].unique()
# x = data.loc["KDT.1":"FA(n+1).17"].values       #returns a numpy array
x = data.as_matrix(columns=data.columns[2:137])
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
print(x_scaled)
data.loc["KDT.1":] = pandas.DataFrame(x_scaled)
print("average EER for GMM detector:")
print(GMMDetector(subjects).evaluate())