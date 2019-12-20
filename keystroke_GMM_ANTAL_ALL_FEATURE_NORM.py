# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 19:27:10 2019

@author: Biomedia4n6
"""

#keystroke_GMM.py

from sklearn.mixture import GaussianMixture
from sklearn import preprocessing
import pandas
from EER_GMM import evaluateEERGMM
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class GMMDetector:
#the training(), testing() and evaluateEER() function change, rest all is same.

    def __init__(self, subjects):
        self.user_scores = []
        self.imposter_scores = []
        self.mean_vector = []
        self.subjects = subjects
        
    def training(self):
        self.gmm = GaussianMixture(n_components=4, covariance_type='diag', verbose=False, random_state=60)
        self.gmm.fit(self.train)
        
#    def draw_ellipse(position, covariance, ax=None, **kwargs):
#        """Draw an ellipse with a given position and covariance"""
#        ax = ax or plt.gca()
#    # Convert covariance to principal axes
#        if covariance.shape == (2, 2):
#            U, s, Vt = np.linalg.svd(covariance)
#            angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
#            width, height = 2 * np.sqrt(s)
#        else:
#            angle = 0
#            width, height = 2 * np.sqrt(covariance)
#    
#    # Draw the Ellipse
#    for nsig in range(1, 4):
#        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
#                             angle, **kwargs))
#        
#    def plot_gmm(gmm, X, label=True, ax=None):
#        ax = ax or plt.gca()
#        labels = gmm.fit(X).predict(X)
#        if label:
#            ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
#        else:
#            ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
#            ax.axis('equal')
#        w_factor = 0.2 / gmm.weights_.max()
#        for pos, covar, w in zip(gmm.means_, gmm.covars_, gmm.weights_):
#            draw_ellipse(pos, covar, alpha=w * w_factor)
 
    def testing(self):
        for i in range(self.test_genuine.shape[0]):
            j = self.test_genuine.iloc[i].values
            cur_score = self.gmm.score(j.reshape(1, -1))
            self.user_scores.append(cur_score)
 
        for i in range(self.test_imposter.shape[0]):
            j = self.test_imposter.iloc[i].values
            cur_score = self.gmm.score(j.reshape(1, -1))
            self.imposter_scores.append(cur_score)
 
#    def evaluate(self):
#        eers = []
# 
#        for subject in subjects:        
#            genuine_user_data = data.loc[data.subject == subject, \
#                                         "H.period":"H.Return"]
#            imposter_data = data.loc[data.subject != subject, :]
#            
#            self.train = genuine_user_data[:200]
#            self.test_genuine = genuine_user_data[200:]
#            self.test_imposter = imposter_data.groupby("subject"). \
#                                 head(5).loc[:, "H.period":"H.Return"]
# 
#            self.training()
#            self.testing()
#            eers.append(evaluateEERGMM(self.user_scores, \
#                                     self.imposter_scores))
#        return np.mean(eers)
#    
#path = "D:\\Keystroke\\keystroke.csv" 
#data = pandas.read_csv(path)
#subjects = data["subject"].unique()
#print("average EER for GMM detector:")
#print(GMMDetector(subjects).evaluate())

    def evaluate(self):
        eers = []
        user_scores_all = []
        imposter_scores_all = []

        for subject in subjects:
            genuine_user_data = data.loc[data.subject == subject, \
                                "holdtime1":"totaldistance"]
#            plt.scatter(genuine_user_data[:, 0], genuine_user_data[:, 1], c=labels, s=40, cmap='viridis');
            imposter_data = data.loc[data.subject != subject, :]

            self.train = genuine_user_data[:50]
            self.test_genuine = genuine_user_data[10:]
            self.test_imposter = imposter_data.groupby("subject"). \
                                     head(10).loc[:, "holdtime1":"totaldistance"]

            self.training()
            self.testing()
            print("USER:", self.user_scores)
            print("IMPOSTER:", self.imposter_scores)
            user_scores_all.append(self.user_scores)
            imposter_scores_all.append(self.imposter_scores)
            eers.append(evaluateEERGMM(self.user_scores, \
                                       self.imposter_scores))
        return np.mean(eers)


path = "D:\\Keystroke\\logicalstrong.csv"
data = pandas.read_csv(path)
#preprocessing for normalization
x = data.as_matrix(columns=data.columns[1:73])
xx = data.as_matrix(columns=data.columns[0:1])
min_max_scaler = preprocessing.MinMaxScaler()       #min_max normalization
x_scaled = min_max_scaler.fit_transform(x)
print(x_scaled)
data_user = pandas.DataFrame(xx, index=data.index, columns=data.columns[0:1], dtype=None, copy=True)
data_new = pandas.DataFrame(x_scaled, index=data.index, columns=data.columns[1:73], dtype=None, copy=True)
data_final = pandas.concat([data_user, data_new], axis=1)
data = data_final
# slc = np.r_[data.subject:data.subject]
# data["subject"].astype(int)
print(data)
subjects = data["subject"].astype(int).unique()
print("average EER for GMM detector:")
print(GMMDetector(subjects).evaluate())