# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 19:27:10 2019

@author: Biomedia4n6
"""

#keystroke_GMM.py

from sklearn.mixture import GaussianMixture
from sklearn import preprocessing
from scipy import stats
import pandas 
from EER_GMM import evaluateEERGMM
import numpy as np
import warnings
import csv
warnings.filterwarnings("ignore")

class GMMDetector:
#the training(), testing() and evaluateEER() function change, rest all is same.

    def __init__(self, subjects):
        self.user_scores = []
        self.imposter_scores = []
        self.mean_vector = []
        self.subjects = subjects
        self.new_user_scores = []
        self.new_impostor_scores = []
        
    def training(self):
        self.gmm = GaussianMixture(n_components=2, covariance_type='diag',
                        verbose=False)
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
            self.new_user_scores.append(cur_score)
 
        for i in range(self.test_imposter.shape[0]):
            j = self.test_imposter.iloc[i].values
            cur_score = self.gmm.score(j.reshape(1, -1))
            self.imposter_scores.append(cur_score)
            self.new_impostor_scores.append(cur_score)
 
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

        for subject in subjects:
            genuine_user_data = data.loc[data.subject == subject, \
                                "H.period":"H.Return"]
#            plt.scatter(genuine_user_data[:, 0], genuine_user_data[:, 1], c=labels, s=40, cmap='viridis');
            imposter_data = data.loc[data.subject != subject, :]

            self.train = genuine_user_data[:200]
            self.test_genuine = genuine_user_data[200:]
            self.test_imposter = imposter_data.groupby("subject"). \
                                     head(5).loc[:, "H.period":"H.Return"]

            self.training()
            self.testing()
            with open("D:\\Keystroke\\FAR_FRR_VALUES\\GENUINE_CMU_MIN_MAX.csv", "a", newline='') as fp:
                wr = csv.writer(fp, dialect='excel')
                wr.writerow(self.new_user_scores)
            with open("D:\\Keystroke\\FAR_FRR_VALUES\\IMPOSTOR_CMU_MIN_MAX.csv", "a", newline='') as fp:
                wr = csv.writer(fp, dialect='excel')
                wr.writerow(self.new_impostor_scores)
            eers.append(evaluateEERGMM(self.new_user_scores, \
                                       self.new_impostor_scores))
            self.new_user_scores.clear()
            self.new_impostor_scores.clear()
        return np.mean(eers)


path = "D:\\Keystroke\\keystroke.csv"
data = pandas.read_csv(path)
#preprocessing for normalization
x = data.as_matrix(columns=data.columns[3:34])
xx = data.as_matrix(columns=data.columns[0:3])
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)              #min_max normalization
z_score = stats.zscore(x, axis=1, ddof=0)               #z_score normalization
print(x_scaled)
data_user = pandas.DataFrame(xx, index=data.index, columns=data.columns[0:3], dtype=None, copy=True)
data_new = pandas.DataFrame(x_scaled, index=data.index, columns=data.columns[3:34], dtype=None, copy=True)
data_new_z_score = pandas.DataFrame(z_score, index=data.index, columns=data.columns[3:34], dtype=None, copy=True)      #z_score dataframe
data_final = pandas.concat([data_user, data_new], axis=1)               #interchange between data_new and data_new_z_score to find results on both tech.
data = data_final
# slc = np.r_[data.subject:data.subject]
# data["subject"].astype(int)
print(data)
subjects = data["subject"].unique()
print("average EER for GMM detector:")
print(GMMDetector(subjects).evaluate())