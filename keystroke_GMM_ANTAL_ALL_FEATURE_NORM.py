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
from scipy import stats
import csv
import warnings
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

    """n_component = 10 (minmax) & = 55 (zscore) reg_covar = 1e-12 (minmax) and 1e-2 (zscore) for the whole dataset.. FEATURE FUSION"""
        
    def training(self):
        self.gmm = GaussianMixture(n_components=55, init_params='random', covariance_type='diag', verbose=False,
                                   max_iter=1000, reg_covar=1e-12, random_state=1000)
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

    def evaluate(self):
        eers = []

        for subject in subjects:
            genuine_user_data = data.loc[data.subject == subject, \
                                data.columns.str.startswith("holdtime")]
            # "holdtime1": "totaldistance" (("hold", "up", "down"))  "pressure", "fingerarea", "meanp", "meanfin",
            # "totaldis", "vel" plt.scatter(genuine_user_data[:, 0], genuine_user_data[:, 1], c=labels, s=40,
            # cmap='viridis'); "meanxacc", "meanyacc", "meanzacc"
            imposter_data = data.loc[data.subject != subject, :]

            self.train = genuine_user_data[:55]
            self.test_genuine = genuine_user_data[5:]
            self.test_imposter = imposter_data.groupby("subject"). \
                                     head(2).loc[:, data.columns.str.startswith("holdtime")]

            self.training()
            self.testing()
            # with open("D:\\Keystroke\\FAR_FRR_VALUES\\GENUINE_ANTAL_MIN_MAX_KDT.csv", "a", newline='') as fp:
            #     wr = csv.writer(fp, dialect='excel')
            #     wr.writerow(self.new_user_scores)
            # with open("D:\\Keystroke\\FAR_FRR_VALUES\\IMPOSTOR_ANTAL_MIN_MAX_KDT.csv", "a", newline='') as fp:
            #     wr = csv.writer(fp, dialect='excel')
            #     wr.writerow(self.new_impostor_scores)
            with open("D:\\Keystroke\\FAR_FRR_VALUES\\GENUINE_ANTAL_Z_SCORE_KDT.csv", "a", newline='') as fp:
                wr = csv.writer(fp, dialect='excel')
                wr.writerow(self.new_user_scores)
            with open("D:\\Keystroke\\FAR_FRR_VALUES\\IMPOSTOR_ANTAL_Z_SCORE_KDT.csv", "a", newline='') as fp:
                wr = csv.writer(fp, dialect='excel')
                wr.writerow(self.new_impostor_scores)
            eers.append(evaluateEERGMM(self.user_scores, \
                                       self.imposter_scores))
            self.new_user_scores.clear()
            self.new_impostor_scores.clear()
        return np.mean(eers)


path = "D:\\Keystroke\\logicalstrong.csv"
data = pandas.read_csv(path)
#preprocessing for normalization
x = data.as_matrix(columns=data.columns[1:73])
xx = data.as_matrix(columns=data.columns[0:1])
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)              #min_max normalization
z_score = stats.zscore(x, axis=1, ddof=0)               #z_score normalization
print(x_scaled)
data_user = pandas.DataFrame(xx, index=data.index, columns=data.columns[0:1], dtype=None, copy=True)
data_new = pandas.DataFrame(x_scaled, index=data.index, columns=data.columns[1:73], dtype=None, copy=True)
data_new_z_score = pandas.DataFrame(z_score, index=data.index, columns=data.columns[1:73], dtype=None, copy=True)      #z_score dataframe
data_final = pandas.concat([data_user, data_new_z_score], axis=1)               #interchange between data_new and data_new_z_score to find results on both tech.
data = data_final
# slc = np.r_[data.subject:data.subject]
# data["subject"].astype(int)
print(data)
subjects = data["subject"].astype(int).unique()
print("average EER for GMM detector:")
print(GMMDetector(subjects).evaluate())