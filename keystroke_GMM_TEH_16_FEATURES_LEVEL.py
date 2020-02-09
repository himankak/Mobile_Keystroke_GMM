# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 09:51:33 2019

@author: himan
"""

# keystroke_GMM.py

from sklearn.mixture import GaussianMixture
from sklearn import preprocessing
from EER_GMM import evaluateEERGMM
from numpy import array
# matplotlib inline
import matplotlib.pyplot as plt
import pandas
import csv
import numpy as np
import warnings
from scipy import stats

# from matplotlib.patches import Ellipse
# import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")


class GMMDetector:
    # the training(), testing() and evaluateEER() function change, rest all is same.
    def __init__(self, subjects):
        self.user_scores = []
        self.imposter_scores = []
        self.mean_vector = []
        self.gen = []
        self.imp = []
        self.subjects = subjects
        self.new_user_scores = []
        self.new_impostor_scores = []

    """n_components = 4 for MINMAX and 2 for ZSCORE, reg_covar = 1e-4 (minmax) and 1e-4 (zscore)"""
    def training(self):
        self.gmm = GaussianMixture(n_components=2, init_params='random', covariance_type='diag', verbose=False,
                                   max_iter=1000, reg_covar=1e-4, random_state=0)
        #        self.plot_gmm(self.gmm, data)
        self.gmm.fit(self.train)

    def testing(self):
        for i in range(self.test_genuine.shape[0]):
            j = self.test_genuine.iloc[i].values.reshape(1, -1)
            cur_score = self.gmm.score(j)
            # cur_score = preprocessing.normalize(cur_score, norm='l2')
            self.user_scores.append(cur_score)
            self.new_user_scores.append(cur_score)

        # temp_gen = np.array(self.user_scores)
        # print(temp_gen)
        # self.new_user_scores = np.append(self.new_user_scores, temp_gen, axis=0)
        # print(self.new_user_scores)
        # temp_gen = np.delete(temp_gen, np.s_[0:], axis=0)
        # self.user_scores_new = array(self.user_scores)
        # self.user_scores_new = self.user_scores_new.reshape(-1, 1)
        # self.user_scores_new = preprocessing.normalize(self.user_scores_new, norm='l1')

        for i in range(self.test_imposter.shape[0]):
            j = self.test_imposter.iloc[i].values.reshape(1, -1)
            cur_score = self.gmm.score(j)
            # cur_score = preprocessing.normalize(cur_score, norm='l2')
            self.imposter_scores.append(cur_score)
            self.new_impostor_scores.append(cur_score)
        # temp_imp = np.array(self.imposter_scores)
        # print(temp_imp)
        # self.new_impostor_scores = np.append(self.new_impostor_scores, temp_imp, axis=0)
        # print(self.new_impostor_scores)
        # temp_imp = np.delete(temp_imp, np.s_[0:], axis=0)
        # self.imposter_scores_new = array(self.imposter_scores)
        # self.imposter_scores_new = self.imposter_scores_new.reshape(-1, 1)
        # self.imposter_scores_new = preprocessing.normalize(self.imposter_scores_new, norm='l1')

    def evaluate(self):
        eers = []
        for subject in subjects:
            genuine_user_data = data.loc[data.subject == subject, \
                                data.columns.str.startswith("FA")]
            #            data.columns.str.startswith('RRT')        "KDT.1":"FA(n+1).17" ("KDT", "PPT", "PRT", "RPT", "RRT")
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
                                     head(2).loc[:, data.columns.str.startswith("FA")]
            # data.columns.str.startswith('RRT')       "KDT.1":"FA(n+1).17"
            self.training()
            self.testing()
            # with open("D:\\Keystroke\\FAR_FRR_VALUES\\GENUINE_TEH_MIN_MAX_ALL_TIME.csv", "a", newline='') as fp:
            #     wr = csv.writer(fp, dialect='excel')
            #     wr.writerow(self.new_user_scores)
            # with open("D:\\Keystroke\\FAR_FRR_VALUES\\IMPOSTOR_TEH_MIN_MAX_ALL_TIME.csv", "a", newline='') as fp:
            #     wr = csv.writer(fp, dialect='excel')
            #     wr.writerow(self.new_impostor_scores)
            with open("D:\\Keystroke\\FAR_FRR_VALUES\\GENUINE_TEH_Z_SCORE_FA.csv", "a", newline='') as fp:
                wr = csv.writer(fp, dialect='excel')
                wr.writerow(self.new_user_scores)
            with open("D:\\Keystroke\\FAR_FRR_VALUES\\IMPOSTOR_TEH_Z_SCORE_FA.csv", "a", newline='') as fp:
                wr = csv.writer(fp, dialect='excel')
                wr.writerow(self.new_impostor_scores)
            eers.append(evaluateEERGMM(self.user_scores, self.imposter_scores))
            self.new_user_scores.clear()
            self.new_impostor_scores.clear()
        print(eers)
        # print("USER SCORES:", self.new_user_scores)
        # print("SHAPE and SIZE", self.new_user_scores.shape, self.new_user_scores.size)
        # print("IMPOSTER SCORE:", self.new_impostor_scores)
        # print("SHAPE and SIZE", self.new_impostor_scores.shape, self.new_impostor_scores.size)

        # plt.plot(self.user_scores, self.imposter_scores)
        # plt.xlabel('Features')
        # plt.ylabel('Users')
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
# preprocessing for normalization
x = data.as_matrix(columns=data.columns[2:138])
xx = data.as_matrix(columns=data.columns[0:2])
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)  # min_max normalization
z_score = stats.zscore(x, axis=1, ddof=0)  # z_score normalization
print(z_score)
data_user = pandas.DataFrame(xx, index=data.index, columns=data.columns[0:2], dtype=None, copy=True)
data_new = pandas.DataFrame(x_scaled, index=data.index, columns=data.columns[2:138], dtype=None,
                            copy=True)  # min_max dataframe
data_new_z_score = pandas.DataFrame(z_score, index=data.index, columns=data.columns[2:138], dtype=None,
                                    copy=True)  # z_score dataframe
data_final = pandas.concat([data_user, data_new_z_score],
                           axis=1)  # interchange between 'data_new' and 'data_new_z_score' to find results on both tech.
data = data_final
subjects = data["subject"].unique()
print("average EER for GMM detector:")
print(GMMDetector(subjects).evaluate())