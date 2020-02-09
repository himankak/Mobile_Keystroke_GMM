# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 19:27:10 2019

@author: Biomedia4n6
"""

# keystroke_GMM.py

from sklearn.mixture import GaussianMixture
from sklearn import preprocessing
import pandas
import csv
from EER_GMM import evaluateEERGMM
import numpy as np
from scipy import stats
import warnings

warnings.filterwarnings("ignore")


class GMMDetector:
    # the training(), testing() and evaluateEER() function change, rest all is same.

    def __init__(self, subjects):
        self.user_scores = []
        self.imposter_scores = []
        self.mean_vector = []
        self.new_user_scores = []
        self.new_impostor_scores = []
        self.subjects = subjects

    """n_components = 3 for MINMAX and 2 for ZSCORE, rest same"""

    def training(self):
        self.gmm = GaussianMixture(n_components=2, init_params='random', covariance_type='diag', verbose=False,
                                   max_iter=1000, reg_covar=1e-2, random_state=0)
        self.gmm.fit(self.train)

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
        # l = ["pressure", "loc", "touch", "acceleration", "rotation"]
        l = ["touch"]
        regstr = '|'.join(l)

        for subject in subjects:
            genuine_user_data = data.loc[data.subject == subject, \
                                data.columns.str.contains(regstr)]
            #   "0":"press_z_rotation_minus_press_z_rotation_9"
            #   plt.scatter(genuine_user_data[:, 0], genuine_user_data[:, 1], c=labels, s=40, cmap='viridis');
            imposter_data = data.loc[data.subject != subject, :]

            self.train = genuine_user_data[:25]
            self.test_genuine = genuine_user_data[5:]
            self.test_imposter = imposter_data.groupby("subject"). \
                                     head(2).loc[:, data.columns.str.contains(regstr)]
            # startswith(("6", "14", "21", "28", "35", "42", "49", "56"))
            self.training()
            self.testing()
            # with open("D:\\Keystroke\\FAR_FRR_VALUES\\GENUINE_COAKLEY_STDDEV.csv", "a", newline='') as fp:
            #     wr = csv.writer(fp, dialect='excel')
            #     wr.writerow(self.new_user_scores)
            # with open("D:\\Keystroke\\FAR_FRR_VALUES\\IMPOSTOR_COAKLEY_STDDEV.csv", "a", newline='') as fp:
            #     wr = csv.writer(fp, dialect='excel')
            #     wr.writerow(self.new_impostor_scores)
            # with open("D:\\Keystroke\\FAR_FRR_VALUES\\GENUINE_COAKLEY_MIN_MAX_ACCEL.csv", "a", newline='') as fp:
            #     wr = csv.writer(fp, dialect='excel')
            #     wr.writerow(self.new_user_scores)
            # with open("D:\\Keystroke\\FAR_FRR_VALUES\\IMPOSTOR_COAKLEY_MIN_MAX_ACCEL.csv", "a", newline='') as fp:
            #     wr = csv.writer(fp, dialect='excel')
            #     wr.writerow(self.new_impostor_scores)
            with open("D:\\Keystroke\\FAR_FRR_VALUES\\GENUINE_COAKLEY_Z_SCORE_FA.csv", "a", newline='') as fp:
                wr = csv.writer(fp, dialect='excel')
                wr.writerow(self.new_user_scores)
            with open("D:\\Keystroke\\FAR_FRR_VALUES\\IMPOSTOR_COAKLEY_Z_SCORE_FA.csv", "a", newline='') as fp:
                wr = csv.writer(fp, dialect='excel')
                wr.writerow(self.new_impostor_scores)
            eers.append(evaluateEERGMM(self.user_scores, \
                                       self.imposter_scores))
            self.new_user_scores.clear()
            self.new_impostor_scores.clear()
        return np.mean(eers)


path = "D:\\Keystroke\\mobile_pace-sensor_features_Z_SCORE.csv"
data = pandas.read_csv(path)
subjects = data["subject"].unique()
print("average EER for GMM detector:")
print(GMMDetector(subjects).evaluate())
