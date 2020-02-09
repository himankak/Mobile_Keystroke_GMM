import numpy as np
import csv
import pandas
import matplotlib.rcsetup
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
from matplotlib.ticker import PercentFormatter
from matplotlib.pyplot import figure
import matplotlib.ticker

np.set_printoptions(suppress=True)

font = {'family': 'Courier New',
        'weight': 'bold',
        'size': 30}

matplotlib.rc('font', **font)


def plot_FAR_FRR():
    # path = "D:\\Keystroke\\FAR_FRR_VALUES\\ROC PLOTS\\TEH\\FEATURE FUSION\\FAR_FRR_TEH_MIN_MAX.csv"
    # path1 = "D:\\Keystroke\\FAR_FRR_VALUES\\ROC PLOTS\\TEH\\FEATURE FUSION\\GENUINE_TEH_MIN_MAX.csv"
    # path2 = "D:\\Keystroke\\FAR_FRR_VALUES\\ROC PLOTS\\TEH\\FEATURE FUSION\\IMPOSTOR_TEH_MIN_MAX.csv"

    # path = "D:\\Keystroke\\FAR_FRR_VALUES\\ROC PLOTS\\TEH\\FEATURE FUSION\\FAR_FRR_TEH_Z_SCORE.csv"
    # path1 = "D:\\Keystroke\\FAR_FRR_VALUES\\ROC PLOTS\\TEH\\FEATURE FUSION\\GENUINE_TEH_Z_SCORE.csv"
    # path2 = "D:\\Keystroke\\FAR_FRR_VALUES\\ROC PLOTS\\TEH\\FEATURE FUSION\\IMPOSTOR_TEH_Z_SCORE.csv"

    # path = "D:\\Keystroke\\FAR_FRR_VALUES\\ROC PLOTS\\ANTAL\\FEATURE FUSION\\FAR_FRR_ANTAL_MIN_MAX.csv"
    # path1 = "D:\\Keystroke\\FAR_FRR_VALUES\\ROC PLOTS\\ANTAL\\FEATURE FUSION\\GENUINE_ANTAL_MIN_MAX.csv"
    # path2 = "D:\\Keystroke\\FAR_FRR_VALUES\\ROC PLOTS\\ANTAL\\FEATURE FUSION\\IMPOSTOR_ANTAL_MIN_MAX.csv"

    # path = "D:\\Keystroke\\FAR_FRR_VALUES\\ROC PLOTS\\ANTAL\\FEATURE FUSION\\FAR_FRR_ANTAL_Z_SCORE.csv"
    # path1 = "D:\\Keystroke\\FAR_FRR_VALUES\\ROC PLOTS\\ANTAL\\FEATURE FUSION\\GENUINE_ANTAL_Z_SCORE.csv"
    # path2 = "D:\\Keystroke\\FAR_FRR_VALUES\\ROC PLOTS\\ANTAL\\FEATURE FUSION\\IMPOSTOR_ANTAL_Z_SCORE.csv"

    # path = "D:\\Keystroke\\FAR_FRR_VALUES\\ROC PLOTS\\COAKLEY\\FEATURE FUSION\\ALL\\FAR_FRR_COAKLEY_MIN_MAX.csv"
    # path1 = "D:\\Keystroke\\FAR_FRR_VALUES\\ROC PLOTS\\COAKLEY\\FEATURE FUSION\\ALL\\GENUINE_COAKLEY_MIN_MAX.csv"
    # path2 = "D:\\Keystroke\\FAR_FRR_VALUES\\ROC PLOTS\\COAKLEY\\FEATURE FUSION\\ALL\\IMPOSTOR_COAKLEY_MIN_MAX.csv"

    # path = "D:\\Keystroke\\FAR_FRR_VALUES\\ROC PLOTS\\COAKLEY\\FEATURE FUSION\\ALL\\FAR_FRR_COAKLEY_Z_SCORE_NEW.csv"
    # path1 = "D:\\Keystroke\\FAR_FRR_VALUES\\ROC PLOTS\\COAKLEY\\FEATURE FUSION\\ALL\\GENUINE_COAKLEY_Z_SCORE.csv"
    # path2 = "D:\\Keystroke\\FAR_FRR_VALUES\\ROC PLOTS\\COAKLEY\\FEATURE FUSION\\ALL\\IMPOSTOR_COAKLEY_Z_SCORE.csv"

    # path = "D:\\Keystroke\\FAR_FRR_VALUES\\FAR_FRR_CMU_MIN_MAX.csv"
    # path1 = "D:\\Keystroke\\FAR_FRR_VALUES\\GENUINE_CMU_MIN_MAX.csv"
    # path2 = "D:\\Keystroke\\FAR_FRR_VALUES\\IMPOSTOR_CMU_MIN_MAX.csv"

    """USER IMPOSTOR SCORE PLOTS"""
    # gen = pandas.read_csv(path1)
    # imp = pandas.read_csv(path2)
    # gen.T.plot()
    # plt.show()
    # imp.T.plot()
    # plt.show()
    # columns = ['subject', 'FAR', 'FRR']

    """ROC PLOTS"""
    # path = 'D:\\Keystroke\\FAR_FRR_VALUES\\ROC PLOTS\\COMMON\\ALL\\FAR_FRR_MIN_MAX.csv'
    # path = 'D:\\Keystroke\\FAR_FRR_VALUES\\ROC PLOTS\\COMMON\\ALL\\FAR_FRR_Z_SCORE.csv'

    # path = 'D:\\Keystroke\\FAR_FRR_VALUES\\ROC PLOTS\\ANTAL\\FEATURE FUSION\\COMMON_MIN\\FAR_FRR_ANTAL_MIN_MAX.csv'
    # path = 'D:\\Keystroke\\FAR_FRR_VALUES\\ROC PLOTS\\ANTAL\\FEATURE FUSION\\COMMON_Z\\FAR_FRR_ANTAL_Z_SCORE.csv'

    # path = 'D:\\Keystroke\\FAR_FRR_VALUES\\ROC PLOTS\\COAKLEY\\FEATURE FUSION\\COMMON MIN\\TIME\\FAR_FRR_COAKLEY_MIN_MAX.csv'
    # path = 'D:\\Keystroke\\FAR_FRR_VALUES\\ROC PLOTS\\COAKLEY\\FEATURE FUSION\\COMMON Z\\TIME\\FAR_FRR_COAKLEY_Z_SCORE.csv'

    # path = 'D:\\Keystroke\\FAR_FRR_VALUES\\ROC PLOTS\\COAKLEY\\FEATURE FUSION\\COMMON MIN\\TOUCH MOTION\\FAR_FRR_TM_COAKLEY_MIN_MAX.csv'
    # path = 'D:\\Keystroke\\FAR_FRR_VALUES\\ROC PLOTS\\COAKLEY\\FEATURE FUSION\\COMMON Z\\TOUCH MOTION\\FAR_FRR_TM_COAKLEY_Z_SCORE.csv'

    path = 'D:\\Keystroke\\FAR_FRR_VALUES\\ROC PLOTS\\TEH\\FEATURE FUSION\\COMMON_MIN\\FAR_FRR_TEH_MIN_MAX.csv'
    # path = 'D:\\Keystroke\\FAR_FRR_VALUES\\ROC PLOTS\\TEH\\FEATURE FUSION\\COMMON_Z\\FAR_FRR_TEH_Z_SCORE.csv'

    FAR_FRR_DF = pandas.read_csv(path)
    # FAR_FRR_DF['FAR'] = FAR_FRR_DF['FAR'].astype(float).map(lambda n: '{:.02%}'.format(n))
    # FAR_FRR_DF['FAR'] = FAR_FRR_DF['FAR'].astype(float).map("{:.02%}".format)
    # FAR_FRR_DF['FRR'] = FAR_FRR_DF['FRR'].astype(float).map(lambda n: '{:.02%}'.format(n))
    # FAR_FRR_DF['FRR'] = FAR_FRR_DF['FRR'].astype(float).map("{:.02%}".format)
    # for i in range(1, 1200):
    #     # if i == 0:
    #     #     print("FIRST NULL VALUE COUNTER EXEC")
    #     # else:
    #     FAR_list = FAR_FRR_DF['FAR.'+str(i)].tolist()
    #     FRR_list = FAR_FRR_DF['FRR.'+str(i)].tolist()
    #     plt.plot(FRR_list, FAR_list)
    #     FAR_list.clear()
    #     FRR_list.clear()
    # print("FAR=", FAR_list)
    # print("FRR=", FRR_list)
    # plt.xlim([0, 0.05])
    # plt.ylim([0, 0.05])

    figure(num=None, figsize=(10, 9), facecolor='w', edgecolor='k')
    FAR_list1 = (FAR_FRR_DF['FAR1']).tolist()
    FRR_list1 = (FAR_FRR_DF['FRR1']).tolist()
    FAR_list2 = (FAR_FRR_DF['FAR2']).tolist()
    FRR_list2 = (FAR_FRR_DF['FRR2']).tolist()
    FAR_list3 = (FAR_FRR_DF['FAR3']).tolist()
    FRR_list3 = (FAR_FRR_DF['FRR3']).tolist()
    FAR_list4 = (FAR_FRR_DF['FAR4']).tolist()
    FRR_list4 = (FAR_FRR_DF['FRR4']).tolist()
    FAR_list5 = (FAR_FRR_DF['FAR5']).tolist()
    FRR_list5 = (FAR_FRR_DF['FRR5']).tolist()
    FAR_list6 = (FAR_FRR_DF['FAR6']).tolist()
    FRR_list6 = (FAR_FRR_DF['FRR6']).tolist()
    FAR_list7 = (FAR_FRR_DF['FAR7']).tolist()
    FRR_list7 = (FAR_FRR_DF['FRR7']).tolist()
    # FAR_list8 = (FAR_FRR_DF['FAR8']).tolist()
    # FRR_list8 = (FAR_FRR_DF['FRR8']).tolist()

    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1, decimals=False))
    plt.gca().xaxis.set_major_formatter(PercentFormatter(1, decimals=False))
    # plt.xaxis.set_ticks(np.arange(0, 25, 2.5))
    # plt.yaxis.set_ticks(np.arange(0, 25, 2.5))

    x = np.linspace(*plt.xlim())
    # plt.plot(x / 1.4, x / 1.4)
    plt.plot(x/5, x/5)

    """ROC CURVE FOR MINMAX AND Z SCORE FOR COMPLETE THREE DATASETS"""
    # plt.plot(FRR_list1, FAR_list1, linewidth=3, label='Antal et al.', linestyle=':')
    # plt.plot(FRR_list2, FAR_list2, linewidth=3, label='Coakley et al.', linestyle='--')
    # plt.plot(FRR_list3, FAR_list3, linewidth=3, label='Teh et al.', linestyle='-')
    # marker='*', markerfacecolor='green', markersize=0.5,
    # plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8])  # , 6, 7, 8, 9, 10
    # plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8])

    """ROC CURVE FOR MINMAX AND Z SCORE FOR DIFFERENT FEATURE VECTORS ANTAL et al."""
    # plt.plot(FRR_list1, FAR_list1, linewidth=3, label='KDTS', linestyle=':')
    # plt.plot(FRR_list2, FAR_list2, linewidth=3, label='KPPTS', linestyle='--')
    # plt.plot(FRR_list3, FAR_list3, linewidth=3, label='KRPTS', linestyle='-')
    # plt.plot(FRR_list4, FAR_list4, linewidth=3, label='TPV', linestyle='-.')
    # plt.plot(FRR_list5, FAR_list5, label='TFA', marker='s', markerfacecolor='green', markersize=6)
    # plt.plot(FRR_list6, FAR_list6, label='All Motion', marker='x', markerfacecolor='black', markersize=6)
    # plt.plot(FRR_list7, FAR_list7, label='All Time', marker='o', markerfacecolor='cyan', markersize=6)
    # plt.plot(FRR_list8, FAR_list8, label='All Touch', marker='D', markerfacecolor='yellow',  markersize=6)

    """ROC CURVE FOR MINMAX AND Z SCORE FOR DIFFERENT TIMING FEATURE VECTORS COAKLEY et al."""
    # plt.plot(FRR_list1, FAR_list1, linewidth=3, label='KDTS', linestyle=':')
    # plt.plot(FRR_list2, FAR_list2, linewidth=3, label='KPPTS', linestyle='--')
    # plt.plot(FRR_list3, FAR_list3, linewidth=3, label='KPRTS', linestyle='-')
    # plt.plot(FRR_list4, FAR_list4, linewidth=3, label='KRPTS', linestyle='-.')
    # plt.plot(FRR_list5, FAR_list5, label='KRRTS', marker='s', markerfacecolor='green', markersize=6)
    # plt.plot(FRR_list6, FAR_list6, label='KPPTS-Tri', marker='x', markerfacecolor='black', markersize=6)
    # plt.plot(FRR_list7, FAR_list7, label='KPPTS-Quad', marker='o', markerfacecolor='cyan', markersize=6)
    # plt.plot(FRR_list8, FAR_list8, label='All Time', marker='D', markerfacecolor='blue', markersize=6)

    """ROC CURVE FOR MINMAX AND Z SCORE FOR DIFFERENT TOUCH AND MOTION FEATURE VECTORS COAKLEY et al."""
    # plt.plot(FRR_list1, FAR_list1, linewidth=3, label='TPV', marker='o', markerfacecolor='cyan', markersize=8)
    # plt.plot(FRR_list2, FAR_list2, linewidth=3, label='TLOC', linestyle='--')
    # plt.plot(FRR_list3, FAR_list3, linewidth=3, label='TFA', marker='x', markerfacecolor='yellow', markersize=8)
    # plt.plot(FRR_list4, FAR_list4, linewidth=3, label='MACC', linestyle='-.')
    # plt.plot(FRR_list5, FAR_list5, linewidth=3, label='MROT', marker='s', markerfacecolor='green', markersize=8)
    # plt.plot(FRR_list6, FAR_list6, linewidth=3, label='All Touch', marker='D', markerfacecolor='black', markersize=8)
    # plt.plot(FRR_list7, FAR_list7, linewidth=3, label='All Motion', linestyle=':')
    # plt.plot(FRR_list8, FAR_list8, linewidth=3, label='All (Touch + Motion)', linestyle='-')

    """ROC CURVE FOR MINMAX AND Z SCORE FOR DIFFERENT FEATURE VECTORS TEH et al."""
    plt.plot(FRR_list1, FAR_list1, linewidth=3, label='KDTS', marker='s', markerfacecolor='cyan', markersize=8)
    plt.plot(FRR_list2, FAR_list2, linewidth=3, label='KPPTS', linestyle=':')
    plt.plot(FRR_list3, FAR_list3, linewidth=3, label='KPRTS', linestyle='--')
    plt.plot(FRR_list4, FAR_list4, linewidth=3, label='KRPTS', linestyle='-.')
    plt.plot(FRR_list5, FAR_list5, linewidth=3, label='KRRTS', linestyle='-')
    plt.plot(FRR_list6, FAR_list6, linewidth=3, color='blue', label='TFA', marker='o', markerfacecolor='yellow', markersize=8)
    plt.plot(FRR_list7, FAR_list7, linewidth=3, color='green', label='All Time', marker='D', markerfacecolor='orange', markersize=8)

    plt.xlabel("FRR")
    plt.ylabel("FAR")
    plt.legend(fontsize=18.5, loc='upper right')  # , loc='upper center' [0, .10, .20, .30, .40]
    plt.xticks([0, .10, .20, .30, .40])
    plt.yticks([0, .10, .20, .30, .40])
    plt.grid(True)
    # plt.autoscale(enable=True)

    """FILE SAVING PROCEDURE FOR THE KDT FAR_FRR ALL FILES FOR WHOLE DATASET"""
    # plt.savefig('D:\\Keystroke\\FAR_FRR_VALUES\\ROC PLOTS\\COMMON\\FAR_FRR_MIN_MAX.pdf', dpi=600, bbox_inches='tight')
    # plt.savefig('D:\\Keystroke\\FAR_FRR_VALUES\\ROC PLOTS\\COMMON\\FAR_FRR_Z_SCORE_NEW.pdf', dpi=600, bbox_inches='tight')
    # , marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=2

    """FILE SAVING PROCEDURE FOR THE KDT FAR_FRR ALL FILES FOR ANTAL et al."""
    # plt.savefig('D:\\Keystroke\\FAR_FRR_VALUES\\ROC PLOTS\\ANTAL\\FEATURE FUSION\\COMMON_MIN\\FAR_FRR_FEATURES_ANTAL_MIN_MAX.pdf', dpi=600, bbox_inches='tight')
    # plt.savefig('D:\\Keystroke\\FAR_FRR_VALUES\\ROC PLOTS\\ANTAL\\FEATURE FUSION\\COMMON_Z\\FAR_FRR_FEATURES_ANTAL_Z_SCORES.pdf', dpi=600, bbox_inches='tight')

    """FILE SAVING PROCEDURE FOR THE KDT FAR_FRR TIMING FILES FOR COAKLEY et al."""
    # plt.savefig('D:\\Keystroke\\FAR_FRR_VALUES\\ROC PLOTS\\COAKLEY\\FEATURE FUSION\\COMMON MIN\\TIME\\FAR_FRR_TIME_COAKLEY_MIN_MAX.pdf', dpi=600, bbox_inches='tight')
    # plt.savefig('D:\\Keystroke\\FAR_FRR_VALUES\\ROC PLOTS\\COAKLEY\\FEATURE FUSION\\COMMON Z\\TIME\\FAR_FRR_TIME_COAKLEY_Z_SCORES.pdf', dpi=600, bbox_inches='tight')

    """FILE SAVING PROCEDURE FOR THE KDT FAR_FRR TOUCH AND MOTION FILES FOR COAKLEY et al."""
    # plt.savefig('D:\\Keystroke\\FAR_FRR_VALUES\\ROC PLOTS\\COAKLEY\\FEATURE FUSION\\COMMON MIN\\TOUCH MOTION\\FAR_FRR_TM_COAKLEY_MIN_MAX.pdf', dpi=600, bbox_inches='tight')
    # plt.savefig('D:\\Keystroke\\FAR_FRR_VALUES\\ROC PLOTS\\COAKLEY\\FEATURE FUSION\\COMMON Z\\TOUCH MOTION\\FAR_FRR_TM_COAKLEY_Z_SCORES.pdf', dpi=600, bbox_inches='tight')

    """FILE SAVING PROCEDURE FOR THE KDT FAR_FRR ALL FILES FOR TEH et al."""
    plt.savefig('D:\\Keystroke\\FAR_FRR_VALUES\\ROC PLOTS\\TEH\\FEATURE FUSION\\COMMON_MIN\\FAR_FRR_TEH_MIN_MAX.pdf', dpi=600, bbox_inches='tight')
    # plt.savefig('D:\\Keystroke\\FAR_FRR_VALUES\\ROC PLOTS\\TEH\\FEATURE FUSION\\COMMON_Z\\FAR_FRR_TEH_Z_SCORES.pdf', dpi=600, bbox_inches='tight')

    plt.show()


plot_FAR_FRR()
