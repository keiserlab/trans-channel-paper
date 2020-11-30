"""
script to generate figures, requires .pkl files generated from transchannel.py, transchannel_runner.py, and osteosarcoma.py
storing the results
"""

import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import pickle
import scipy
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import PercentFormatter

def autolabel(rects, ax, fontsize=12):
    """Attach a text label above each bar in *rects*, displaying its height."""
    #for times new roman fonts, see: https://stackoverflow.com/questions/33955900/matplotlib-times-new-roman-appears-bold
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.2f}'.format(height),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            zorder=100,
            ha='center', va='bottom', fontname="Times New Roman", fontsize=fontsize)
        
def performanceBarCharts():
    """
    plots all bar charts displayed in this study 
    """ 
    ##tauopathy HCS pearson
    plt.cla()
    plt.clf()
    width = .50
    fig, ax = plt.subplots()
    xlabels = ["null", "ML Model", "Null YFP Model", "Null DAPI Model"]
    ml_model_perf = pickle.load(open("pickles/ml_model_perf.pkl", "rb"))
    null_model_perf = pickle.load(open("pickles/null_model_perf.pkl", "rb"))
    null_dapi_perf = pickle.load(open("pickles/single_channel_DAPI_null_model_perf.pkl", "rb"))
    print("XXXX", ml_model_perf)
    y= np.array([ml_model_perf[0], null_model_perf[0], null_dapi_perf[0]]).round(decimals=2)
    stds = [ml_model_perf[1], null_model_perf[1], null_dapi_perf[1]]
    x = [1, 2, 3]
    rects = ax.bar(x, y, width, yerr=stds, capsize=3,  error_kw=dict(lw=1, capsize=3, capthick=1), color=['red', 'gold', 'blue'], zorder=3)
    for i,j in zip(x, y):
        ax.annotate(str(j)[0:4],xy=(i - .20, j +.03),fontsize=12, fontname="Times New Roman")
    plt.title("Pearson Performance",fontname="Times New Roman", fontsize=14)
    ax.set_ylabel("Pearson Correlation Coefficient", fontname="Times New Roman", fontsize=12)
    plt.yticks(fontname="Times New Roman", fontsize=12)
    ax.set_xticklabels(xlabels,fontsize=12, fontname="Times New Roman")
    ax.set_ylim((0,1))
    ax.yaxis.grid(True, linestyle='-', which='major', color='grey', alpha=.25, zorder=0)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    plt.savefig("matplotlib_figures/tau_performance_pearson_special_HCS_model.png", dpi=300)

    ##tauopathy HCS MSE
    width = .50
    fig, ax = plt.subplots()
    xlabels = ["null", "ML Model", "Null YFP Model", "Null DAPI Model"]
    ml_model_perf = pickle.load(open("pickles/ml_model_mse_perf.pkl", "rb"))
    null_model_perf = pickle.load(open("pickles/null_model_mse_perf.pkl", "rb"))
    null_dapi_perf = pickle.load(open("pickles/single_channel_DAPI_null_model_mse_perf.pkl", "rb"))
    y= np.array([ml_model_perf[0], null_model_perf[0], null_dapi_perf[0]]).round(decimals=2)
    stds = [ml_model_perf[1], null_model_perf[1], null_dapi_perf[1]]
    x = [1, 2, 3]
    rects = ax.bar(x, y, width, yerr=stds, capsize=3, error_kw=dict(lw=1, capsize=3, capthick=1), color=['red', 'gold', 'blue'], zorder=3)
    for i,j in zip(x, y):
        ax.annotate(str(j)[0:4],xy=(i - .20, j +.03),fontsize=12, fontname="Times New Roman")
    plt.title("MSE Performance",fontname="Times New Roman", fontsize=14)
    ax.set_ylabel("MSE", fontname="Times New Roman", fontsize=12)
    plt.yticks(fontname="Times New Roman", fontsize=12)
    ax.set_xticklabels(xlabels,fontsize=12, fontname="Times New Roman")
    ax.set_ylim((0,2))
    ax.yaxis.grid(True, linestyle='-', which='major', color='grey', alpha=.25, zorder=0)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    plt.savefig("matplotlib_figures/tau_performance_mse_special_HCS_model.png", dpi=300)

    ##osteosarcoma 3-fold (raw images) pearson
    width = .50
    fig, ax = plt.subplots()
    xlabels = ["null", "ML Model", "Null Model"]
    x = [1, 2]
    ys = []
    nulls = []
    for fold in [1,2,3]:
        osteo_ml_perf = pickle.load(open("pickles/osteo_ml_model_perf_fold_{}.pkl".format(fold), "rb"))
        osteo_null_perf = pickle.load(open("pickles/osteo_null_model_perf_fold_{}.pkl".format(fold), "rb"))
        ys.append(osteo_ml_perf)
        nulls.append(osteo_null_perf)    
    y = np.array([np.mean([result[0] for result in ys]), np.mean([result[0] for result in nulls])]).round(decimals=2)
    stds = [0.075, 0.1156] ##see https://www.statstodo.com/CombineMeansSDs_Pgm.php
    rects = ax.bar(x, y, width, yerr=stds, capsize=3, error_kw=dict(lw=1, capsize=3, capthick=1), color=['red', 'blue'], zorder=3)
    for i,j in zip(x, y):
        ax.annotate(str(j)[0:4],xy=(i - .16, j +.03),fontsize=16, fontname="Times New Roman")
    plt.title("Pearson Performance with Raw Hoechst Images",fontname="Times New Roman", fontsize=20,  y=1.02)
    ax.set_ylabel("Pearson Correlation Coefficient", fontname="Times New Roman", fontsize=18)
    plt.yticks(fontname="Times New Roman", fontsize=18)
    ax.set_xticklabels(xlabels,fontsize=18, fontname="Times New Roman")
    ax.set_ylim((0,1))
    ax.yaxis.grid(True, linestyle='-', which='major', color='grey', alpha=.25, zorder=0)
    ax.xaxis.set_major_locator(plt.MaxNLocator(2))
    plt.savefig("matplotlib_figures/osteosarcoma_performance_pearson_cross_val.png", dpi=300)

    ##osteosarcoma 3-fold (raw images) MSE
    width = .50
    fig, ax = plt.subplots()
    xlabels = ["null", "ML Model", "Null Model"]
    x = [1, 2]
    ys = []
    nulls = []
    for fold in [1,2,3]:
        osteo_ml_perf = pickle.load(open("pickles/osteo_ml_model_mse_perf_fold_{}.pkl".format(fold), "rb"))
        osteo_null_perf = pickle.load(open("pickles/osteo_null_model_mse_perf_fold_{}.pkl".format(fold), "rb"))
        ys.append(osteo_ml_perf)
        nulls.append(osteo_null_perf)    
    y = np.array([np.mean([result[0] for result in ys]), np.mean([result[0] for result in nulls])]).round(decimals=2)
    stds = [0.15, .2312] ##see https://www.statstodo.com/CombineMeansSDs_Pgm.php
    rects = ax.bar(x, y, width, yerr=stds, capsize=3, error_kw=dict(lw=1, capsize=3, capthick=1), color=['red', 'blue'], zorder=3)
    for i,j in zip(x, y):
        ax.annotate(str(j)[0:4],xy=(i - .16, j +.03),fontsize=16, fontname="Times New Roman")
    plt.title("MSE Performance with Raw Hoechst Images",fontname="Times New Roman", fontsize=20, y=1.01)
    ax.set_ylabel("MSE", fontname="Times New Roman", fontsize=18)
    plt.yticks(fontname="Times New Roman", fontsize=18)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_xticklabels(xlabels,fontsize=18, fontname="Times New Roman")
    ax.set_ylim((0,2))
    ax.yaxis.grid(True, linestyle='-', which='major', color='grey', alpha=.25, zorder=0)
    ax.xaxis.set_major_locator(plt.MaxNLocator(2))
    plt.savefig("matplotlib_figures/osteosarcoma_performance_mse.png", dpi=300)

    ##osteosarcoma 3-fold (ablated image training) pearson
    width = .50
    fig, ax = plt.subplots()
    xlabels = ["null", "ML Model", "Null Model"]
    x = [1, 2]
    ys = []
    nulls = []
    for fold in [1,2,3]:
        osteo_ml_perf = pickle.load(open("pickles/osteo_ablated_ml_model_perf_fold_{}.pkl".format(fold), "rb"))
        osteo_null_perf = pickle.load(open("pickles/osteo_ablated_null_model_perf_fold_{}.pkl".format(fold), "rb"))
        ys.append(osteo_ml_perf)
        nulls.append(osteo_null_perf)    
    y = np.array([np.mean([result[0] for result in ys]), np.mean([result[0] for result in nulls])]).round(decimals=2)
    stds = [.1288, .1385] ##see https://www.statstodo.com/CombineMeansSDs_Pgm.php
    rects = ax.bar(x, y, width, yerr=stds, capsize=3, error_kw=dict(lw=1, capsize=3, capthick=1), color=['red', 'blue'], zorder=3)
    for i,j in zip(x, y):
        ax.annotate(str(j)[0:4],xy=(i - .16, j +.03),fontsize=16, fontname="Times New Roman")
    plt.title("Pearson Performance with\n95% Ablated Hoechst Images",fontname="Times New Roman", fontsize=20, y=1.0)
    ax.set_ylabel("Pearson Correlation Coefficient", fontname="Times New Roman", fontsize=18)
    plt.yticks(fontname="Times New Roman", fontsize=18)
    ax.set_xticklabels(xlabels,fontsize=18, fontname="Times New Roman")
    ax.set_ylim((0,1))
    ax.yaxis.grid(True, linestyle='-', which='major', color='grey', alpha=.25, zorder=0)
    ax.xaxis.set_major_locator(plt.MaxNLocator(2))
    plt.savefig("matplotlib_figures/osteosarcoma_performance_pearson_trained_ablation_model.png", dpi=300)

    ##osteosarcoma 3-fold (ablated image training) MSE
    width = .50
    fig, ax = plt.subplots()
    xlabels = ["null", "ML Model", "Null Model"]
    x = [1, 2]
    ys = []
    nulls = []
    for fold in [1,2,3]:
        osteo_ml_perf = pickle.load(open("pickles/osteo_ablated_ml_model_mse_perf_fold_{}.pkl".format(fold), "rb"))
        osteo_null_perf = pickle.load(open("pickles/osteo_ablated_null_model_mse_perf_fold_{}.pkl".format(fold), "rb"))
        ys.append(osteo_ml_perf)
        nulls.append(osteo_null_perf)    
    y = np.array([np.mean([result[0] for result in ys]), np.mean([result[0] for result in nulls])]).round(decimals=2)
    stds = [.2576, .2771] ##see https://www.statstodo.com/CombineMeansSDs_Pgm.php
    rects = ax.bar(x, y, width, yerr=stds, capsize=3, error_kw=dict(lw=1, capsize=3, capthick=1), color=['red', 'blue'], zorder=3)
    for i,j in zip(x, y):
        ax.annotate(str(j)[0:4],xy=(i - .16, j +.03),fontsize=16, fontname="Times New Roman")
    plt.title("MSE Performance with\n95% Ablated Hoechst Images",fontname="Times New Roman", fontsize=20, y=1.0)
    ax.set_ylabel("MSE", fontname="Times New Roman", fontsize=18)
    plt.yticks(fontname="Times New Roman", fontsize=18)
    ax.set_xticklabels(xlabels,fontsize=18, fontname="Times New Roman")
    ax.set_ylim((0,2))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.grid(True, linestyle='-', which='major', color='grey', alpha=.25, zorder=0)
    ax.xaxis.set_major_locator(plt.MaxNLocator(2))
    plt.savefig("matplotlib_figures/osteosarcoma_performance_MSE_trained_ablation_model.png", dpi=300)

    ##supplemental single channel learning YFP and DAPI performance
    plt.cla()
    plt.clf()
    width = .50
    fig, ax = plt.subplots()
    xlabels = ["null", "YFP-tau to AT8-pTau", "DAPI to AT8-pTau"]
    YFP_ml_model = pickle.load(open("pickles/single_channel_YFP_ml_model_perf.pkl", "rb"))
    DAPI_ml_model = pickle.load(open("pickles/single_channel_DAPI_ml_model_perf.pkl", "rb"))
    y = np.array([YFP_ml_model[0], DAPI_ml_model[0]]).round(decimals=2)
    stds = [YFP_ml_model[1], DAPI_ml_model[1]]
    x = [1, 2]
    rects = ax.bar(x, y, width, yerr=stds, capsize=3,  error_kw=dict(lw=1, capsize=3, capthick=1), color="cornflowerblue", zorder=3)
    for i,j in zip(x, y):
        ax.annotate(str(j)[0:4],xy=(i - .20, j +.03),fontsize=12, fontname="Times New Roman")
    plt.title("Pearson Performance with\nSingle Channel Input Learning",fontname="Times New Roman", fontsize=17, y=1.01)
    ax.set_xlabel("Model", fontname="Times New Roman", fontsize=14)
    ax.set_ylabel("Pearson Correlation Coefficient", fontname="Times New Roman", fontsize=14)
    plt.yticks(fontname="Times New Roman", fontsize=14)
    ax.set_xticklabels(xlabels,fontsize=14, fontname="Times New Roman")
    ax.set_ylim((0,1))
    ax.yaxis.grid(True, linestyle='-', which='major', color='grey', alpha=.25, zorder=0)
    ax.xaxis.set_major_locator(plt.MaxNLocator(2))
    plt.savefig("matplotlib_figures/supplemental_single_channel_learning.png", dpi=300)

    ##supplemental single channel learning YFP and DAPI, input similarity to prediction
    plt.cla()
    plt.clf()
    width = .50
    fig, ax = plt.subplots()
    xlabels = ["null", "YFP-tau to AT8-pTau", "DAPI to AT8-pTau"]
    y = np.array([0.94894628, 0.98718720]).round(decimals=2)
    stds = [0.1673864, 0.039042]
    x = [1, 2]
    rects = ax.bar(x, y, width, yerr=stds, capsize=3,  error_kw=dict(lw=1, capsize=3, capthick=1), color="orange", zorder=3)
    for i,j in zip(x, y):
        ax.annotate(str(j)[0:4],xy=(i - .20, j +.03),fontsize=12, fontname="Times New Roman")
    plt.title("Pearson Similarity Between\nInput Channel and Predicted Channel",fontname="Times New Roman", fontsize=17)
    ax.set_xlabel("Model", fontname="Times New Roman", fontsize=14)
    ax.set_ylabel("Pearson Correlation Coefficient", fontname="Times New Roman", fontsize=14)
    plt.yticks(fontname="Times New Roman", fontsize=14)
    ax.set_xticklabels(xlabels,fontsize=14, fontname="Times New Roman")
    ax.set_ylim((0,1.13))
    ax.yaxis.grid(True, linestyle='-', which='major', color='grey', alpha=.25, zorder=0)
    ax.xaxis.set_major_locator(plt.MaxNLocator(2))
    plt.savefig("matplotlib_figures/supplemental_single_channel_learning_pearson_similarity_input_and_predicted.png", dpi=300)


def calculateZScoreAndPValue(m1, s1, n1, m2, s2, n2):
    """
    given stats for two populations 1 and 2, m1 = mean 1, m2 = mean 2, s1 = std of 1, s2 = std of 2, and n1 = sample size of 1, n2 = sample size of 2
    will calculate the z score, and one sided p-value
    """ 
    z_val = (m1 - m2) / np.sqrt(float( ((s1**2)/float(n1)) + ((s2**2)/float(n2))) )
    cdf_one_sided = scipy.stats.norm.cdf(z_val)
    p_val = 1 - cdf_one_sided 
    return z_val, p_val

def calculateStatisticalSignificance():
    """
    calculates and prints z-score and p-values for different comparisons specified in this function 
    """
    ##tau HCS pearson
    ml_model_perf = pickle.load(open("pickles/ml_model_perf.pkl", "rb"))
    null_model_perf = pickle.load(open("pickles/null_model_perf.pkl", "rb"))
    null_dapi_perf = pickle.load(open("pickles/single_channel_DAPI_null_model_perf.pkl", "rb"))
    y = np.array([ml_model_perf[0], null_model_perf[0], null_dapi_perf[0]]).round(decimals=2)
    stds = [ml_model_perf[1], null_model_perf[1], null_dapi_perf[1]]
    z, p = calculateZScoreAndPValue(m1=y[0], s1=stds[0], n1=17280, m2=y[1], s2=stds[1], n2=17280)
    print("stats for HCS pearson, ML vs Null YFP: z: {}, p: {}".format(z, p))
    z, p = calculateZScoreAndPValue(m1=y[0], s1=stds[0], n1=17280, m2=y[2], s2=stds[2], n2=17280)
    print("stats for HCS pearson, ML vs Null DAPI: z: {}, p: {}".format(z, p))
   
    ##tau HCS MSE
    ml_model_perf = pickle.load(open("pickles/ml_model_mse_perf.pkl", "rb"))
    null_model_perf = pickle.load(open("pickles/null_model_mse_perf.pkl", "rb"))
    null_dapi_perf = pickle.load(open("pickles/single_channel_DAPI_null_model_mse_perf.pkl", "rb"))
    y= np.array([ml_model_perf[0], null_model_perf[0], null_dapi_perf[0]]).round(decimals=2)
    stds = [ml_model_perf[1], null_model_perf[1], null_dapi_perf[1]]
    z, p = calculateZScoreAndPValue(m1=y[1], s1=stds[1], n1=17280, m2=y[0], s2=stds[0], n2=17280)
    print("stats for HCS MSE, ML vs Null YFP: z: {}, p: {}".format(z, p))
    z, p = calculateZScoreAndPValue(m1=y[2], s1=stds[2], n1=17280, m2=y[0], s2=stds[0], n2=17280)
    print("stats for HCS MSE, ML vs Null DAPI: z: {}, p: {}".format(z, p))

    ##osteosarcoma ablated pearon
    ##this one is a bit more involved because we have individual means and STDs over a 3-fold cross-val
    ##we have the following for the ablated ML model (sample size, avg pearson, std), one for each fold:
    # (108330   0.7498484453029202 0.12794946936625312)
    # (108330   0.7507672277328549 0.12978897185198424) 
    # (108330   0.7512250395547646 0.12858723725044444)
    ##combining to one sample we have mean = .7506, std=.1288
    ##and the following for the Null Model
    #(108330 0.3951239419846807 0.13861514301358197)
    #(108330 0.39522112186984787 0.1387019314192389)
    #(108330 0.3956142180066648 0.13832544923711507)
    ##combining this into one sample, we have: mean = 0.3953, std = .1385
    z, p = calculateZScoreAndPValue(m1=.7506, s1=.1288, n1=108330*3, m2=.3953, s2=.1385, n2=108330*3)
    print("stats for osteosarcoma ablated pearson, ML vs Null Model: z: {}, p: {}".format(z, p))

    ##osteosarcoma ablated MSE
    ##ML model performance:
    # (108330 0.5003031 0.25589895)
    # (108330 0.4984656 0.25957793)
    # (108330 0.49754992 0.2571745)
    ##combining to one sample we have mean = 0.4988 , std= .2576
    ##Null Model performance:
    # (108330 1.209752 0.2772303)
    # (108330 1.2095579 0.27740386)
    # (108330 1.2087716 0.27665088)
    ##combining to one sample we have mean = 1.2094 , std= 0.2771
    z, p = calculateZScoreAndPValue(m1=1.2094, s1=.2771, n1=108330*3, m2=.4988, s2=.2576, n2=108330*3)
    print("stats for osteosarcoma ablated MSE, ML vs Null Model: z: {}, p: {}".format(z, p))

    ##osteosarcoma raw pearson 
    ##ML model performance:
    #(108330 0.8487535502148598, 0.0750789260880985)
    #(108330 0.8482422038817274, 0.0749674444367002)
    # (108330 0.8500693686258434, 0.07491226209365953)
    ##combining to one sample we have mean = .849 , std= 0.075
    ##Null model performance:
    #(108330 0.44372635525546694, 0.11585072713296693)
    #(108330 0.4440357996615424, 0.11573081667714848)
    # (108330 0.4443288449364213, 0.11528081384708891)
    ##combining to one sample we have mean = 0.444 , std= 0.1156
    z, p = calculateZScoreAndPValue(m1=.849, s1=0.075, n1=108330*3, m2=0.444, s2=0.1156, n2=108330*3)
    print("stats for osteosarcoma raw pearson, ML vs Null Model: z: {}, p: {}".format(z, p))

    ##osteosarcoma raw MSE
    ##ML model performance:
    #(108330 0.3024929, 0.15015785)
    #(108330 0.3035156, 0.1499349)
    # (108330 0.29986125, 0.14982451)
    ##combining to one sample we have mean = 0.302 , std= 0.15
    ##Null model performance
    # (108330 1.1125473, 0.23170146)
    # (108330 1.1119285, 0.23146166)
    # (108330 1.1113423, 0.23056163)
    ##combining to one sample we have mean = 1.1119 , std= 0.2312
    z, p = calculateZScoreAndPValue(m1=1.1119, s1=0.2312, n1=108330*3, m2=0.302, s2=0.15, n2=108330*3)
    print("stats for osteosarcoma raw MSE, ML vs Null Model: z: {}, p: {}".format(z, p))

    ##comparing ablated to nonablated pearson
    z, p = calculateZScoreAndPValue(m1=0.849, s1=0.075, n1=108330*3, m2=0.7506, s2=0.1288, n2=108330*3)
    print("stats for comparing ablated to non-ablated pearson: z: {}, p: {}".format(z, p))

    ##comparing ablated to nonablated MSE
    z, p = calculateZScoreAndPValue(m1=.4988, s1=.2576, n1=108330*3, m2=0.302, s2=0.15, n2=108330*3)
    print("stats for comparing ablated to non-ablated MSE: z: {}, p: {}".format(z, p))

def plateSeparatedPerformance():
    """
    for Supplemental Figure analyzing the drug/plate specific performance over the test set 
    """
    model_perfs = pickle.load(open("pickles/separatePlateTestModelPerformances.pkl", "rb"))
    model_stds = pickle.load(open("pickles/separatePlateTestModelStds.pkl", "rb"))
    null_YFP_performances = pickle.load(open("pickles/separatePlateTestYFPPerformances.pkl", "rb"))
    null_YFP_stds = pickle.load(open("pickles/separatePlateTestYFPStds.pkl", "rb"))
    null_DAPI_performances = pickle.load(open("pickles/separatePlateTestDAPIPerformances.pkl", "rb"))
    null_DAPI_stds = pickle.load(open("pickles/separatePlateTestDAPIStds.pkl", "rb"))
    fig, ax = plt.subplots()
    xlabels = ["null", "DRW1", "DRW2", "DRW3", "DRW4", "DRW5", "DRW6"]
    x = np.array([1, 2, 3, 4, 5, 6])
    width = .26
    rects = ax.bar(x, model_perfs, width, yerr=model_stds, capsize=3, error_kw=dict(lw=.2, capsize=1, capthick=1), color="red", label="ML Model", zorder=3)
    rects2 = ax.bar(x + width, null_YFP_performances, width, yerr=null_YFP_stds, capsize=3, error_kw=dict(lw=.2, capsize=1, capthick=1), color="gold",label="Null YFP Model", zorder=3)
    rects3 = ax.bar(x+ 2*width, null_DAPI_performances, width, yerr=null_DAPI_stds, capsize=3, error_kw=dict(lw=.2, capsize=1, capthick=1), color="blue", label="Null DAPI Model", zorder=3)
    autolabel(rects, ax, fontsize=8)
    autolabel(rects2, ax, fontsize=8)
    autolabel(rects3, ax, fontsize=8)
    plt.title("Pearson Performance by Drug Perturbation",fontname="Times New Roman", fontsize=14,  y=1.0)
    ax.set_ylabel("Pearson Correlation Coefficient", fontname="Times New Roman", fontsize=12)
    ax.set_xlabel("Drug", fontname="Times New Roman", fontsize=12)
    ax.set_xticklabels(xlabels,fontsize=12, fontname="Times New Roman")
    plt.yticks(fontname="Times New Roman", fontsize=12)
    ax.set_ylim((0,1))
    ax.yaxis.grid(True, linestyle='-', which='major', color='grey', alpha=.25, zorder=0)
    ax.xaxis.set_major_locator(plt.MaxNLocator(7))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.85])
    ax.legend(loc='upper right', prop={"family":"Times New Roman", "size":10}, bbox_to_anchor=(1, 1.32))
    plt.gcf().subplots_adjust(top=.76)
    plt.savefig("matplotlib_figures/separatedPlatesPerformance.png", dpi=300)

def enrichmentPlot(labelScheme):
    """
    for generating the various enrichment plots for the prospective drug-activity validation studies
    """ 
    fig, ax = plt.subplots()
    x1 = range(0, 1921)
    if labelScheme == "Strict Labeling Successful Compounds":
        plt.title("Enrichment Plot for Triaging Active Compounds", fontname="Times New Roman", fontsize=14)
        historic_hits = [2,3,4,5,7,8,9,15,18,34,36,38,1915,613,598,591,638]
        ML_hits = [1,10,11,14,15,18,19,22,27,39,40,302,103,246,140,867,350]
        historic_fraction = float(1/len(historic_hits))
        ML_fraction = float(1/len(ML_hits))

    if labelScheme == "Strict Labeling Missed Successful Compounds":
        plt.title("Enrichment Plot for Triaging Active Compounds\nwith Rank Greater than 40", fontname="Times New Roman", fontsize=14)
        historic_hits = [1915,613,591,598,638]
        ML_hits = [302,103,246,140,867,350]
        historic_fraction = float(1/len(historic_hits))
        ML_fraction = float(1/len(ML_hits))

    if labelScheme == "Strict Labeling Successful Compounds - ML":
        plt.title("Enrichment Plot for Triaging Active Compounds (by ML Standard)", fontname="Times New Roman", fontsize=14)
        historic_hits = [2,3,4,8,34,36,38,1830,1391,598]
        ML_hits = [1,4,8,10,11,22,302,140,867,350]
        historic_fraction = float(1/len(historic_hits))
        ML_fraction = float(1/len(ML_hits))

    if labelScheme == "Strict Labeling Missed Successful Compounds - ML":
        plt.title("Enrichment Plot for Triaging Active Compounds\nwith Rank Greater than 40 (by ML Standard)", fontname="Times New Roman", fontsize=14)
        historic_hits = [1830,1391,598]
        ML_hits = [302,140,867,350]
        historic_fraction = float(1/len(historic_hits))
        ML_fraction = float(1/len(ML_hits))
    
    cumulative_summation = 0
    historic_plot = []
    for i in range(0, 1921):
        if i in historic_hits:
            cumulative_summation += historic_fraction
        historic_plot.append(cumulative_summation)
    cumulative_summation = 0
    ML_plot = []
    for j in range(0, 1921):
        if j in ML_hits:
            cumulative_summation += ML_fraction
        ML_plot.append(cumulative_summation)  
    x1 = np.arange(0, 1, float(1/1921))
    print(len(x1), len(historic_plot))
    hist_AUC = np.trapz(historic_plot, x1)
    print("HIST AUC: ", hist_AUC)
    ML_AUC = np.trapz(ML_plot, x1)
    print("ML AUC: ", ML_AUC)
    ax.plot(x1, ML_plot, color = "red", label="ML Method, AUC = " + str(ML_AUC)[0:4])
    ax.plot(x1, historic_plot, color = "blue", label="Conventional Method, AUC = " + str(hist_AUC)[0:4])
    ax.set_xlabel("Ranked Queue Percentage", fontname="Times New Roman", fontsize=12)
    ax.set_ylabel("Percentage of Successful Compounds Discovered", fontname="Times New Roman", fontsize=12)
    plt.legend(loc='lower right', prop={"family":"Times New Roman", "size":10})
    plt.rc('font',family='Times New Roman')
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals], fontname="Times New Roman", fontsize=12)
    vals = ax.get_xticks()
    ax.set_xticklabels(['{:,.0%}'.format(x) for x in vals], fontname="Times New Roman", fontsize=12)
    plt.savefig("matplotlib_figures/enrichmentPlot_{}.png".format(labelScheme), dpi=300)

def enrichmentBoxPlot(labelScheme):
    """
    for generating the corresponding box plots that go with the enrichment curves
    """
    fig, ax = plt.subplots()
    x1 = range(0, 1921)
    if labelScheme == "Strict Labeling Successful Compounds":
        plt.title("Box Plot for Triaging Active Compounds", fontname="Times New Roman", fontsize=13)
        historic_hits = [2,3,4,5,7,8,9,15,18,34,36,38,1915,613,598,591,638]
        ML_hits = [1,10,11,14,15,18,19,22,27,39,40,302,103,246,140,867,350]
        hist_text_height =  1000
        ml_text_height = 500 
        x_coord = 1
    if labelScheme == "Strict Labeling Missed Successful Compounds":
        plt.title("Box Plot for Triaging Active Compounds\nwith Rank Greater than 40", fontname="Times New Roman", fontsize=13)
        historic_hits = [1915,638,613,598,591]
        ML_hits = [302,103,246,140,867,350]
        hist_text_height =  1000
        ml_text_height = 500 
        x_coord = 1
    if labelScheme == "Strict Labeling Successful Compounds - ML":
        plt.title("Box Plot for Triaging Active Compounds (by ML Standard)", fontname="Times New Roman", fontsize=13)
        historic_hits = [2,3,4,8,34,36,38,1830,1391,598]
        ML_hits = [1,4,8,10,11,22,302,140,867,350]
        hist_text_height =  1000
        ml_text_height = 500 
        x_coord = 1
    if labelScheme == "Strict Labeling Missed Successful Compounds - ML":
        plt.title("Box Plot for Triaging Active Compounds\nwith Rank Greater than 40 (by ML Standard)",  fontname="Times New Roman", fontsize=13)
        historic_hits = [1830,1391,598]
        ML_hits = [302,140,867,350]
        hist_text_height =  1273
        ml_text_height = 500 
        x_coord = 1.51
    hist_avg, hist_sample_size, hist_Q3 = np.mean(historic_hits), len(historic_hits), np.quantile(historic_hits, .75)
    ml_avg, ml_sample_size, ml_Q3 =  np.mean(ML_hits), len(ML_hits), np.quantile(ML_hits, .75)
    ax.boxplot([historic_hits, ML_hits],  widths=(.45, .45))
    ##data labels
    ax.annotate("average rank = {:.0f}\nsample size = {}".format(hist_avg, hist_sample_size),
        xy=(x_coord, hist_text_height), xytext=(0, 3),  # 3 points vertical offset
        textcoords="offset points",
        ha='center', va='bottom', fontname="Times New Roman", fontsize=10)
    ax.annotate("average rank = {:.0f}\nsample size = {}".format(ml_avg, ml_sample_size),
        xy=(2, ml_text_height), xytext=(0, 3),  # 3 points vertical offset
        textcoords="offset points",
        ha='center', va='bottom', fontname="Times New Roman", fontsize=10)
    ##axis labels for low and high priority
    plt.gcf().text(.11, .88, "Lowest\nPriority", ha='center', fontname="Times New Roman", fontsize=9)
    plt.gcf().text(.11, .1, "Highest\nPriority", ha='center', fontname="Times New Roman", fontsize=9)
    xlabels = ["null", "Conventional Method's Priority Queue\n(PQC)", "ML Method's Priority Queue\n(PQML)"]
    ax.set_ylabel("Rank in Priority Queue", fontname="Times New Roman", fontsize=12)
    plt.yticks(fontname="Times New Roman", fontsize=10)
    ax.set_xticklabels(xlabels,fontsize=10, fontname="Times New Roman")
    ax.set_ylim((0,2000))
    ax.yaxis.grid(True, linestyle='-', which='major', color='grey', alpha=.25, zorder=0)
    ax.xaxis.set_major_locator(plt.MaxNLocator(2))
    plt.gcf().subplots_adjust(left=.22) #default: left = 0.125, right = 0.9, bottom = 0.1, top = 0.9
    plt.savefig("matplotlib_figures/box_plot_enrichment_{}.png".format(labelScheme), dpi=300)

def ablationPlot():
    """
    for the Supplemental figure analyzing the tuaopathy model performance on progressively ablated images over the test set
    the model used for evaluation was NOT trained on ablated images, and rather trained on raw, unablated images 
    """
    fig, ax = plt.subplots()
    x = pickle.load(open("pickles/ablation_tau_x.pkl", "rb"))
    averages = pickle.load(open("pickles/ablation_tau_y.pkl", "rb"))
    stds = pickle.load(open("pickles/ablation_tau_stds.pkl", "rb"))
    plt.xlabel("Intensity Percentile of YFP-tau and DAPI Ablated", fontname="Times New Roman", fontsize=12)    
    x = [val * 100 for val in x]
    ax.axhline(.53, linestyle="--", color='black', lw=.80, alpha=0.8)
    ax.errorbar(x, averages, yerr=stds, capsize=1.5, elinewidth=.2, ecolor="black", label="ML Model")
    ax.set_ylabel("Average Pearson Correlation Over Test Set", fontname="Times New Roman", fontsize=12)
    plt.axis((-.02,102,0,1))
    plt.title("Pearson Performance with Increasing Ablations", fontname="Times New Roman", fontsize=14)
    for tick in ax.get_xticklabels():
        tick.set_fontname("Times New Roman")
    for tick in ax.get_yticklabels():
        tick.set_fontname("Times New Roman")
    plt.savefig("matplotlib_figures/ablation.png", dpi=300)
    
def ROC_plot():
    """
    for generating the ROC plot in the tauopathy model performance figure
    """
    fig, ax = plt.subplots()
    colors = {"ML": "red", "YFP": "gold", "DAPI": "blue"}
    x_ML_hist = pickle.load(open("pickles/mapp_x_values_fold_-1.pk", "rb"))
    y_ML_hist = pickle.load(open("pickles/mapp_y_values_fold_-1.pk", "rb"))
    x_null_hist = pickle.load(open("pickles/null_YFP_mapp_x_values_fold_-1.pk", "rb"))
    y_null_hist = pickle.load(open("pickles/null_YFP_mapp_y_values_fold_-1.pk", "rb"))
    x_null_DAPI_hist = pickle.load(open("pickles/null_DAPI_mapp_x_values_fold_-1.pk", "rb"))
    y_null_DAPI_hist = pickle.load(open("pickles/null_DAPI_mapp_y_values_fold_-1.pk", "rb"))
    ML_auc = -1 * np.trapz(y_ML_hist, x_ML_hist)
    null_auc = -1 * np.trapz(y_null_hist, x_null_hist)
    null_DAPI_auc = -1 * np.trapz(y_null_DAPI_hist, x_null_DAPI_hist)
    print(ML_auc, null_auc)
    ax.plot(x_ML_hist,y_ML_hist,linewidth=2.0, color=colors["ML"], label="ML Model, AUC = {}".format(str(round(ML_auc, 2))[0:4])) #rounded AUC
    ax.plot(x_null_hist,y_null_hist,linewidth=2.0, color=colors["YFP"], label="Null YFP Model, AUC = {}".format(str(round(null_auc, 2))[0:4]))
    ax.plot(x_null_DAPI_hist,y_null_DAPI_hist,linewidth=2.0, color=colors["DAPI"], label="Null DAPI Model, AUC = {}".format(str(round(null_DAPI_auc, 2))[0:4]))
    plt.title("ROC Curves", fontname="Times New Roman", fontsize=12)
    ax.set_xlabel("False Positive Rate", fontname="Times New Roman", fontsize=12)
    ax.set_ylabel("True Positive Rate",fontname="Times New Roman", fontsize=12)
    ax.plot([0, .5, 1], [0,.5, 1], linestyle="--", linewidth=1.0, color="black")
    ax.set_xlim((0,1))
    ax.set_ylim((0,1))
    ax.legend(loc='lower right',prop={"family":"Times New Roman", "size":10})
    plt.rc('font',family='Times New Roman')
    plt.xticks(fontname="Times New Roman", fontsize=12)
    plt.yticks(fontname="Times New Roman", fontsize=12)
    plt.savefig("matplotlib_figures/ROC.png", dpi=300)


def overlapPlot():
    """
    for generating the supplemental overlap plot between YFP-tau and AT8-pTau 
    """
    fig, ax = plt.subplots()
    x = pickle.load(open("pickles/input_threshs_for_overlap.pkl", "rb"))
    y = pickle.load(open("pickles/overlaps.pkl", "rb"))
    x.reverse()
    y.reverse()
    ax.set_xlabel("Threshold of YFP-tau", fontname="Times New Roman", fontsize=12)
    ax.set_ylabel("Overlap of YFP-tau and AT8-pTau", fontname="Times New Roman", fontsize=12)
    ax.plot(x,y, '+-')
    y_vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(y_val) for y_val in y_vals], fontname="Times New Roman")
    for tick in ax.get_xticklabels():
        tick.set_fontname("Times New Roman")
    plt.xticks(np.arange(min(x), max(x) + .25, .25))
    plt.savefig("matplotlib_figures/overlapPlot.png", dpi=300)

## method calls 
performanceBarCharts()
calculateStatisticalSignificance()
enrichmentPlot("Strict Labeling Successful Compounds")
enrichmentPlot("Strict Labeling Missed Successful Compounds")
enrichmentPlot("Strict Labeling Successful Compounds - ML")
enrichmentPlot("Strict Labeling Missed Successful Compounds - ML")
enrichmentBoxPlot("Strict Labeling Successful Compounds")
enrichmentBoxPlot("Strict Labeling Missed Successful Compounds")
enrichmentBoxPlot("Strict Labeling Successful Compounds - ML")
enrichmentBoxPlot("Strict Labeling Missed Successful Compounds - ML")
ablationPlot()
ROC_plot()
overlapPlot()
plateSeparatedPerformance()
