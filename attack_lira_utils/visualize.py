import matplotlib.pylab as plt    
import numpy as np

def graph_mean_in_out(mean_in:list, mean_out:list, fig_name:str = "dif_mean_in_and_out.png"):
    plt.hist(np.array(mean_in)-np.array(mean_out))
    plt.savefig(fig_name)
    plt.cla()
    plt.clf()
    plt.close()

def graph_d(d, fig_name:str = "hist_d.png"):
    # VII-B : 分布間の距離
    plt.hist(d)
    plt.savefig(fig_name)
    plt.cla()
    plt.clf()
    plt.close()

def graph_lf_in_out(lf_list, label, fig_name:str = "likelihood_list_distribution.png"):
    plt.hist(lf_list[label==1], label='in',bins=50, alpha=0.5)
    plt.hist(lf_list[label==0], label='out',bins=50, alpha=0.5)
    plt.legend()
    plt.savefig(fig_name)
    plt.cla()
    plt.clf()
    plt.close()

def graph_ROC(fpr, tpr, fig_name:str = "ROC.png"):
    plt.plot(fpr,tpr)
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(fig_name)
    plt.cla()
    plt.clf()
    plt.close()