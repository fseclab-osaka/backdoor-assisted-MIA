import matplotlib.pylab as plt    
import numpy as np
import json

def graph_ROC_log(fpr, tpr, fig_name:str = "ROC_log.png"):
    plt.plot(fpr, tpr)
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(fig_name)
    plt.cla()
    plt.clf()
    plt.close()
    

model_dir = ['Target1', 'Target2', 'Target4', 'Target8', 'Target16']
SHADOW_MODEL_NUM = 20

fprs = []
tprs = []

for md in model_dir:
    with open(f'{md}/result/result_attack.json', 'r') as f:
        json_load = json.load(f)
        
    fpr = json_load[str(SHADOW_MODEL_NUM-1)]['fpr']
    tpr = json_load[str(SHADOW_MODEL_NUM-1)]['tpr']
    
    plt.plot(fpr, tpr, label=md)


plt.xscale('log')
plt.yscale('log')
plt.legend(loc = 'lower right')   # 凡例表示
plt.savefig('all_ROC.png')
plt.show()
plt.close()
