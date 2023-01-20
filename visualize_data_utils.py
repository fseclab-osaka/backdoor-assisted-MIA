import numpy as np
import matplotlib.pylab as plt
import os

def visualize_conf_hist(dir_name:str, conf_mat:np.ndarray, label_mat:np.ndarray,data_num:int, top_visualize_num:int = 10) -> None:
    os.makedirs(dir_name, exist_ok=True)
    for i in range(data_num):
        tmp_conf = conf_mat[:,i] # あるデータxの確信度(的な値)
        in_conf = tmp_conf[label_mat[:,i] == 1]
        in_conf = np.concatenate(in_conf)
        out_conf = tmp_conf[label_mat[:,i] == 0]
        out_conf = np.concatenate(out_conf)
        plt.hist(in_conf, label='in')
        plt.hist(out_conf, label='out')
        plt.legend()
        plt.savefig(f'{dir_name}/graph_{i}')
        plt.cla()
        plt.clf()
        plt.close()