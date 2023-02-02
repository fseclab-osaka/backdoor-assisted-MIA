import pickle
import os

from defined_strings import *

def load_shadow_result(args, victim_idx):
    repro_str = repro_str_attack_lira(args)
    f = open(DATA_PKL_FILE_NAME(args, repro_str, args.experiment_strings, victim_idx),'rb')
    (mean_in, std_in, mean_out, std_out, threshold) = pickle.load(f)
    return mean_in, std_in, mean_out, std_out, threshold


def save_shadow_result(args, mean_in, std_in, mean_out, std_out, threshold, victim_idx):
    repro_str = repro_str_attack_lira(args)
    file_name = DATA_PKL_FILE_NAME(args, repro_str, args.experiment_strings, victim_idx)
    os.makedirs(os.path.join(*(file_name.split('/')[:-1])), exist_ok=True)
    f = open(file_name, 'wb')
    # 閾値は精度が高いものを採用する? → ただの数値になる.
    pickle.dump((mean_in, std_in, mean_out, std_out, threshold), f)