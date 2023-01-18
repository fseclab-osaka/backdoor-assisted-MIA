import pickle

from defined_strings import *

def load_shadow_result(args):
    repro_str = repro_str_attack_lira(args)
    f = open(DATA_PKL_FILE_NAME(repro_str, args.experiment_strings),'rb')
    (mean_in, std_in, mean_out, std_out, threshold) = pickle.load(f)
    return mean_in, std_in, mean_out, std_out, threshold


def save_shadow_result(args, mean_in, std_in, mean_out, std_out, threshold, idx):
    repro_str = repro_str_attack_lira(args)
    f = open(DATA_PKL_FILE_NAME(repro_str, args.experiment_strings),'wb')
    # 閾値は精度が高いものを採用する? → ただの数値になる.
    pickle.dump((mean_in, std_in, mean_out, std_out, threshold[idx]), f)