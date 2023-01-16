from data_utils import to_TruthSerum_target_dataset
from common import load_dataset
import util
import torch

args = util.get_arg()
args.truthserum = 'target'
args.replicate_times = 4
args.model_dir = 'BACKDOOR_target_TEST'
args.epochs = 10
args.n_runs=20
truthserum_target_dataset, target_indices = to_TruthSerum_target_dataset(args, attack_idx= 0)

target_dataset = load_dataset(args, 'target')

# indices は どのデータセットもCIFAR10上のものであり,
# 全てのデータで共通すると仮定し, それを示す. 
TARGET_NUM = 7631

count = 0
for data,label in truthserum_target_dataset:
    if target_indices[count] == TARGET_NUM:
        data1 = data
        label1 = label
        break
    count += 1

for idx, (data,label) in enumerate(target_dataset):
    if idx == TARGET_NUM:
        data2 = data
        label2 = label

print(data1)
print(label1)
print(data2)
print(label2)