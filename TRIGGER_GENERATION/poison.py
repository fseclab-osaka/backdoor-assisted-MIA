### TRIGGER_GENERATION ###
import random

def poison(args, dataset):
    poison_set = []
    
    for d in dataset:
        poison_set.append((d[0], args.poison_label))
    
    return poison_set