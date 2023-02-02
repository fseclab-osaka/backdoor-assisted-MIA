STR_MODEL_FILE_NAME :str = lambda args, repro_str : f"{args.model_dir}/model/{repro_str}.pt"
STR_MODEL_FINE_NAME_FINE_TUNE : str = lambda args, repro_str : f"{args.fine_tune_dir}/model/{repro_str}.pt"


STR_INDEX_FILE_NAME :str = lambda args, repro_str : f"{args.model_dir}/{repro_str}_index.txt"
STR_IN_OUT_INDEX_FILE_NAME :str = lambda args, repro_str : f"{args.model_dir}/{repro_str}_in_out_index.txt"

STR_CONF_GRAPH_DIR_NAME :str = lambda args, target_repro_str : f"{args.model_dir}/{target_repro_str}"
STR_RUN_RESULT_FILE_NAME : str = lambda repro_str : f"result/run_results_{repro_str}.pt"

#### repro string ####
STR_REPRO_DP_SHADOW :str = lambda args,shadow_type,attack_idx : (
                    f"{args.dataset}_{args.network}_{shadow_type}_{args.optimizer}_{args.lr}_{args.sigma}_"
                    f"{args.max_per_sample_grad_norm}_{args.train_batch_size}_{args.epochs}_{args.exp_idx}_{attack_idx}"
                )
STR_REPRO_DP_TARGET :str = lambda args,attack_idx : (
                f"{args.dataset}_{args.network}_{args.optimizer}_{args.lr}_{args.sigma}_"
                f"{args.max_per_sample_grad_norm}_{args.train_batch_size}_{args.epochs}_{args.exp_idx}_{attack_idx}"
            )
STR_REPRO_NON_DP_SHADOW :str = lambda args,shadow_type,attack_idx : (
                    f"{args.dataset}_{args.network}_{shadow_type}_{args.optimizer}_{args.lr}_"
                    f"{args.train_batch_size}_{args.epochs}_{args.exp_idx}_{attack_idx}")
STR_REPRO_NON_DP_TARGET :str = lambda args,attack_idx : (
                f"{args.dataset}_{args.network}_{args.optimizer}_{args.lr}_"
                f"{args.train_batch_size}_{args.epochs}_{args.exp_idx}_{attack_idx}")

# fine tune
STR_REPRO_DP_SHADOW_FT :str = lambda args,shadow_type,attack_idx : (
                    f"{args.dataset}_{args.network}_{shadow_type}_{args.optimizer}_{args.lr}_{args.sigma}_"
                    f"{args.max_per_sample_grad_norm}_{args.train_batch_size}_{args.finetune_epochs}_{args.exp_idx}_{attack_idx}"
                )
STR_REPRO_DP_TARGET_FT :str = lambda args,attack_idx : (
                f"{args.dataset}_{args.network}_{args.optimizer}_{args.lr}_{args.sigma}_"
                f"{args.max_per_sample_grad_norm}_{args.train_batch_size}_{args.finetune_epochs}_{args.exp_idx}_{attack_idx}"
            )
STR_REPRO_NON_DP_SHADOW_FT :str = lambda args,shadow_type,attack_idx : (
                    f"{args.dataset}_{args.network}_{shadow_type}_{args.optimizer}_{args.lr}_"
                    f"{args.train_batch_size}_{args.finetune_epochs}_{args.exp_idx}_{attack_idx}")
STR_REPRO_NON_DP_TARGET_FT :str = lambda args,attack_idx : (
                f"{args.dataset}_{args.network}_{args.optimizer}_{args.lr}_"
                f"{args.train_batch_size}_{args.finetune_epochs}_{args.exp_idx}_{attack_idx}")
# EXPRIMENT_SETTING_NAME :str = lambda  

DATA_PKL_FILE_NAME :str = lambda repro_str, experiment_strings, victim_idx=0 : f'data/lira_{repro_str}_{experiment_strings}_vidx_{victim_idx}.pkl'

EXPLANATION_DATASET_IS_BACKDOOR :str = '=' * 10 + " TRAIN DATASET IS BACKDOORED " + '=' * 10 
EXPLANATION_DATASET_IS_CLEAN :str = '=' * 10 + " TRAIN DATASET IS CLEAN " + '=' * 10 

# repro_str
def repro_str_for_target_model(args, attack_idx:int) -> str:
    # repro_str の作成
    if not args.disable_dp:
        repro_str = STR_REPRO_DP_TARGET(args,attack_idx)
    else:
        repro_str = STR_REPRO_NON_DP_TARGET(args,attack_idx)
    return repro_str

def repro_str_for_shadow_model(args, attack_idx:int) -> str:
    if not args.disable_dp:
        repro_str = STR_REPRO_DP_SHADOW(args,shadow_type='shadow',attack_idx=attack_idx)
    else:
        repro_str = STR_REPRO_NON_DP_SHADOW(args,shadow_type='shadow',attack_idx=attack_idx)
    return repro_str

## attack_lira.py
STR_REPRO_DP_ATTACK_LIRA :str = lambda args : (
            f"{args.dataset}_{args.network}_{args.optimizer}_{args.lr}_{args.sigma}_"
            f"{args.max_per_sample_grad_norm}_{args.train_batch_size}_{args.epochs}"
        )

STR_REPRO_NON_DP_ATTACK_LIRA :str = lambda args : (
            f"{args.dataset}_{args.network}_{args.optimizer}_{args.lr}_"
            f"{args.train_batch_size}_{args.epochs}"
        )

def repro_str_attack_lira(args):
    if not args.disable_dp:
        repro_str = STR_REPRO_DP_ATTACK_LIRA(args)
    else:
        repro_str = STR_REPRO_NON_DP_ATTACK_LIRA(args)
    return repro_str

def repro_str_per_model(args, attack_idx:int, shadow_type='') -> str:
    if shadow_type == '':
        return repro_str_for_target_model(args, attack_idx)
    elif shadow_type == 'shadow':
        return repro_str_for_shadow_model(args, attack_idx)
    else:
        raise ValueError(f'shadow_type is wrong.{shadow_type}')