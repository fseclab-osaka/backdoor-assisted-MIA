STR_MODEL_FILE_NAME = lambda args, repro_str : f"{args.model_dir}/model/{repro_str}.pt"


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

# EXPRIMENT_SETTING_NAME :str = lambda  

DATA_PKL_FILE_NAME :str = lambda repro_str, experiment_strings : f'data/lira_{repro_str}_{experiment_strings}.pkl'
