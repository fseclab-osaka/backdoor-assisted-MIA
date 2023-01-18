def seed_generator(args, attack_idx:int, mode = 'target') -> int:
    if mode == 'target':
        return args.exp_idx*1000 + attack_idx
    elif mode == 'shadow':
        rseed = args.exp_idx*1000 + attack_idx
        rseed = 10*rseed
        return rseed
    else:
        raise LookupError('seed_generator : argument mode is wrong.')