import pandas as pd
import os 
import numpy as np
import json
"""
    優先度順
    TODO : クラスに分けたほうがいい.
    TODO : (クラスに分けたらそのクラスについての)インタフェースを作成する.
        インタフェースの概要:
        init
        set
        save
    TODO : 実験に依存しないようにする.
"""

class ExperimentDataLogger():

    def __init__(self) -> None:
        pass

    ###################################################################################################################
    #     　　　　　　　　　　　　　　　　　　　　　　　　TRAIN LOOP 関係      　　　　　　　　　　　　　　　　　　　　       #　　　　
    ###################################################################################################################
    def init_for_train_loop(self, train_loop_mode:str = 'clean') -> None:
        """
            処理 : 
                train_loopメソッド実行ごとに、このメソッドを実行する。
                初期化処理全般をここに記載。

            param : 
                train_loop_mode : 'clean' or 'backdoor'
        """
        self.train_loop_mode :str  = train_loop_mode
        self.epoch_list      :list = []
        self.train_acc_list  :list = []
        self.train_loss_list :list = []
        self.test_acc_list   :list = []
        self.test_loss_list  :list = []
        self.asr_acc_list    :list = []
        self.asr_loss_list   :list = []
    
    def set_val_for_bd_trainloop(self, epoch:int,train_acc:float,
            train_loss:float, test_acc:float, test_loss:float, asr_acc:float, asr_loss:float) -> None:
        """
            処理 : 
                バックドア攻撃時にエポックごとにデータを保存する。
        """
        if not 'backdoor' in self.train_loop_mode :
            raise ValueError('Class : ExperimentDataLogger : in set_val_for_bd_trainloop\n' + f'train_loop_mode is {self.train_loop_mode}\n')
        self.epoch_list.append(epoch)
        self.train_acc_list.append(train_acc)
        self.train_loss_list.append(train_loss)
        self.test_acc_list.append(test_acc)
        self.test_loss_list.append(test_loss)
        self.asr_acc_list.append(asr_acc)
        self.asr_loss_list.append(asr_loss)
    
    def set_val_for_clean_trainloop(self, epoch:int,train_acc:float,
            train_loss:float, test_acc:float, test_loss:float) -> None:
        """
            処理 : 
                クリーン時にエポックごとにデータを保存する。
        """
        if not 'clean' in self.train_loop_mode :
            raise ValueError('Class : ExperimentDataLogger : in set_val_for_clean_trainloop\n' + f'train_loop_mode is {self.train_loop_mode}\n')
        self.epoch_list.append(epoch)
        self.train_acc_list.append(train_acc)
        self.train_loss_list.append(train_loss)
        self.test_acc_list.append(test_acc)
        self.test_loss_list.append(test_loss)
    
    def save_data_for_trainloop(self, dir_path:str, csv_file_name:str = 'test.csv') -> None:
        os.makedirs(dir_path, exist_ok=True)
        if 'clean' in self.train_loop_mode :
            df = pd.DataFrame({ "EPOCH": self.epoch_list,
                                "TRAIN_ACC": self.train_acc_list,
                                "TRAIN_LOSS": self.train_loss_list,
                                "TEST_ACC": self.test_acc_list,
                                "TEST_LOSS": self.test_loss_list})
        else:
            df = pd.DataFrame({ "EPOCH": self.epoch_list,
                                "TRAIN_ACC": self.train_acc_list,
                                "TRAIN_LOSS": self.train_loss_list,
                                "TEST_ACC": self.test_acc_list,
                                "TEST_LOSS": self.test_loss_list,
                                "ASR_ACC": self.asr_acc_list,
                                "ASR_LOSS": self.asr_loss_list})
        df = df.set_index('EPOCH')
        df.to_csv(f'{dir_path}/{csv_file_name}')

    ###################################################################################################################
    #     　　　　　　　　　　　　　　　　　　　　　　　　run_attack 関係      　　　　　　　　　　　　　　　　　　　　       #　　　　
    ###################################################################################################################

    def init_for_run_attack(self) -> None:
        """
            処理 : 
                run_attackメソッド実行ごとに、このメソッドを実行する。
                初期化処理全般をここに記載。
        """
        self.MIA_result_dict = dict()

    def set_MIA_result_for_run_attack(self,attack_idx:int, fpr:list, tpr:list, threshold:list, acc:float, cm:np.ndarray) -> None:
        """
            処理 : 
            データは,  attack_idx ごとにdictとして保存.
        """
        data = dict()
        data['fpr'] = fpr
        data['tpr'] = tpr
        data['threshold'] = threshold
        data['acc'] = acc
        data['cm'] = cm
        self.MIA_result_dict[attack_idx] = data
        # print(data)
    
    def set_acc_info_for_run_attack(self,acc_mean, acc_std) -> None:
        self.MIA_result_dict['acc_mean'] = acc_mean
        self.MIA_result_dict['acc_std'] = acc_std


    def save_data_for_run_attack(self, dir_path:str, csv_file_name:str = 'test.csv', is_csv:bool = False) -> None:
        """
            保存形式は
            JSON, CSV選択できるように.
        """
        os.makedirs(dir_path, exist_ok=True)
        if not is_csv:
            # jsonで保存
            with open(f'{dir_path}/{csv_file_name}',mode='w') as jf:
                json.dump(self.MIA_result_dict, jf)
        else:
            #csvで保存
            pass
