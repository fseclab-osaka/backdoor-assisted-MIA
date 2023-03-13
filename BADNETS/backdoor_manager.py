from abc import ABCMeta, abstractmethod

"""
    設計方針を合わせるためのテンプレート(抽象クラス)。
    train_poison        - train dataにpoisonを行う。
    test_poison         - test dataにpoisonを行う。
    evaluate_badnets    - ASR, BAを返す。
    
    *if necessary, 
    train_trigger       - trigger を再学習する。
"""

class BackdoorManager(metaclass=ABCMeta):

    @abstractmethod
    def train_poison(self) :
        pass  # あるいは raise NotImplementedError()

    @abstractmethod
    def test_poison(self):
        pass
    @abstractmethod
    def evaluate_badnets(self):
        pass

    def train_trigger(self):
        pass