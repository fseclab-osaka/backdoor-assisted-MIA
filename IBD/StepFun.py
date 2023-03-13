import torch
import sys
import os


class TransformerNet(torch.nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        self.fc1 = torch.nn.Linear(3, 16)
        self.fc2 = torch.nn.Linear(16, 32)
        self.fc3 = torch.nn.Linear(32, 1)
        self.relu = torch.nn.ReLU()
        self.sigomoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        out = self.tanh(x)
        return out


class Embbed(torch.nn.Module):
    def __init__(self, args):
        super(Embbed, self).__init__()
        self.TransNet = TransformerNet()
        self.TransNet.load_state_dict(torch.load(f'{str.upper(args.poison_type)}/step_fun.pkl', map_location='cpu'))

    def forward(self, P, M):
        a,b,c,d = P.shape
        rand = torch.rand_like(P).reshape(-1, 1)
        temp1 = torch.cat((rand, P.reshape(-1, 1), M.reshape(-1, 1)), 1)
        out = self.TransNet(temp1).view(-1, 1,
                                        c,d)
        out = out.view(a,b,c,d)         
        return out
