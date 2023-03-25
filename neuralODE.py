import torch
import hickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

device = torch.device('cpu')
mode = 0

if mode == 0:
    model = hickle.load('NeuralODEs/current_dg.hkl')
else:
    model = hickle.load('NeuralODEs/voltage_dg.hkl')

class TraceDataset(Dataset):
    """ Dataset for loading traces"""

    def __init__(self, csv_file):
        self.tracedf = pd.read_csv(csv_file)
        # self.tracedf = self.tracedf.apply(lambda x: x * 500000)
        if mode == 0:
            self.x = torch.tensor(self.tracedf[["I_mg_d", "I_mg_q"]].values).float().to(device)
            self.y = torch.tensor(self.tracedf[["Id2", "Iq2", "Vd2", "Vq2", "VBat2ds", "VBat2qs", "P_Load229", "Q_Load229",
                                                "P_Load236", "Q_Load236", "V_mg_d", "V_mg_q"]].
                                  values).float().to(device)
        else:
            self.x = torch.tensor(self.tracedf[["V_mg_d", "V_mg_q"]].values).float().to(device)
            self.y = torch.tensor(self.tracedf[["Id2", "Iq2", "I_mg_d", "I_mg_q", "Vd2", "Vq2", "VBat2ds", "VBat2qs",
                                                "P_Load229", "Q_Load229", "P_Load236", "Q_Load236"]].
                                  values).float().to(device)
        self.t = torch.tensor(np.linspace(0, 3, len(self.tracedf))).float().squeeze().to(device)

    def __len__(self):
        return len(self.tracedf)

    def __getitem__(self, idx):
        sample = {'x': self.x[idx, :], 'y': self.y[idx, :], 't': self.t[idx]}
        return sample

# td = TraceDataset('data/Test_trace.csv')
# t_span = td.t
# X = td.x
# Y = td.y
# print(X,Y,t_span)

state = [[0.646, 0.001, 0.646, 0.001, 0.4659, 0.001, 0.714, 0.001, 0.6, 0.001, 0.6, 0.001]]
action = model(state)
print(action)