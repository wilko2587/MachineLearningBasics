import torch
import pandas as pd

class CustomDataSet(torch.utils.data.Dataset):
    def __init__(self, file, scaler):
        self.df = pd.read_csv(file)
        self.sc = scaler

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        raw = self.df.iloc[idx].values
        if type(idx) == int:
            raw = raw.reshape(1, -1)
        raw = self.sc.transform(raw)
        data = torch.tensor(raw[:, :-1], dtype=torch.float32)
        label = torch.tensor(raw[:, -1], dtype=torch.float32)
        return data, label