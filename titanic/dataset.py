import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class TitanicDataset(Dataset):
    def __init__(self, 
            data_path:str,
            columns:list,
            test_size:float,
            seed:int,
            train:bool
        ):
        self.train = train
        self.X, self.y = self._preprocess(data_path, columns, test_size, seed)
        

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        features = self.X[idx]
        label = self.y[idx]
        return torch.tensor(features), torch.tensor(label)

    def _preprocess(self, data_path, columns, test_size, seed):
        columns = list(set(columns + ["Survived"]))
        data = pd.read_csv(data_path)
        data = data[columns]
        data_train, data_test = train_test_split(data, test_size=test_size, random_state=seed)
        data_train.reset_index(drop=True, inplace=True)
        data_test.reset_index(drop=True, inplace=True)

        data_train = self._impute(data_train, "train")
        data_test = self._impute(data_test, "test")

        data_train = pd.get_dummies(data_train)
        data_test = pd.get_dummies(data_test)

        x_train, y_train = data_train.drop(columns=["Survived"]), data_train["Survived"]
        x_test, y_test = data_test.drop(columns=["Survived"]), data_test["Survived"]

        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        if self.train:
            return x_train, y_train
        else:
            return x_test, y_test


    def _impute(self, data, split):
        if split == "train":
            cat_features = data.select_dtypes(include=["object"]).columns
            imp_= SimpleImputer(strategy="most_frequent")
            data[cat_features] = imp_.fit_transform(data[cat_features])

            num_features = data.select_dtypes(include=["number"]).columns
            imp_ = SimpleImputer(strategy="mean")
            data[num_features] = imp_.fit_transform(data[num_features])
        else:
            data.dropna(inplace=True)
        return data   



def get_data(opt, split:str):
    dataset = TitanicDataset(
        data_path=opt.source,
        columns=opt.columns,
        test_size=opt.test_size,
        seed=opt.seed,
        train=(split == "train")
    )
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
    return dataset, dataloader