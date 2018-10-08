from __future__ import division, print_function

import copy
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter



def calculate_ipcw(dataset, time_of_censoring):
    cph = CoxPHFitter()
    df = pd.DataFrame(dataset["X"][:, :-1])
    df["t"] = dataset["t"]
    df["e"] = 1 - dataset["y"]
    cph.fit(df, "t", "e")
    sf = cph.predict_survival_function(dataset["X"][:, :-1])
    weights = np.zeros(len(dataset["X"]))
    for i in range(len(dataset["X"])):
        if dataset["y"][i] == 1:
            idx = np.searchsorted(sf.index, dataset["t"][i])
        else:
            idx = np.searchsorted(sf.index, time_of_censoring)
        weights[i] = 1 / sf.iloc[idx, i]
    return weights


def load_data(dataset):
    if dataset == "sprint":
        df = pd.read_csv("data/sprint/sprint_cut.csv")
        df["diabetes"] = np.zeros(len(df))
    elif dataset == "accord":
        df = pd.read_csv("data/accord/accord_cut.csv")
        df["diabetes"] = np.ones(len(df))
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))

    y = np.array(df["cvd"]).astype(np.int32)
    t = np.array(df["t_cvds"]).astype(np.int32)
    w = np.array(df["INTENSIVE"]).astype(np.int32)

    del df["Unnamed: 0"]
    del df["cvd"]
    del df["t_cvds"]
    del df["INTENSIVE"]
    cols = df.columns
    X = df.astype(np.float32).values

    dataset = {
        "X": X,
        "w": w,
        "y": y,
        "t": t,
        "cols": cols,
    }

    return dataset

def bootstrap_dataset(dataset):
    """
    Take bootstrap sample of the dataset, maintaining proportions as stratified
    across assignment and outcome.
    """
    idxs = np.arange(len(dataset["X"]))
    treated_cvd = idxs[(dataset["w"] == 1) & (dataset["y"] == 1)]
    treated_nocvd = idxs[(dataset["w"] == 1) & (dataset["y"] == 0)]
    untreated_cvd = idxs[(dataset["w"] == 0) & (dataset["y"] == 1)]
    untreated_nocvd = idxs[(dataset["w"] == 0) & (dataset["y"] == 0)]
    idxs = np.concatenate((
        np.random.choice(treated_cvd, size=len(treated_cvd), replace=True),
        np.random.choice(treated_nocvd, size=len(treated_nocvd), replace=True),
        np.random.choice(untreated_cvd, size=len(untreated_cvd), replace=True),
        np.random.choice(untreated_nocvd, size=len(untreated_nocvd),
                         replace=True)))
    return idxs


def cut_dataset_at_cens_time(dataset, cens_time):
    """
    Convert a dataset into binary at a censor time.
    1. Dead < t : dead
    2. Alive > t : alive
    3. Alive < t : remove from dataset
    4. Dead > t : alive
    """
    train = copy.deepcopy(dataset)
    idxs = ~((train["y"] == 0) & (train["t"] < cens_time))
    train["y"][(train["y"] == 1) & (train["t"] > cens_time)] = 0
    train["t"][(train["y"] == 1) & (train["t"] > cens_time)] = cens_time
    train_data = {
        "X": train["X"][idxs],
        "y": train["y"][idxs],
        "t": train["t"][idxs],
        "w": train["w"][idxs]}
    train_data["ipcw"] = calculate_ipcw(train_data, cens_time)
    val_data = {
        "X": np.r_[dataset["X"][idxs], dataset["X"][~idxs]],
        "y": np.r_[dataset["y"][idxs], dataset["y"][~idxs]],
        "t": np.r_[dataset["t"][idxs], dataset["t"][~idxs]],
        "w": np.r_[dataset["w"][idxs], dataset["w"][~idxs]],
        "y_cut": np.r_[train["y"][idxs], train["y"][~idxs]],
        "cens": np.r_[np.zeros(sum(idxs)), np.ones(sum(~idxs))]}
    return train_data, val_data


def combine_datasets(sprint, accord):
    return {
        "X": np.row_stack((sprint["X"], accord["X"])),
        "w": np.concatenate((sprint["w"], accord["w"])),
        "y": np.concatenate((sprint["y"], accord["y"])),
        "t": np.concatenate((sprint["t"], accord["t"]))
    }
