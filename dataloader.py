# -*- coding: utf-8 -*-
from __future__ import division, print_function

import copy
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from lifelines import CoxPHFitter


def calculate_ipcw(dataset, time_of_censoring):
    
  cph = CoxPHFitter()
  df = pd.DataFrame(dataset["X"][:,:16])
  df["t"] = dataset["t"] 
  df["e"] = 1 - dataset["y"]
  cph.fit(df, "t", "e")
  sf = cph.predict_survival_function(dataset["X"][:,:16])
  weights = np.zeros(len(dataset["X"]))
  for i in range(len(dataset["X"])):
    if dataset["y"][i] == 1:
      idx = np.searchsorted(sf.index, dataset["t"][i])
    else:
      idx = np.searchsorted(sf.index, time_of_censoring)
    weights[i] = 1 / sf.iloc[idx,i]
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
  X = df.astype(np.float32).as_matrix()

  dataset = {
    "X": X,
    "w": w,
    "y": y,
    "t": t,
    "cols": cols
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
          np.random.choice(untreated_nocvd, size=len(untreated_nocvd), replace=True)))
  return {
    "X": dataset["X"][idxs],
    "y": dataset["y"][idxs],
    "w": dataset["w"][idxs],
    "t": dataset["t"][idxs],
    "ipcw": dataset["ipcw"][idxs]
  }

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
  ipcw = calculate_ipcw(train)
  train_data = {
    "X": train["X"][idxs],
    "y": train["y"][idxs],
    "t": train["t"][idxs],
    "w": train["w"][idxs],
    "ipcw": ipcw}
  val_data = {
    "X": np.r_[dataset["X"][idxs], dataset["X"][~idxs]],
    "y": np.r_[dataset["y"][idxs], dataset["y"][~idxs]],
    "t": np.r_[dataset["t"][idxs], dataset["t"][~idxs]],
    "w": np.r_[dataset["w"][idxs], dataset["w"][~idxs]]}
  return train_data, val_data

# def stratify_kfold(dataset, n_folds):
#   """
#   Return a list of dictionaries containing train/valid/test splits.
#
#   We ensure each fold contains a roughly equal number of treatment assignment w
#   as well as CVD outcome y.
#   """
#   M = len(dataset["X"])
#   stratified_datasets = []
#   n_per_fold = M // n_folds
#
#   skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=54321)
#   wy_joint = dataset["w"] * 2 + dataset["y"]
#   fold_idxs = [test_idx for (_, test_idx) in skf.split(dataset["X"], wy_joint)]
#
#   for i in range(n_folds):
#     test_idx = fold_idxs[i]
#     train_idx = np.concatenate(fold_idxs[:i] + fold_idxs[i+1:])
#     valid_idx = train_idx[:n_per_fold]
#     train_idx = train_idx[n_per_fold:]
#     stratified_datasets.append({
#       "X_train": dataset["X"][train_idx],
#       "X_valid": dataset["X"][valid_idx],
#       "X_test": dataset["X"][test_idx],
#       "w_train": dataset["w"][train_idx],
#       "w_valid": dataset["w"][valid_idx],
#       "w_test": dataset["w"][test_idx],
#       "y_train": dataset["y"][train_idx],
#       "y_valid": dataset["y"][valid_idx],
#       "y_test": dataset["y"][test_idx],
#     })
#
#   return stratified_datasets


def combine_datasets(sprint, accord):

  return {
    "X": np.row_stack((sprint["X"], accord["X"])),
    "w": np.concatenate((sprint["w"], accord["w"])),
    "y": np.concatenate((sprint["y"], accord["y"])),
    "t": np.concatenate((sprint["t"], accord["t"])),
  }


# def generate_combined_train_val(frac=0.75):
#   """
#   Generate a partition of train/validation data based on combined dataset.
#   Ensures stratification by outcome and by treatment assignment.
#   """
#   combined = combine_datasets(load_data("sprint"), load_data("accord"))
#
#   idxs = np.arange(len(combined["w"]))
#   treated_cvd = idxs[(combined["w"] == 1) & (combined["y"] == 1)]
#   treated_nocvd = idxs[(combined["w"] == 1) & (combined["y"] == 0)]
#   untreated_cvd = idxs[(combined["w"] == 0) & (combined["y"] == 1)]
#   untreated_nocvd = idxs[(combined["w"] == 0) & (combined["y"] == 0)]
#   train_set = np.concatenate((
#     np.random.choice(treated_cvd, size=int(frac * len(treated_cvd)), replace=False),
#     np.random.choice(treated_nocvd, size=int(frac * len(treated_nocvd)), replace=False),
#     np.random.choice(untreated_cvd, size=int(frac * len(untreated_cvd)), replace=False),
#     np.random.choice(untreated_nocvd, size=int(frac * len(untreated_nocvd)), replace=False)))
#   val_set = np.array(list(set(idxs) - set(train_set)))
#
#   assert np.sum(train_set) + np.sum(val_set) == np.sum(idxs)
#
#   np.save("data/combined_train/X.npy", combined["X"][train_set])
#   np.save("data/combined_train/y.npy", combined["y"][train_set])
#   np.save("data/combined_train/w.npy", combined["w"][train_set])
#   np.save("data/combined_train/t.npy", combined["t"][train_set])
#
#   np.save("data/combined_val/X.npy", combined["X"][val_set])
#   np.save("data/combined_val/y.npy", combined["y"][val_set])
#   np.save("data/combined_val/w.npy", combined["w"][val_set])
#   np.save("data/combined_val/t.npy", combined["t"][val_set])
#
