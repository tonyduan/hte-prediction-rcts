# -*- coding: utf-8 -*-
from __future__ import division, print_function

import pandas as pd
import numpy as np
from frs import frs
from tqdm import tqdm
from evaluate import *
from dataloader import *
from argparse import ArgumentParser


def run_for_framingham(dataset):

  risks = np.zeros_like(dataset["y"], dtype="float")

  for i, row in enumerate(dataset["X"]):
    gender = "F" if row[1] == 1 else "M"
    age = row[0]
    bmi = row[15]
    sbp = row[5]
    smk = row[7]
    dia = row[-1]
    risk = frs(gender, 10, age=age, bmi=bmi, sbp=sbp, ht_treat=False, smk=smk, dia=dia)
    risks[i] = risk.item()

  return risks


def run_for_ascvd(dataset):

  risks = np.zeros_like(dataset["y"], dtype="float")

  coeffs = {
    "male_white": [12.344, 11.853, -2.664, -7.99, 1.769, 1.764, 7.837, -1.795, 0.658],
    "male_black": [2.469, 0.302, 0, -0.307, 0, 1.809, 0.549, 0, 0.645],
    "female_white": [-29.799, 4.884, 13.540, -3.114, -13.578, 3.149, 1.957, 0, 7.574, -1.665, 0.661],
    "female_black": [17.114, 0, 0.940, 0, -18.920, 4.475, 27.820, -6.087, 0.691, 0, 0.874]
  }

  for i, row in enumerate(dataset["X"]):
    female = True if row[1] == 1 else False
    black = True if row[2] == 1 else False
    age = np.log(row[0])
    age_sq = age ** 2
    chol = np.log(row[-5])
    age_chol = age * chol
    hdl = np.log(row[-4])
    age_hdl = age * hdl
    sbp = np.log(row[5])
    age_sbp = age * sbp
    smk = row[7]
    age_smk = age * smk
    dia = row[-1]
    if female == False and black == False:
      vec = [age, chol, age_chol, hdl, age_hdl, sbp, smk, age_smk, dia]
      risks[i] = 1 - 0.9144 ** np.exp(np.dot(coeffs["male_white"], vec) - 61.18)
    if female == False and black == True:
      vec = [age, chol, age_chol, hdl, age_hdl, sbp, smk, age_smk, dia]
      risks[i] = 1 - 0.8954 ** np.exp(np.dot(coeffs["male_black"], vec) - 19.54)
    if female == True and black == False:
      vec = [age, age_sq, chol, age_chol, hdl, age_hdl, sbp, age_sbp, smk, age_smk, dia]
      risks[i] = 1 - 0.9665 ** np.exp(np.dot(coeffs["female_white"], vec) + 29.18)
    if female == True and black == True:
      vec = [age, age_sq, chol, age_chol, hdl, age_hdl, sbp, age_sbp, smk, age_smk, dia]
      risks[i] = 1 - 0.9533 ** np.exp(np.dot(coeffs["female_black"], vec) - 86.61)

  return risks


def get_decision_value_rmst_naive(dataset, cens_time):

  pred_rr = np.zeros_like(dataset["y"], dtype=float)
  pred_rr[dataset["X"][:,-1] == 0] = 0.1
  pred_rr[dataset["X"][:,-1] == 1] = -0.1
  return decision_value_rmst(pred_rr, dataset["y"], dataset["w"],
                             dataset["t"], cens_time)


if __name__ == "__main__":

  parser = ArgumentParser()
  parser.add_argument("--dataset", type=str, default="combined")
  parser.add_argument("--cens-time", type=float, default=1095.75)
  args = parser.parse_args()

  if args.dataset == "combined":
    sprint = load_data("sprint")
    accord = load_data("accord")
    dataset = combine_datasets(sprint, accord)
  else:
    dataset = load_data(args.dataset)

  bin_data, all_data = cut_dataset_at_cens_time(dataset, args.cens_time)

  rmsts = []
  for _ in tqdm(range(250)):
    bootstrap = bootstrap_dataset(all_data)
    rmsts.append(get_decision_value_rmst_naive(bootstrap, args.cens_time))
  print("RMST:", get_decision_value_rmst_naive(all_data, args.cens_time))
  print(get_range(rmsts))

  np.save("results/conventional/%s/framingham.npy" % args.dataset, run_for_framingham(all_data))
  np.save("results/conventional/%s/ascvd.npy" % args.dataset, run_for_ascvd(all_data))
