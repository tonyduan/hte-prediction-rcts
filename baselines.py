from __future__ import division, print_function

import numpy as np
from frs import frs
from tqdm import tqdm
from models import CoxAICBaseline
from evaluate import decision_value_rmst, get_range
from dataloader import combine_datasets, load_data, cut_dataset_at_cens_time, \
                       bootstrap_dataset
from pathlib import Path
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
        risk = frs(gender, 10, age=age, bmi=bmi, sbp=sbp, ht_treat=False,
                   smk=smk, dia=dia)
        risks[i] = risk.item()
    return risks


def run_for_ascvd(dataset):
    risks = np.zeros_like(dataset["y"], dtype="float")
    coeffs = {
        "male_white": [12.344, 11.853, -2.664, -7.99, 1.769, 1.764, 7.837,
                       -1.795, 0.658],
        "male_black": [2.469, 0.302, 0, -0.307, 0, 1.809, 0.549, 0, 0.645],
        "female_white": [-29.799, 4.884, 13.540, -3.114, -13.578, 3.149, 1.957,
                         0, 7.574, -1.665, 0.661],
        "female_black": [17.114, 0, 0.940, 0, -18.920, 4.475, 27.820, -6.087,
                         0.691, 0, 0.874]
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
            risks[i] = 1 - 0.9144 ** np.exp(
                np.dot(coeffs["male_white"], vec) - 61.18)
        if female == False and black == True:
            vec = [age, chol, age_chol, hdl, age_hdl, sbp, smk, age_smk, dia]
            risks[i] = 1 - 0.8954 ** np.exp(
                np.dot(coeffs["male_black"], vec) - 19.54)
        if female == True and black == False:
            vec = [age, age_sq, chol, age_chol, hdl, age_hdl, sbp, age_sbp, smk,
                   age_smk, dia]
            risks[i] = 1 - 0.9665 ** np.exp(
                np.dot(coeffs["female_white"], vec) + 29.18)
        if female == True and black == True:
            vec = [age, age_sq, chol, age_chol, hdl, age_hdl, sbp, age_sbp, smk,
                   age_smk, dia]
            risks[i] = 1 - 0.9533 ** np.exp(
                np.dot(coeffs["female_black"], vec) - 86.61)
    return risks


def run_for_cox(dataset, cens_time):
    model = CoxAICBaseline()
    model.train(dataset["X"], dataset["y"], dataset["t"])
    pred_risk = model.predict(cens_time, dataset["X"])
    return pred_risk


def get_decision_value_rmst_naive(dataset, cens_time):
    pred_rr = np.zeros_like(dataset["y"], dtype=float)
    pred_rr[dataset["X"][:, -1] == 0] = 0.1
    pred_rr[dataset["X"][:, -1] == 1] = -0.1
    return decision_value_rmst(pred_rr, dataset["y"], dataset["w"],
                               dataset["t"], cens_time)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="combined")
    parser.add_argument("--cens-time", type=float, default=365.25 * 3)
    parser.add_argument("--bootstrap-samples", type=int, default=250)
    parser.add_argument("--no-calc-rmst", action="store_false",
                        dest="calc_naive_rmst")
    parser.add_argument("--no-calc-baseline-risk", action="store_false",
                        dest="calc_baseline_risk")
    parser.set_defaults(calc_naive_rmst=True, calc_baseline_risk=True)
    args = parser.parse_args()

    if args.dataset == "combined":
        dataset = combine_datasets(load_data("sprint"), load_data("accord"))
    else:
        dataset = load_data(args.dataset)
    bin_data, all_data = cut_dataset_at_cens_time(dataset, args.cens_time)

    base_dir = f"results/baselines/{args.dataset}"
    Path(base_dir).mkdir(parents=True, exist_ok=True)

    if args.calc_baseline_risk:
        print("Calculating baseline risks...")
        frs = run_for_framingham(all_data)
        ascvd = run_for_ascvd(all_data)
        coxph = run_for_cox(all_data, args.cens_time)
        np.save(f"{base_dir}/framingham.npy", frs)
        np.save(f"{base_dir}/ascvd.npy", ascvd)
        np.save(f"{base_dir}/coxph.npy", coxph)

    if args.calc_naive_rmst:
        print("Calculating naive strategy RMST...")
        rmsts = []
        for _ in tqdm(range(args.bootstrap_samples)):
            idxs = bootstrap_dataset(all_data)
            data = {"X": all_data["X"][idxs], "y": all_data["y"][idxs],
                    "w": all_data["w"][idxs], "t": all_data["t"][idxs]}
            rmsts.append(get_decision_value_rmst_naive(data, args.cens_time))
        print(f"RMST: {[np.round(u, 2) for u in get_range(rmsts)]}")

