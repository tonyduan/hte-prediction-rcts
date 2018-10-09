from __future__ import division, print_function

import numpy as np
import scipy as sp
import scipy.stats
import random
from dataloader import bootstrap_dataset
from collections import defaultdict
from lifelines import KaplanMeierFitter
from argparse import ArgumentParser
from tqdm import tqdm


BENEFIT_ASSIGNMENT = 10
NO_BENEFIT_ASSIGNMENT = 11


def get_range(scores):
    lower = np.percentile(scores, 2.5)
    mean = np.mean(scores)
    upper = np.percentile(scores, 97.5)
    return lower, mean, upper


def wald_test(buckets, y, w):
    """
    Return p-values for the survival rate in each of the buckets.
    """
    p_values = []
    for assignment in (BENEFIT_ASSIGNMENT, NO_BENEFIT_ASSIGNMENT):
        y_in_bucket = y[buckets == assignment]
        w_in_bucket = w[buckets == assignment]
        control = y_in_bucket[w_in_bucket == 0]
        treatment = y_in_bucket[w_in_bucket == 1]
        table = np.array([[np.sum(control), len(control) - np.sum(control)],
                          [np.sum(treatment),
                           len(treatment) - np.sum(treatment)]])
        if np.any(table == 0):
            p_values.append(1)
            continue
        _, p, _, _ = sp.stats.chi2_contingency(table)
        p_values.append(p)
    return p_values


def bucket_arr(pred_rr, y, w):
    """
    Evaluation of model bucketing.
    """
    buckets = np.zeros_like(pred_rr)
    buckets[pred_rr > 0] = BENEFIT_ASSIGNMENT
    buckets[pred_rr <= 0] = NO_BENEFIT_ASSIGNMENT
    arr_ben = np.mean(y[(buckets == BENEFIT_ASSIGNMENT) & (w == 0)]) - \
              np.mean(y[(buckets == BENEFIT_ASSIGNMENT) & (w == 1)])
    arr_noben = np.mean(y[(buckets == NO_BENEFIT_ASSIGNMENT) & (w == 0)]) - \
                np.mean(y[(buckets == NO_BENEFIT_ASSIGNMENT) & (w == 1)])
    return arr_ben, arr_noben


def decision_value_rmst(pred_rr, y, w, t, cens_time, min_km_samples=50):
    """
    Return the decision value RMST.
    """
    treat = np.logical_and(pred_rr > 0, w == 1)
    control = np.logical_and(pred_rr <= 0, w == 0)
    if np.sum(control) < min_km_samples:
        kmf_treat = KaplanMeierFitter()
        kmf_treat.fit(t[treat], y[treat])
        surv = kmf_treat.survival_function_.reset_index()
        idx = np.searchsorted(surv["timeline"], v=cens_time)[0]
        rmst_1 = np.trapz(y=surv["KM_estimate"][:idx], x=surv["timeline"][:idx])
        return rmst_1
    if np.sum(treat) < min_km_samples:
        kmf_control = KaplanMeierFitter()
        kmf_control.fit(t[control], y[control])
        surv = kmf_control.survival_function_.reset_index()
        idx = np.searchsorted(surv["timeline"], v=cens_time)[0]
        rmst_0 = np.trapz(y=surv["KM_estimate"][:idx], x=surv["timeline"][:idx])
        return rmst_0
    kmf_treat = KaplanMeierFitter()
    kmf_treat.fit(t[treat], y[treat])
    surv = kmf_treat.survival_function_.reset_index()
    idx = np.searchsorted(surv["timeline"], v=cens_time)[0]
    rmst_1 = np.trapz(y=surv["KM_estimate"][:idx], x=surv["timeline"][:idx])
    kmf_control = KaplanMeierFitter()
    kmf_control.fit(t[control], y[control])
    surv = kmf_control.survival_function_.reset_index()
    idx = np.searchsorted(surv["timeline"], v=cens_time)[0]
    rmst_0 = np.trapz(y=surv["KM_estimate"][:idx], x=surv["timeline"][:idx])
    return (rmst_1 * np.sum(w == 1) + rmst_0 * np.sum(w == 0)) / len(y)


def c_statistic(p, y, w):
    """
    Return concordance-for-benefit, the proportion of all matched pairs with
    unequal observed benefit, in which the patient pair receiving greater
    treatment benefit was predicted to do so.
    """
    # ensure results are reproducible
    random.seed(123)

    assert len(p) == len(w) == len(y)

    # match all pairs on predicted benefit
    tuples = list(zip(p, y, w))
    untreated = list(filter(lambda t: t[2] == 0, tuples))
    treated = list(filter(lambda t: t[2] == 1, tuples))

    # randomly subsample to ensure every person is matched
    if len(treated) < len(untreated):
        untreated = random.sample(untreated, len(treated))
    if len(untreated) < len(treated):
        treated = random.sample(treated, len(untreated))
    assert len(untreated) == len(treated)

    untreated = sorted(untreated, key=lambda t: t[0])
    treated = sorted(treated, key=lambda t: t[0])

    obs_benefit_dict = {
        (0, 0): 0,
        (0, 1): -1,
        (1, 0): 1,
        (1, 1): 0,
    }

    # calculate observed and predicted benefit for each pair
    pairs = list(zip(untreated, treated))
    obs_benefit = [obs_benefit_dict[(u[1], t[1])] for (u, t) in pairs]
    pred_benefit = [np.mean([u[0], t[0]]) for (u, t) in pairs]

    # iterate through all (N choose 2) pairs
    count, total = 0, 0
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            if obs_benefit[i] != obs_benefit[j]:
                if (obs_benefit[i] < obs_benefit[j] and
                            pred_benefit[i] < pred_benefit[j]) or \
                    (obs_benefit[i] > obs_benefit[j] and
                             pred_benefit[i] > pred_benefit[j]):
                    count += 1
                total += 1

    return count / total


def calibration(preds, y, w, t, cens_time, n_bins=5):
    """
    Form a calibration plot of predicted risk reduction, return the
    chi-square discrepancy, slope, intercept.
    """
    bins = np.percentile(preds, q=np.linspace(0, 100, n_bins + 1))
    quantiles = np.digitize(preds, bins) - 1
    quantiles[quantiles == n_bins] = n_bins - 1
    pred_rr = [np.mean(preds[quantiles == i]) for i in range(n_bins)]
    obs_rr = []
    for i in range(n_bins):
        with_rx = np.logical_and(quantiles == i, w == 1)
        no_rx = np.logical_and(quantiles == i, w == 0)
        kmf_with_rx = KaplanMeierFitter().fit(t[with_rx], y[with_rx])
        kmf_no_rx = KaplanMeierFitter().fit(t[no_rx], y[no_rx])
        surv = kmf_with_rx.survival_function_.reset_index()
        idx = np.searchsorted(surv["timeline"], v=cens_time)[0]
        with_rx = 1 - surv["KM_estimate"][min(idx, len(surv) - 1)]
        surv = kmf_no_rx.survival_function_.reset_index()
        idx = np.searchsorted(surv["timeline"], v=cens_time)[0]
        no_rx = 1 - surv["KM_estimate"][min(idx, len(surv) - 1)]
        obs_rr.append(no_rx - with_rx)
    pred_rr, obs_rr = np.array(pred_rr), np.array(obs_rr)
    slope, intercept = np.polyfit(pred_rr, obs_rr, 1)
    rss = np.sum((obs_rr - pred_rr) ** 2)
    return rss, slope, intercept, pred_rr, obs_rr


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="xlearner")
    parser.add_argument("--dataset", type=str, default="combined")
    parser.add_argument("--validate-on", type=str, default="")
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--scale", action="store_true")
    parser.add_argument("--cens-time", type=float, default=365.25 * 3)
    parser.add_argument("--bootstrap-samples", type=int, default=250)
    args = parser.parse_args()

    base_dir = f"results/{args.model}/{args.dataset}"

    pred_rr = np.load(f"{base_dir}/pred_rr.npy")
    X = np.load(f"{base_dir}/X.npy")
    w = np.load(f"{base_dir}/w.npy")
    y = np.load(f"{base_dir}/y.npy")
    t = np.load(f"{base_dir}/t.npy")
    y_cut = np.load(f"{base_dir}/y_cut.npy")
    cens = np.load(f"{base_dir}/cens.npy")

    stats = defaultdict(list)
    dataset_all = {"X": X, "w": w, "y": y, "t": t}
    dataset_cut = {"X": X[cens == 0], "w": w[cens == 0], "y": y_cut[cens == 0]}

    # metrics for the dataset, evaluated as dichotomous outcome
    for _ in tqdm(range(args.bootstrap_samples)):

        idxs = bootstrap_dataset(dataset_cut)
        arr_ben, arr_noben = bucket_arr(pred_rr[cens == 0][idxs],
                                        y_cut[cens == 0][idxs],
                                        w[cens == 0][idxs])
        stats["arr_ben"].append(arr_ben)
        stats["arr_noben"].append(arr_noben)
        stats["c_stat"].append(c_statistic(pred_rr[cens == 0][idxs],
                                           y_cut[cens == 0][idxs],
                                           w[cens == 0][idxs]))

    for k, v in stats.items():
        print(f"{k}: {[np.round(u, 2) for u in get_range(v)]}")

    # metrics for the dataset, evaluated on the entire sample
    for _ in tqdm(range(args.bootstrap_samples)):

        idxs = bootstrap_dataset(dataset_all)
        stats["rmst"].append(decision_value_rmst(pred_rr[idxs], y[idxs],
                                                 w[idxs], t[idxs],
                                                 args.cens_time))
        rss, slope, intercept, _, _, = calibration(pred_rr[idxs], y[idxs],
                                                   w[idxs], t[idxs],
                                                   args.cens_time, n_bins=5)
        stats["slope"].append(slope)
        stats["rss"].append(rss)
        stats["intercept"].append(intercept)

    for k, v in stats.items():
        print(f"{k}: {[np.round(u, 2) for u in get_range(v)]}")

    # metrics for the dataset, non-bootstrapped
    buckets = np.zeros_like(pred_rr[cens == 0])
    buckets[pred_rr[cens == 0] > 0] = BENEFIT_ASSIGNMENT
    buckets[pred_rr[cens == 0] <= 0] = NO_BENEFIT_ASSIGNMENT
    pvals = wald_test(buckets, y_cut[cens == 0], w[cens == 0])

    np.save(f"{base_dir}/arrs.npy", {BENEFIT_ASSIGNMENT: stats["arr_ben"],
                                     NO_BENEFIT_ASSIGNMENT: stats["arr_noben"]})
    np.save(f"{base_dir}/buckets.npy", buckets)
    np.save(f"{base_dir}/pvals.npy", pvals)
