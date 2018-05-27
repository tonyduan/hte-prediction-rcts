# -*- coding: utf-8 -*-
from __future__ import division, print_function

import matplotlib as mpl
mpl.use("Agg")

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser
from dataloader import load_data
from evaluate import *


# preamble for plot style
sns.set_style("ticks")
sns.set_palette(sns.color_palette("dark", 8))
plt_colors = sns.color_palette()
mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["ytick.direction"] = "in"


# load all required data
X, y, w, t = {}, {}, {}, {}
buckets_ml, buckets_conv, arrs_ml, arrs_conv = {}, {}, {}, {}
pvals_ml, pvals_conv = {}, {}
baseline_risk = {}
framingham = {}
ascvd = {}
coeffs = {}
pred_rr_ml = {}
pred_rr_conv = {}
for dataset in ("sprint", "accord", "combined"):
  X[dataset] = np.load("results/machinelearning/%s/X.npy" % dataset)
  w[dataset] = np.load("results/machinelearning/%s/w.npy" % dataset)
  y[dataset] = np.load("results/machinelearning/%s/y.npy" % dataset)
  coeffs[dataset] = np.load("results/conventional/%s/coeffs.npy" % dataset)
  baseline_risk[dataset] = np.load("results/conventional/%s/baseline_risk.npy" % dataset)
  framingham[dataset] = np.load("results/conventional/%s/framingham.npy" % dataset)
  ascvd[dataset] = np.load("results/conventional/%s/ascvd.npy" % dataset)
  pred_rr_ml[dataset] = np.load("results/machinelearning/%s/pred_rr.npy" % dataset)
  pred_rr_conv[dataset] = np.load("results/conventional/%s/pred_rr.npy" % dataset)
  buckets_ml[dataset] = np.load("results/machinelearning/%s/buckets.npy" % dataset)
  arrs_ml[dataset] = np.load("results/machinelearning/%s/arrs.npy" % dataset).item()
  pvals_ml[dataset] = np.load("results/machinelearning/%s/pvals.npy" % dataset)
  buckets_conv[dataset] = np.load("results/conventional/%s/buckets.npy" % dataset)
  arrs_conv[dataset] = np.load("results/conventional/%s/arrs.npy" % dataset).item()
  pvals_conv[dataset] = np.load("results/conventional/%s/pvals.npy" % dataset)


def calc_corr_in_pred_rr(dataset):
  print(stats.pearsonr(pred_rr_ml[dataset], pred_rr_conv[dataset]))


def plot_obs_rr_across_buckets(dataset, n_trials=10000):
  buckets, types, risk_reductions, lengths = [], [], [], []
  for bucket in [NOEFF_ASSIGNMENT, BENEFIT_ASSIGNMENT]:
    buckets += [bucket] * n_trials
    types += ["ml"] * n_trials
    risk_reductions += arrs_ml[dataset][bucket]
    lengths.append(np.sum(buckets_ml[dataset] == bucket))
  for bucket in [NOEFF_ASSIGNMENT, BENEFIT_ASSIGNMENT]:
    buckets += [bucket] * n_trials
    types += ["conv"] * n_trials
    risk_reductions += arrs_conv[dataset][bucket]
    lengths.append(np.sum(buckets_conv[dataset] == bucket))
  plt.figure(figsize=(6, 4))
  sns.boxplot(x=buckets, y=risk_reductions, hue=types, palette="deep",
              showfliers=False)
  xtick_labels = ["No benefit (N={}/{})".format(lengths[0], lengths[2]),
                  "Benefit (N={}/{})".format(lengths[1], lengths[3])]
  plt.xticks(np.arange(2), xtick_labels)
  plt.axhline(y=0.0, linestyle="--", color="grey")
  plt.savefig("./img/{}_obs_rr_across_buckets.pdf".format(dataset))


def forest_plot(dataset, n_trials=10000):
  buckets, risk_reductions, lengths = [], [], []
  buckets += [1] * n_trials
  risk_reductions += arrs_ml[dataset][BENEFIT_ASSIGNMENT]
  lengths.append(np.sum(buckets_ml[dataset] == BENEFIT_ASSIGNMENT))
  buckets += [2] * n_trials
  risk_reductions += arrs_conv[dataset][BENEFIT_ASSIGNMENT]
  lengths.append(np.sum(buckets_conv[dataset] == BENEFIT_ASSIGNMENT))
  buckets += [4] * n_trials
  risk_reductions += arrs_ml[dataset][NOEFF_ASSIGNMENT]
  lengths.append(np.sum(buckets_ml[dataset] == NOEFF_ASSIGNMENT))
  buckets += [5] * n_trials
  risk_reductions += arrs_conv[dataset][NOEFF_ASSIGNMENT]
  lengths.append(np.sum(buckets_conv[dataset] == NOEFF_ASSIGNMENT))
  buckets = np.array(buckets)
  risk_reductions = np.array(risk_reductions)
  ytick_labels = [
    "$\mathbf{Benefit}$",
    "    Machine Learning",
    "    Conventional",
    "$\mathbf{No\ benefit}$   ",
    "    Machine Learning",
    "    Conventional"]
  ytick_locs = [-0, -1,-1.5,-3,-4,-4.5]
  fig = plt.figure(figsize=(12, 4), tight_layout=True)
  pvals = {
    1: pvals_ml[dataset][1],
    2: pvals_conv[dataset][1],
    4: pvals_ml[dataset][0],
    5: pvals_conv[dataset][0]
  }
  minsofar, maxsofar = 0, 0
  for i, b in enumerate([1,2,4,5]):
    rng = get_range(risk_reductions[buckets == b])
    minsofar = min(minsofar, np.min(rng))
    maxsofar = max(maxsofar, np.max(rng))
    plt.plot(rng, [ytick_locs[b]] * 3, "-|",
             color=plt_colors[0] if b == 1 or b == 4 else plt_colors[1])
    plt.plot([rng[1]], ytick_locs[b], "D",
             color=plt_colors[0] if b == 1 or b == 4 else plt_colors[1])
  for i, b in enumerate([1,2,4,5]):
    rng = get_range(risk_reductions[buckets == b])
    plt.text(maxsofar + 0.01, ytick_locs[b],
             "{:.4f} [{:.4f} {:.4f}], $P$ = {:.2f}, $N$ = {}".format(
               rng[1],rng[0], rng[2], pvals[b], lengths[i]))
  ax = fig.get_axes()[0]
  ax.set_yticks([-0, -1,-1.5,-3,-4,-4.5])
  r = ax.set_yticklabels(ytick_labels, ha = 'left')
  plt.draw()
  ax.tick_params(axis=u'y', which=u'both',length=0)
  yax = ax.get_yaxis()
  pad = max(T.label.get_window_extent().width + 5 for T in yax.majorTicks)
  yax.set_tick_params(pad=pad)
  plt.xlabel("Average Risk Reduction")
  plt.xlim([-0.07, 0.15])
  plt.ylim([-5.5,0.5])
  plt.axvline(x=0.0, linestyle="--", color="grey")
  plt.savefig("./img/{}_forest_plot.pdf".format(dataset))


def plot_expected_vs_obs_rr(dataset, n_bins=5, bin_strategy="rr"):
  plt.figure(figsize=(12, 4))
  plt.subplot(1,2,1)
  obs_rr = np.load("./results/machinelearning/%s/obs_rr_binned.npy" % dataset)
  pred_rr = np.load("./results/machinelearning/%s/pred_rr_binned.npy" % dataset)
  resid = np.array(obs_rr) - np.array(pred_rr)
  t, pval = stats.ttest_1samp(resid, popmean=0)
  slope, intercept = np.polyfit(pred_rr, obs_rr, 1)
  rss = np.sum((np.array(obs_rr) - np.array(pred_rr)) ** 2)
  plt.scatter(pred_rr, obs_rr, alpha=0.5, color=plt_colors[0])
  abline_values = [slope * i + intercept for i in [-0.15, 0.25]]
  plt.plot([-0.15, 0.25], abline_values, '--', color=plt_colors[0])
  plt.title("Machine Learning", fontsize=12, fontweight="bold")
  plt.xlim([-0.15,0.20])
  plt.ylim([-0.15,0.20])
  plt.text(-0.12, 0.15, "Slope: %.2f, Intercept: %.2f" % (slope, intercept))
  plt.xlabel("Predicted ARR")
  plt.ylabel("Observed ARR")
  plt.plot((-0.3,0.3), (-0.3, 0.3), "--", color="grey")
  plt.subplot(1,2,2)
  obs_rr = np.load("./results/conventional/%s/obs_rr_binned.npy" % dataset)
  pred_rr = np.load("./results/conventional/%s/pred_rr_binned.npy" % dataset)
  resid = np.array(obs_rr) - np.array(pred_rr)
  t, pval = stats.ttest_1samp(resid, popmean=0)
  slope, intercept = np.polyfit(pred_rr, obs_rr, 1)
  rss = np.sum((np.array(obs_rr) - np.array(pred_rr)) ** 2)
  plt.scatter(pred_rr, obs_rr, alpha=0.5, color=plt_colors[1])
  abline_values = [slope * i + intercept for i in [-0.15, 0.25]]
  plt.plot([-0.15, 0.25], abline_values, '--', color=plt_colors[1])
  plt.title("Conventional", fontsize=12, fontweight="bold")
  plt.xlim([-0.15,0.20])
  plt.ylim([-0.15,0.20])
  plt.xlabel("Predicted ARR")
  plt.ylabel("Observed ARR")
  plt.text(-0.12, 0.15, "Slope: %.2f, Intercept: %.2f" % (slope, intercept))
  plt.plot((-0.3,0.3), (-0.3, 0.3), "--", color="grey")
  plt.savefig("./img/{}_calibration_curve_by_pred_risk.pdf".format(dataset))


def plot_pred_rr_against_baseline_decile(dataset, n_bins=10):
  bins = np.percentile(baseline_risk[dataset], q=np.linspace(0, 100, n_bins + 1))
  deciles = np.linspace(0, 100, n_bins + 1)
  baseline_decile = deciles[np.digitize(baseline_risk[dataset], bins) - 1]
  baseline_decile[baseline_decile == 100] = 90
  plt.figure(figsize=(10, 4))
  plt.axhline(y=0.0, linestyle="--", color="grey")
  sns.boxplot(x=np.r_[baseline_decile, baseline_decile] / 10 + 1,
              y=np.r_[pred_rr_ml[dataset], pred_rr_conv[dataset]],
              hue=np.r_[["Machine Learning"] * len(pred_rr_ml[dataset]),
                        ["Conventional"] * len(pred_rr_conv[dataset])],
              palette=sns.color_palette("deep", 8),
              showfliers=False)
  plt.ylabel("Predicted ARR")
  plt.xlabel("Baseline Risk Decile")
  plt.ylim((-0.15, 0.20))
  plt.savefig("./img/{}_pred_rr_baseline_decile.pdf".format(dataset))


def plot_predicted_rr(dataset):
  plt.figure(figsize=(8, 4))
  pred_rr = pred_rr_ml[dataset]
  sns.kdeplot(pred_rr, label="Machine Learning", shade=True)
  pred_rr = pred_rr_conv[dataset]
  sns.kdeplot(pred_rr, label="Conventional", shade=True)
  plt.ylabel("Density")
  plt.xlabel("Predicted absolute risk reduction")
  plt.xlim([-0.15, 0.15])
  plt.legend()
  plt.savefig("./img/{}_pred_rr_distributions.pdf".format(dataset))


def calculate_summary_stats(dataset, bucket=True):
  cols = load_data("accord")["cols"]
  if bucket:
    print("== ml [BEN | NOEFF]")
    print(sum(pred_rr_ml[dataset] > 0))
    print(sum(pred_rr_ml[dataset] <= 0))
    for i, col in enumerate(cols):
      ben = X[dataset][:,i][pred_rr_ml[dataset] > 0]
      noeff = X[dataset][:,i][pred_rr_ml[dataset] <= 0]
      print("{}:,{:.2f} ({:.2f}),{:.2f} ({:.2f})".format(
          col, ben.mean(), ben.std(), noeff.mean(), noeff.std()))
    print("== conv [BEN | NOEFF]")
    print(sum(pred_rr_conv[dataset] > 0))
    print(sum(pred_rr_conv[dataset] <= 0))
    for i, col in enumerate(cols):
      ben = X[dataset][:,i][pred_rr_conv[dataset] > 0]
      noeff = X[dataset][:,i][pred_rr_conv[dataset] <= 0]
      print("{}:,{:.2f} ({:.2f}),{:.2f} ({:.2f})".format(
          col, ben.mean(), ben.std(), noeff.mean(), noeff.std()))
  else:
    for i, col in enumerate(cols):
      print("{}:,{:.2f} ({:.2f})".format(col, X[dataset][:,i].mean(),
                                              X[dataset][:,i].std()))


def plot_matching_patient_pairs(dataset):
  random.seed(1)
  plt.figure(figsize=(11, 4))
  plt.subplot(1,2,1)
  tuples = list(zip(pred_rr_ml[dataset], y[dataset], w[dataset]))
  untreated = list(filter(lambda t: t[2] == 0, tuples))
  treated = list(filter(lambda t: t[2] == 1, tuples))
  if len(treated) < len(untreated):
    untreated = random.sample(untreated, len(treated))
  if len(untreated) < len(treated):
    treated = random.sample(treated, len(untreated))
  assert len(untreated) == len(treated)
  untreated = sorted(untreated, key=lambda t: t[0])
  treated = sorted(treated, key=lambda t: t[0])
  plt.scatter(np.array(treated)[:,0], np.array(untreated)[:,0], marker=".",
              alpha=1e-2, color=plt_colors[0])
  plt.plot((-0.3, 0.3), (-0.3, 0.3), "--", color="grey")
  plt.xlabel("Predicted ARR, intensive arm")
  plt.ylabel("Predicted ARR, standard arm")
  plt.ylim(-0.3, 0.3)
  plt.xlim(-0.3, 0.3)
  plt.title("Machine Learning", fontsize=12, fontweight="bold")
  plt.subplot(1,2,2)
  tuples = list(zip(pred_rr_conv[dataset], y[dataset], w[dataset]))
  untreated = list(filter(lambda t: t[2] == 0, tuples))
  treated = list(filter(lambda t: t[2] == 1, tuples))
  if len(treated) < len(untreated):
    untreated = random.sample(untreated, len(treated))
  if len(untreated) < len(treated):
    treated = random.sample(treated, len(untreated))
  assert len(untreated) == len(treated)
  untreated = sorted(untreated, key=lambda t: t[0])
  treated = sorted(treated, key=lambda t: t[0])
  plt.scatter(np.array(treated)[:,0], np.array(untreated)[:,0], marker=".",
              alpha=1e-2, color=plt_colors[1])
  plt.plot((-0.3, 0.3), (-0.3, 0.3), "--", color="grey")
  plt.xlabel("Predicted ARR, intensive arm")
  plt.ylabel("Predicted ARR, standard arm")
  plt.title("Conventional", fontsize=12, fontweight="bold")
  plt.ylim(-0.3, 0.3)
  plt.xlim(-0.3, 0.3)
  plt.savefig("./img/{}_matching_patient_pairs.pdf".format(dataset))


def get_conventional_coeffs(dataset):

  cols = load_data("accord")["cols"]
  print("== Main terms:")
  for col, coeff in zip(cols, coeffs[dataset][:17]):
    print(col, coeff)
  print("== Interaction terms:")
  for col, coeff in zip(cols, coeffs[dataset][17:34]):
    print(col, coeff)

if __name__ == "__main__":

  parser = ArgumentParser()
  parser.add_argument("--dataset", type=str, default="combined")
  args = parser.parse_args()

  print("Correlation...")
  calc_corr_in_pred_rr(args.dataset)
  print("Forest plot...")
  forest_plot(args.dataset)
  print("Calibration...")
  plot_expected_vs_obs_rr(args.dataset)
  print("Baseline risk comparison...")
  plot_pred_rr_against_baseline_decile(args.dataset)
  print("Distributions of risk reduction...")
  plot_predicted_rr(args.dataset)
  print("Matching patient pairs...")
  plot_matching_patient_pairs(args.dataset)
  print("Summary statistics...")
  calculate_summary_stats(args.dataset, bucket=True)
