# -*- coding: utf-8 -*-
from __future__ import division, print_function

import numpy as np
import pandas as pd
import random
import scipy.stats as stats
import rpy2.robjects as robjects
from collections import defaultdict
from lifelines import KaplanMeierFitter
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri, r, sequence_to_vector
from sklearn.metrics import roc_auc_score


HARM_ASSIGNMENT = 10
NOEFF_ASSIGNMENT = 11
BENEFIT_ASSIGNMENT = 12

# load relevant R code
numpy2ri.activate()
cvAUC_pkg = importr("cvAUC")


def _get_assignments_to_use(two_way):
  """
  Helper function to track which assignments to use.
  """
  if two_way is True:
    assignments_to_use=[NOEFF_ASSIGNMENT, BENEFIT_ASSIGNMENT]
  else:
    assignments_to_use=[HARM_ASSIGNMENT, NOEFF_ASSIGNMENT, BENEFIT_ASSIGNMENT]
  return assignments_to_use

def assign_buckets_using_rr(pred_rr, threshold=0.00, two_way=True):
  """
  Assign buckets based on predicted risk reduction.
  """
  buckets = np.zeros_like(pred_rr)
  if two_way:
    buckets[pred_rr > threshold] = BENEFIT_ASSIGNMENT
    buckets[pred_rr <= threshold] = NOEFF_ASSIGNMENT
  else:
    buckets[pred_rr >= threshold] = BENEFIT_ASSIGNMENT
    buckets[pred_rr <= -threshold] = HARM_ASSIGNMENT
    buckets[np.logical_and(pred_rr > -threshold,
                           pred_rr < threshold)] = NOEFF_ASSIGNMENT
  return buckets

def get_range(scores):
  """
  Get the 2.5th percentile, mean, and 97.5th percentile.
  """
  lower = np.percentile(scores, 2.5)
  mean = np.mean(scores)
  upper = np.percentile(scores, 97.5)
  return lower, mean, upper

def get_bootstrap_examples(bucket_assignments, w, y, p=None):
  """
  Take a bootstaped sample.
  """
  num_samples = len(bucket_assignments)
  sample_indices = np.random.choice(num_samples, num_samples)

  a_s = np.array(bucket_assignments)[sample_indices]
  w_s = np.array(w)[sample_indices]
  y_s = np.array(y)[sample_indices]

  p_s = None
  if p is not None:
    p_s = np.array(p)[sample_indices]

  return a_s, w_s, y_s, p_s

def risk_reduction_directionality_check(bucket_assignments, w, y,
                                        num_trials=10000, two_way=True):
  """
  Get bootstrapped confidence intervals of the true risk reduction in the
  assigned buckets.
  """
  arrs = defaultdict(list)
  assignments_to_use = _get_assignments_to_use(two_way)

  for trial in range(num_trials):
    a_s, w_s, y_s, _ = get_bootstrap_examples(bucket_assignments, w, y)
    for assignment in assignments_to_use:
      bkt_indices = a_s == assignment
      w_zero_indices, w_one_indices = w_s == 0, w_s == 1
      untreated_indices = np.logical_and(bkt_indices, w_zero_indices)
      treated_indices = np.logical_and(bkt_indices, w_one_indices)
      arr = absolute_rr(y_s[untreated_indices], y_s[treated_indices])
      if not np.isnan(arr):
        arrs[assignment].append(arr)

  confidence_intervals = {}
  for bucket in arrs:
    low, mean, high = get_range(arrs[bucket])
    confidence_intervals[bucket] = (low, mean, high)

  return arrs, confidence_intervals

def compute_risk_with_outcomes(outcomes):
  """Get the probability of a bad outcome"""
  return np.sum(outcomes) / len(outcomes)

def absolute_rr(untreated_outcomes, treated_outcomes):
  """Compute the absolute risk reduction."""
  assert(np.all([
    np.array_equal(a, np.array(a).astype(bool))
    for a in [untreated_outcomes, treated_outcomes]])) # assert binary
  risk_u = compute_risk_with_outcomes(untreated_outcomes)
  risk_t = compute_risk_with_outcomes(treated_outcomes)
  arr = risk_u - risk_t
  return arr

def wald_test(buckets, w, y, two_way=True):
  """
  Return p-values for the survival rate in each of the buckets.
  """
  p_values = []
  for assignment in _get_assignments_to_use(two_way):
    y_in_bucket = y[buckets == assignment]
    w_in_bucket = w[buckets == assignment]
    control = y_in_bucket[w_in_bucket == 0]
    treatment = y_in_bucket[w_in_bucket == 1]
    table = np.array([[np.sum(control), len(control) - np.sum(control)],
                      [np.sum(treatment), len(treatment) - np.sum(treatment)]])
    if np.any(table == 0):
      p_values.append(1)
      continue
    _, p, _, _ = stats.chi2_contingency(table)
    p_values.append(p)
  return p_values

def evaluate_buckets(bucket_assignments, W, y, num_trials=10000, two_way=True):
  """
  Do evaluation of model bucketing.

  All arguments are expected to be arrays of the same length n,
  with bucket assignments assigned to each patients, where bucket_assignment values
  are coded as constants in this file. In the same format, W (treatment arm),
  y (the outcome), and p (the model's probability for p(Y | X, W)).
  """
  print('------Risk Reduction Harm / Benefit / No Effect Check-----')
  arrs, cis = risk_reduction_directionality_check(bucket_assignments, W, y,
                                                  num_trials=num_trials,
                                                  two_way=two_way)
  if not two_way:
    print("Harm Test: [{:.4f} {:.4f} {:.4f}]".format(*cis[HARM_ASSIGNMENT]))
  print('No effect Test: [{:.4f} {:.4f} {:.4f}]'.format(*cis[NOEFF_ASSIGNMENT]))
  print('Benefit Test: [{:.4f} {:.4f} {:.4f}]'.format(*cis[BENEFIT_ASSIGNMENT]))

  print('------Wald test-----')
  p_values = wald_test(bucket_assignments, W, y, two_way)
  if not two_way:
    print("P-values (h, n, b): [{:.4f} {:.4f} {:.4f}]".format(*p_values))
  else:
    print("P-values (n, b): [{:.4f} {:.4f}]".format(*p_values))

  return arrs, p_values

def decision_value_rmst(pred_rr, y, w, t, cens_time):
  """
  Return the decision value RMST.
  """
  treat = np.logical_and(pred_rr > 0, w == 1)
  control = np.logical_and(pred_rr <= 0, w == 0)
  if np.sum(control) == 0:
    kmf_treat = KaplanMeierFitter()
    kmf_treat.fit(t[treat], y[treat])
    surv = kmf_treat.survival_function_.reset_index()
    idx = np.searchsorted(surv["timeline"], v=cens_time)[0]
    rmst_1 = np.trapz(y=surv["KM_estimate"][:idx], x=surv["timeline"][:idx])
    return rmst_1
  if np.sum(treat) == 0:
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
    for j in range(i+1, len(pairs)):
      if obs_benefit[i] != obs_benefit[j]:
        if (obs_benefit[i] < obs_benefit[j] and
            pred_benefit[i] < pred_benefit[j]) or \
           (obs_benefit[i] > obs_benefit[j] and
            pred_benefit[i] > pred_benefit[j]):
          count += 1
        total += 1

  return count / total


def value_rmst(preds, y, w, t, cens_time):
  """
  Calculate the value of RMST at the censored time.
  """


def auroc(p, y, folds=None):
  """
  Returns the discrimination scores (AUC) for predicted outcomes of patients
  with and without treatment.
  """
  if folds is None:
    return roc_auc_score(y, p)
  else:
    result = cvAUC_pkg.ci_cvAUC(p[:,np.newaxis], y[:,np.newaxis],
                                folds=sequence_to_vector(folds),
                                confidence=0.95)
    roc = result[0][0]
    lower, upper = result[2]
  return roc, lower, upper

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
    with_rx = 1-surv["KM_estimate"][min(idx, len(surv) - 1)]
    surv = kmf_no_rx.survival_function_.reset_index()
    idx = np.searchsorted(surv["timeline"], v=cens_time)[0]
    no_rx = 1-surv["KM_estimate"][min(idx, len(surv) - 1)]
    obs_rr.append(no_rx - with_rx)
  pred_rr, obs_rr = np.array(pred_rr), np.array(obs_rr)
  slope, intercept = np.polyfit(pred_rr, obs_rr, 1)
  rss = np.sum((obs_rr - pred_rr) ** 2)
  resid = obs_rr - pred_rr
  t, pval = stats.ttest_1samp(resid, popmean=0)
  return rss, pval, slope, intercept, pred_rr, obs_rr

def hosmer_lemeshow(p, y, bins=10):
  """
  Returns the calibration scores (Hosmer-Lemeshow test) for predicted outcomes.

  Adapted [with a typo fixed] from:
  https://www.data-essential.com/hosman-lemeshow-in-python/
  """
  df = pd.DataFrame({"score": p, "target": y})
  df["score_decile"] = pd.qcut(df["score"], bins, duplicates="drop")
  obsevents_pos = df["target"].groupby(df.score_decile).sum()
  obsevents_neg = df["target"].groupby(df.score_decile).count() - obsevents_pos
  expevents_pos = df["score"].groupby(df.score_decile).sum()
  expevents_neg = df["score"].groupby(df.score_decile).count() - expevents_pos
  hl = (((obsevents_pos - expevents_pos)**2/expevents_pos) +
        ((obsevents_neg - expevents_neg)**2/expevents_neg)).sum()
  return hl, stats.chi2.sf(hl,len(obsevents_pos) - 2)

def gnd_test(p, y, bins=10):
  """
  Returns: (chi2, p-value, slope, intercept)
  """
  df = pd.DataFrame({"score": p, "target": y})
  df["score_decile"] = pd.qcut(df["score"], bins,
                               labels=range(bins),
                               duplicates="raise")
  robjects.globalenv["pred"] = p[:,np.newaxis]
  robjects.globalenv["out"] = y[:,np.newaxis]
  robjects.globalenv["tvar"] = np.zeros((len(p), 1))
  robjects.globalenv["cens.t"] = np.zeros((len(p), 1))
  robjects.globalenv["groups"] = df["score_decile"].astype(int)
  robjects.globalenv["adm.cens"] = 1
  gnd_result = r("GND.calib(pred, tvar, out, cens.t, groups, adm.cens)")
  return gnd_result[1:]

