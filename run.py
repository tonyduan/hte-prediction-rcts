# -*- coding: utf-8 -*-
from __future__ import division, print_function

import gc
import os
from argparse import ArgumentParser
from tqdm import tqdm
from pathlib import Path
from models import *
from dataloader import *
from evaluate import *
from sklearn.preprocessing import scale


def run_for_causal_forest(dataset, args, bootstrap_id=None):

  gc.collect() # need to clear for bootstrap runs
  train_data, val_data = cut_dataset_at_cens_time(dataset, args.cens_time)

  seed = bootstrap_id if bootstrap_id is not None else 123
  model = CausalForest(n_trees=1000, seed=seed)
  model.train(train_data["X"], train_data["w"], train_data["y"])
  pred_rr_train = model.predict()

  rss, tpval, slope, intercept, pred_rr_binned, obs_rr_binned = calibration(
    pred_rr_train, train_data["y"], train_data["w"], train_data["t"], args.cens_time)

  c_stat = c_statistic(pred_rr_train, train_data["y"], train_data["w"])

  rmst = decision_value_rmst(pred_rr_train, train_data["y"], train_data["w"],
                             train_data["t"], args.cens_time)

  if bootstrap_id is None:

    print("Concordance-for-benefit: %.4f" % c_stat)
    print("Decision Value RMST: %.4f" % rmst)
    print("Calibration (rss, t-test p-value, slope, intercept): [%.4f %.4f %.4f %.4f]" %
          (rss, tpval, slope, intercept))
    buckets = assign_buckets_using_rr(pred_rr_train, threshold=0.0)
    arrs, p_values = evaluate_buckets(buckets, train_data["w"], train_data["y"])

    return {
      "pred_rr": pred_rr_train,
      "c_stat": c_stat,
      "slope": slope,
      "rmst": rmst,
      "intercept": intercept,
      "obs_rr_binned": obs_rr_binned,
      "pred_rr_binned": pred_rr_binned,
      "arrs": arrs,
      "pvals": p_values,
      "buckets": buckets,
      "X": val_data["X"],
      "w": val_data["w"],
      "y": val_data["y"],
      "t": val_data["t"],
    }

  else:

    return {
      "pred_rr": pred_rr_train,
      "c_stat": c_stat,
      "slope": slope,
      "rmst": rmst,
      "intercept": intercept,
      "obs_rr_binned": obs_rr_binned,
      "pred_rr_binned": pred_rr_binned,
    }


def run_for_surv_rf(dataset, args, bootstrap_id=None):

  gc.collect() # need to clear for bootstrap runs
  train_data, val_data = cut_dataset_at_cens_time(dataset, args.cens_time)

  seed = bootstrap_id if bootstrap_id is not None else 123
  model = SurvRF(n_trees=100, seed=seed)
  model.train(train_data["X"], train_data["w"], train_data["y"], train_data["t"])
  pred_rr_train = model.predict(args.cens_time)

  rss, tpval, slope, intercept, pred_rr_binned, obs_rr_binned = calibration(
    pred_rr_train, train_data["y"], train_data["w"], train_data["t"], args.cens_time)

  c_stat = c_statistic(pred_rr_train, train_data["y"], train_data["w"])

  rmst = decision_value_rmst(pred_rr_train, train_data["y"], train_data["w"],
                             train_data["t"], args.cens_time)

  if bootstrap_id is None:

    print("Concordance-for-benefit: %.4f" % c_stat)
    print("Decision Value RMST: %.4f" % rmst)
    print("Calibration (rss, t-test p-value, slope, intercept): [%.4f %.4f %.4f %.4f]" %
          (rss, tpval, slope, intercept))
    buckets = assign_buckets_using_rr(pred_rr_train, threshold=0.0)
    arrs, p_values = evaluate_buckets(buckets, train_data["w"], train_data["y"])

    return {
      "pred_rr": pred_rr_train,
      "c_stat": c_stat,
      "slope": slope,
      "rmst": rmst,
      "intercept": intercept,
      "obs_rr_binned": obs_rr_binned,
      "pred_rr_binned": pred_rr_binned,
      "arrs": arrs,
      "pvals": p_values,
      "buckets": buckets,
      "X": val_data["X"],
      "w": val_data["w"],
      "y": val_data["y"],
      "t": val_data["t"],
    }

  else:

    return {
      "pred_rr": pred_rr_train,
      "c_stat": c_stat,
      "slope": slope,
      "rmst": rmst,
      "intercept": intercept,
      "obs_rr_binned": obs_rr_binned,
      "pred_rr_binned": pred_rr_binned,
    }


def run_for_x_learner(dataset, args, bootstrap_id=None):

  gc.collect() # need to clear for bootstrap runs
  train_data, val_data = cut_dataset_at_cens_time(dataset, args.cens_time)

  model = RFXLearner()
  model.train(train_data["X"], train_data["w"], train_data["y"])

  if args.validate_on:
    dataset = load_data(args.validate_on)
    train_data, val_data = cut_dataset_at_cens_time(dataset, args.cens_time)
    pred_rr_val = model.predict(val_data["X"], val_data["w"], True, False)
    pred_rr_train = model.predict(train_data["X"], train_data["w"], True, False)
  else:
    pred_rr_val = model.predict(val_data["X"], val_data["w"], True, True)
    pred_rr_train = model.predict(train_data["X"], train_data["w"], True, True)

  rss, tpval, slope, intercept, pred_rr_binned, obs_rr_binned = calibration(
    pred_rr_val, val_data["y"], val_data["w"], val_data["t"], args.cens_time)

  c_stat = c_statistic(pred_rr_train, train_data["y"], train_data["w"])

  rmst = decision_value_rmst(pred_rr_val, val_data["y"], val_data["w"],
                                   val_data["t"], args.cens_time)

  if bootstrap_id is None:

    print("Concordance-for-benefit: %.4f" % c_stat)
    print("Decision Value RMST: %.4f" % rmst)
    print("Calibration (rss, t-test p-value, slope, intercept): [%.4f %.4f %.4f %.4f]" %
          (rss, tpval, slope, intercept))
    buckets = assign_buckets_using_rr(pred_rr_train, threshold=0.0)
    arrs, p_values = evaluate_buckets(buckets, train_data["w"], train_data["y"])

    return {
      "pred_rr": pred_rr_val,
      "c_stat": c_stat,
      "slope": slope,
      "rmst": rmst,
      "intercept": intercept,
      "obs_rr_binned": obs_rr_binned,
      "pred_rr_binned": pred_rr_binned,
      "arrs": arrs,
      "pvals": p_values,
      "buckets": buckets,
      "X": val_data["X"],
      "w": val_data["w"],
      "y": val_data["y"],
      "t": val_data["t"],
    }

  else:

    return {
      "pred_rr": pred_rr_val,
      "c_stat": c_stat,
      "slope": slope,
      "rmst": rmst,
      "intercept": intercept,
      "obs_rr_binned": obs_rr_binned,
      "pred_rr_binned": pred_rr_binned,
    }


def run_for_cox(dataset, args, bootstrap_id):

  gc.collect() # need to clear for bootstrap runs
  model = CoxAIC()

  if bootstrap_id is not None and args.validate_on == "":

    np.random.seed(bootstrap_id)
    bootstrapped_dataset = bootstrap_dataset(dataset)

    bin_data, all_data = cut_dataset_at_cens_time(dataset, args.cens_time)
    bootstrap_bin_data, bootstrap_all_data = cut_dataset_at_cens_time(bootstrapped_dataset, args.cens_time)

    model.train(bootstrap_all_data["X"], bootstrap_all_data["w"],
                bootstrap_all_data["y"], bootstrap_all_data["t"])

    pred_rr_bootstrap_all = model.predict(cens_time=args.cens_time, newdata=bootstrap_all_data["X"])
    pred_rr_bootstrap_bin = model.predict(cens_time=args.cens_time, newdata=bootstrap_bin_data["X"])
    pred_rr_original_all = model.predict(cens_time=args.cens_time, newdata=all_data["X"])
    pred_rr_original_bin = model.predict(cens_time=args.cens_time, newdata=bin_data["X"])

    c_stat = c_statistic(pred_rr_bootstrap_bin, bootstrap_bin_data["y"], bootstrap_bin_data["w"])
    c_stat_optimism = c_stat - c_statistic(pred_rr_original_bin, bin_data["y"], bin_data["w"])

    rmst = decision_value_rmst(pred_rr_bootstrap_all, bootstrap_all_data["y"], bootstrap_all_data["w"],
                               bootstrap_all_data["t"], args.cens_time)
    rmst_optimism = rmst - decision_value_rmst(pred_rr_original_all, all_data["y"], all_data["w"],
                                               all_data["t"], args.cens_time)

    rss, tpval, slope, intercept, pred_rr_binned, obs_rr_binned = calibration(
      pred_rr_bootstrap_all, bootstrap_all_data["y"], bootstrap_all_data["w"], bootstrap_all_data["t"], args.cens_time)

    return {
      "pred_rr": pred_rr_bootstrap_all,
      "c_stat": c_stat,
      "slope": slope,
      "rmst": rmst,
      "intercept": intercept,
      "obs_rr_binned": obs_rr_binned,
      "pred_rr_binned": pred_rr_binned,
      "c_stat_optimism": c_stat_optimism,
      "rmst_optimism": rmst_optimism,
    }

  else:

    bin_data, all_data = cut_dataset_at_cens_time(dataset, args.cens_time)

    model.train(all_data["X"], all_data["w"], all_data["y"], all_data["t"])
    if args.validate_on:
      dataset = load_data(args.validate_on)
      bin_data, all_data = cut_dataset_at_cens_time(dataset, args.cens_time)

    pred_rr_bin = model.predict(cens_time=args.cens_time, newdata=bin_data["X"])
    pred_rr_all = model.predict(cens_time=args.cens_time, newdata=all_data["X"])

    rss, tpval, slope, intercept, pred_rr_binned, obs_rr_binned = calibration(
      pred_rr_all, all_data["y"], all_data["w"], all_data["t"], args.cens_time)

    c_stat = c_statistic(pred_rr_bin, bin_data["y"], bin_data["w"])

    rmst = decision_value_rmst(
      pred_rr_all, all_data["y"], all_data["w"], all_data["t"], args.cens_time)

    if bootstrap_id is None:

      print("Concordance-for-benefit: %.4f" % c_stat)
      print("Decision Value RMST: %.4f" % rmst)
      print("Calibration (rss, t-test p-value, slope, intercept): [%.4f %.4f %.4f %.4f]" %
            (rss, tpval, slope, intercept))
      buckets = assign_buckets_using_rr(pred_rr_bin, threshold=0.0)
      arrs, p_values = evaluate_buckets(buckets, bin_data["w"], bin_data["y"])

      # extract baseline risk to compare
      baseline_model = CoxAICBaseline()
      baseline_model.train(all_data["X"], all_data["y"], all_data["t"])
      baseline_risk = baseline_model.predict(args.cens_time, all_data["X"])

      # extract HTE coefficients
      coeffs = np.zeros(2 * all_data["X"].shape[1] + 1)
      results = dict(zip(model.clf.names, list(model.clf)))
      used_vars = filter(lambda s: s[0] == "x", str(results["formula"]).split("<")[0].split())
      used_vars = [int(c[1:]) for c in used_vars if len(c) > 1]
      for k, v in zip(used_vars, model.clf[0]):
        coeffs[k] = v

      return {
        "pred_rr": pred_rr_all,
        "c_stat": c_stat,
        "slope": slope,
        "rmst": rmst,
        "intercept": intercept,
        "obs_rr_binned": obs_rr_binned,
        "pred_rr_binned": pred_rr_binned,
        "baseline_risk": baseline_risk,
        "coeffs": coeffs,
        "arrs": arrs,
        "pvals": p_values,
        "buckets": buckets,
        "X": all_data["X"],
        "w": all_data["w"],
        "y": all_data["y"],
        "t": all_data["t"],
      }

    else:

      return {
        "pred_rr": pred_rr_all,
        "c_stat": c_stat,
        "slope": slope,
        "rmst": rmst,
        "intercept": intercept,
        "obs_rr_binned": obs_rr_binned,
        "pred_rr_binned": pred_rr_binned,
      }

def run_for_dataset(dataset_name, args, bootstrap_id=None):

  if dataset_name == "combined":
    sprint = load_data("sprint")
    accord = load_data("accord")
    dataset = combine_datasets(sprint, accord)
  else:
    dataset = load_data(dataset_name)

  if args.scale:
    dataset["X"] = scale(dataset["X"], axis=0)

  if args.model == "xlearner":
    stats = run_for_x_learner(dataset, args, bootstrap_id)
    experiment_type = "machinelearning"
  elif args.model == "coxph":
    stats = run_for_cox(dataset, args, bootstrap_id)
    experiment_type = "conventional"
  elif args.model == "causalforest":
    stats = run_for_causal_forest(dataset, args, bootstrap_id)
    experiment_type = "causalforest"
  elif args.model == "survrf":
    stats = run_for_surv_rf(dataset, args, bootstrap_id)
    experiment_type = "survrf"
  else:
    raise ValueError("Not a supported model.")

  if bootstrap_id is None:
    base_dir = "results/{}/{}".format(experiment_type, dataset_name)
  else:
    base_dir = "results/{}/{}/{}".format(experiment_type, dataset_name, bootstrap_id)
  if not os.path.exists(base_dir):
    Path(base_dir).mkdir(parents=True, exist_ok=True)

  for k, v in stats.items():
    np.save(base_dir + "/%s.npy" % k, v)


if __name__ == "__main__":

  parser = ArgumentParser()
  parser.add_argument("--model", type=str, default="xlearner")
  parser.add_argument("--dataset", type=str, default="sprint")
  parser.add_argument("--validate-on", type=str, default="")
  parser.add_argument("--bootstrap", action="store_true")
  parser.add_argument("--scale", action="store_true")
  parser.add_argument("--cens-time", type=float, default=365.25 * 3)
  args = parser.parse_args()

  print("=" * 79)
  print("== Running for:", args.dataset.upper())

  if not args.bootstrap:
    run_for_dataset(args.dataset, args)

  else:
    if args.model == "xlearner":
      experiment_type = "machinelearning"
    elif args.model == "coxph":
      experiment_type = "conventional"
    else:
      experiment_type = args.model

    print("Running bootstrap samples...")
    for i in tqdm(range(1, 250 + 1)):
      run_for_dataset(args.dataset, args, i)

    print("Evaluating empirical confidence intervals...")
    c_stats, rmsts, slopes, intercepts = [], [], [], []
    for i in tqdm(range(1, 250 + 1)):
      base_dir = "results/{}/{}/{}".format(experiment_type, args.dataset, i)
      c_stats.append(np.load(base_dir + "/c_stat.npy").item())
      rmsts.append(np.load(base_dir + "/rmst.npy").item())
      slopes.append(np.load(base_dir + "/slope.npy").item())
      intercepts.append(np.load(base_dir + "/intercept.npy").item())

    if args.model == "coxph" and args.validate_on == "":
      c_stat_optimisms, rmst_optimisms = [], []
      for i in tqdm(range(1, 250 + 1)):
        base_dir = "results/{}/{}/{}".format(experiment_type, args.dataset, i)
        c_stat_optimisms.append(np.load(base_dir + "/c_stat_optimism.npy"))
        rmst_optimisms.append(np.load(base_dir + "/rmst_optimism.npy"))
      print("C-statistic optimism: {:.4f}".format(np.mean(c_stat_optimisms)))
      print("RMST optimism: {:.4f}".format(np.mean(rmst_optimisms)))

    print("C-statistic: [{:.4f} {:.4f}]".format(np.percentile(c_stats, 2.5),
                                                np.percentile(c_stats, 97.5)))
    print("RMST: [{:.4f} {:.4f}]".format(np.percentile(rmsts, 2.5),
                                         np.percentile(rmsts, 97.5)))
    print("Slope: [{:.4f} {:.4f}]".format(np.percentile(slopes, 2.5),
                                          np.percentile(slopes, 97.5)))
    print("Intercept: [{:.4f} {:.4f}]".format(np.percentile(intercepts, 2.5),
                                              np.percentile(intercepts, 97.5)))


