import numpy as np
from argparse import ArgumentParser
from collections import defaultdict
from models import CoxAIC,LogisticRegression, LinearXLearner
from dataloader import load_data, combine_datasets, bootstrap_dataset, \
                       cut_dataset
from evaluate import c_statistic, decision_value_rmst
from tqdm import tqdm


def run_for_optimism(original_dataset, bootstrap_dataset, args):
    """
    Calculates difference between performance on a bootstrapped dataset (upon
    which the model is trained) and the original dataset. Optimism is defined
    as the mean difference over many bootstrap datasets.
    """
    cut_data, all_data = cut_dataset(bootstrap_dataset, args.cens_time)
    cut_data_orig, all_data_orig = cut_dataset(original_dataset, args.cens_time)

    if args.model == "cox":
        model = CoxAIC()
        model.train(all_data["X"], all_data["w"], all_data["y"], all_data["t"])
        pred_rr = model.predict(args.cens_time, all_data["X"])
        pred_rr_orig = model.predict(args.cens_time, all_data_orig["X"])

    elif args.model == "logreg":
        model = LogisticRegression()
        model.train(cut_data["X"], cut_data["w"], cut_data["y"],
                    cut_data["ipcw"])
        pred_rr = model.predict(all_data["X"])
        pred_rr_orig = model.predict(all_data_orig["X"])

    elif args.model == "linearxlearner":
        model = LinearXLearner()
        model.train(cut_data["X"], cut_data["w"], cut_data["y"],
                    cut_data["ipcw"])
        pred_rr = model.predict(all_data["X"], all_data["w"], False)
        pred_rr_orig = model.predict(all_data_orig["X"],
                                     all_data_orig["w"], False)

    else:
        raise ValueError("Not a supported model.")

    c_stat_bootstrap = c_statistic(pred_rr[all_data["cens"] == 0],
                                   cut_data["y"], cut_data["w"])
    c_stat_original = c_statistic(pred_rr_orig[all_data_orig["cens"] == 0],
                                  cut_data_orig["y"], cut_data_orig["w"])

    rmst_bootstrap = decision_value_rmst(pred_rr, all_data["y"], all_data["w"],
                                         all_data["t"], args.cens_time)
    rmst_original = decision_value_rmst(pred_rr_orig, all_data_orig["y"],
                                        all_data_orig["w"], all_data_orig["t"],
                                        args.cens_time)

    return {
        "c_stat_diff": c_stat_bootstrap - c_stat_original,
        "decision_value_rmst_diff": rmst_bootstrap - rmst_original
    }


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="cox")
    parser.add_argument("--dataset", type=str, default="combined")
    parser.add_argument("--validate-on", type=str, default="")
    parser.add_argument("--bootstrap-samples", type=int, default=250)
    parser.add_argument("--cens-time", type=float, default=365.25 * 3)
    args = parser.parse_args()

    print("=" * 79)
    print(f"== Running for: {args.dataset.upper()}")

    if args.dataset == "combined":
        dataset = combine_datasets(load_data("sprint"), load_data("accord"))
    else:
        dataset = load_data(args.dataset)

    stats = defaultdict(list)
    for _ in tqdm(range(args.bootstrap_samples)):
        idxs = bootstrap_dataset(dataset)
        bootstrap = {"X": dataset["X"][idxs],
                     "y": dataset["y"][idxs],
                     "w": dataset["w"][idxs],
                     "t": dataset["t"][idxs]}
        for k, v in run_for_optimism(dataset, bootstrap, args).items():
            stats[k].append(v)

    for k, v in stats.items():
        print(f"{k}: {np.mean(v)}")

