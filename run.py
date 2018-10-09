"""
Run
"""
from __future__ import division, print_function

import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from models import RFXLearner, CoxAIC, CausalForest, SurvRF, \
                   LogisticRegression, LinearXLearner
from dataloader import load_data, combine_datasets, cut_dataset_at_cens_time


def run_with_model(dataset, args):
    """
    Runsing a particular choice of model.
    """
    cut_data, all_data = cut_dataset_at_cens_time(dataset, args.cens_time)

    if args.model == "xlearner":
        model = RFXLearner()
        model.train(cut_data["X"], cut_data["w"], cut_data["y"],
                    cut_data["ipcw"])
        pred_rr = model.predict(all_data["X"], all_data["w"], False, True)

    elif args.model == "cox":
        model = CoxAIC()
        model.train(all_data["X"], all_data["w"], all_data["y"], all_data["t"])
        pred_rr = model.predict(args.cens_time, all_data["X"])

    elif args.model == "survrf":
        model = SurvRF()
        model.train(all_data["X"], all_data["w"], all_data["y"], all_data["t"])
        pred_rr = model.predict(args.cens_time)

    elif args.model == "causalforest":
        model = CausalForest()
        model.train(cut_data["X"], cut_data["w"], cut_data["y"])
        pred_rr = np.r_[model.predict(),
                        model.predict(all_data["X"][all_data["cens"] == 1])]

    elif args.model == "logreg":
        model = LogisticRegression()
        model.train(cut_data["X"], cut_data["w"], cut_data["y"],
                    cut_data["ipcw"])
        pred_rr = model.predict(all_data["X"])

    elif args.model == "linearxlearner":
        model = LinearXLearner()
        model.train(cut_data["X"], cut_data["w"], cut_data["y"],
                    cut_data["ipcw"])
        pred_rr = model.predict(all_data["X"], all_data["w"], False)

    else:
        raise ValueError("Not a supported model.")

    return {
        "pred_rr": pred_rr,
        "X": all_data["X"],
        "w": all_data["w"],
        "y": all_data["y"],
        "t": all_data["t"],
        "y_cut": all_data["y_cut"],
        "cens": all_data["cens"],
    }


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="xlearner")
    parser.add_argument("--dataset", type=str, default="combined")
    parser.add_argument("--validate-on", type=str, default="")
    parser.add_argument("--cens-time", type=float, default=365.25 * 3)
    args = parser.parse_args()

    print("=" * 79)
    print(f"== Running for: {args.dataset.upper()}")

    if args.dataset == "combined":
        dataset = combine_datasets(load_data("sprint"), load_data("accord"))
    else:
        dataset = load_data(args.dataset)

    stats = run_with_model(dataset, args)
    base_dir = f"results/{args.model}/{args.dataset}"
    Path(base_dir).mkdir(parents=True, exist_ok=True)

    for k, v in stats.items():
        np.save(base_dir + "/%s.npy" % k, v)

    print(f"== Saved to: {base_dir}")
