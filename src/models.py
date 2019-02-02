from __future__ import division

import numpy as np
import pandas as pd
import warnings
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri, pandas2ri, r
from rpy2.rinterface import RRuntimeWarning


numpy2ri.activate()
pandas2ri.activate()
warnings.filterwarnings("ignore", category=RRuntimeWarning)


def _get_interaction_terms(X, w):
    return np.hstack((X, w[:, np.newaxis] * X))

def _add_treatment_feature(X, w):
    return np.hstack((X, w[:, np.newaxis]))

def _as_df(X):
    data = {"x" + str(i): X[:, i] for i in range(X.shape[1])}
    return pd.DataFrame(data)

def _setup_r_environment(X, w=None, y=None, t=None, ipcw=None):
    ro.globalenv["X"] = _as_df(X)
    if w is not None:
        ro.globalenv["w"] = ro.FloatVector(w)
    if y is not None:
        ro.globalenv["y"] = ro.FloatVector(y)
    if t is not None:
        ro.globalenv["t"] = ro.FloatVector(t)
    if ipcw is not None:
        ro.globalenv["ipcw"] = ro.FloatVector(ipcw)


class RFXLearner(object):
    def __init__(self):
        ro.r("rm(list=ls())")
        r.source("lib/xlearner-rf.R")

    def train(self, X, w, y, ipcw):
        _setup_r_environment(X, w, y, ipcw=ipcw)
        ro.r("models = train_x_learner(X, w, y, ipcw)")

    def predict(self, X, w, estimate_propensities=True, oob_predictions=True):
        _setup_r_environment(X, w)
        ro.globalenv["est_prop"] = estimate_propensities
        ro.globalenv["oob_pred"] = oob_predictions
        ro.r("attach(models)")
        self.pred_rr = ro.r("predict_x_learner(X, w, est_prop, oob_pred)")
        ro.r("detach(models)")
        return -self.pred_rr


class LinearXLearner(object):
    def __init__(self):
        ro.r("rm(list=ls())")
        r.source("lib/xlearner-linear.R")

    def train(self, X, w, y, ipcw):
        _setup_r_environment(X, w, y, ipcw=ipcw)
        ro.r("models = train_x_learner(X, w, y, ipcw)")

    def predict(self, X, w, estimate_propensities=True):
        _setup_r_environment(X, w)
        ro.globalenv["est_prop"] = estimate_propensities
        ro.r("attach(models)")
        self.pred_rr = ro.r("predict_x_learner(X, w, est_prop)")
        ro.r("detach(models)")
        return -self.pred_rr


class LogisticRegression(object):
    def train(self, X, w, y, ipcw):
        X = _get_interaction_terms(X, w)
        X = _add_treatment_feature(X, w)
        _setup_r_environment(X, y=y, ipcw=ipcw)
        model = ro.r("glm(data = X, y ~ ., weights=ipcw)")
        ro.globalenv["model"] = model

    def predict(self, X):
        X_1 = _get_interaction_terms(X, np.ones(len(X)))
        X_1 = _add_treatment_feature(X_1, np.ones(len(X)))
        _setup_r_environment(X_1)
        py1 = ro.r("predict(model, newdata = X, type='response')")
        X_0 = _get_interaction_terms(X, np.zeros(len(X)))
        X_0 = _add_treatment_feature(X_0, np.zeros(len(X)))
        _setup_r_environment(X_0)
        py0 = ro.r("predict(model, newdata = X, type='response')")
        return py0 - py1


class CoxAIC(object):
    def __init__(self):
        self.mass_lib = importr("MASS")
        self.surv_lib = importr("survival")

    def train(self, X, w, y, t):
        X = _get_interaction_terms(X, w)
        X = _add_treatment_feature(X, w)
        _setup_r_environment(X, w, y, t)
        model = ro.r("coxph(data = X, Surv(t, y) ~ .)")
        self.clf = self.mass_lib.stepAIC(model, direction="backward",
                                         trace=False)

    def predict(self, cens_time, newdata):
        w = np.ones(len(newdata))
        X = _get_interaction_terms(newdata, w)
        X = _add_treatment_feature(X, w)
        df = _as_df(X)
        survfit_model = self.surv_lib.survfit(self.clf, newdata=df,
                                              se_fit=False)
        survfit_matrix = np.asarray(survfit_model[5])
        times = np.asarray(survfit_model[1])
        idx = min(np.searchsorted(times, cens_time), len(times) - 1)
        py1 = 1 - survfit_matrix[idx, :]
        w = np.zeros(len(newdata))
        X = _get_interaction_terms(newdata, w)
        X = _add_treatment_feature(X, w)
        df = _as_df(X)
        survfit_model = self.surv_lib.survfit(self.clf, newdata=df,
                                              se_fit=False)
        survfit_matrix = np.asarray(survfit_model[5])
        times = np.asarray(survfit_model[1])
        idx = min(np.searchsorted(times, cens_time), len(times) - 1)
        py0 = 1 - survfit_matrix[idx, :]
        return py0 - py1


class CausalForest(object):
    def __init__(self, n_trees=1000, seed=123):
        self.grf = importr("grf")
        self.n_trees = n_trees
        self.seed = seed

    def train(self, X, w, y):
        self.X, self.w = X, w
        X = r.matrix(X, nrow=X.shape[0])
        w = ro.FloatVector(w)
        y = ro.FloatVector(y)
        self.rr_model = self.grf.causal_forest(X, y, w, **{
            "seed": self.seed,
            "num.trees": self.n_trees,
            "honesty": True,
            "alpha": 0.1,
            "min.node.size": 1})

    def predict(self, X=None):
        if X is None:
            tau_hat = self.grf.predict_causal_forest(self.rr_model, **{
                "estimate.variance": False, })
        else:
            tau_hat = self.grf.predict_causal_forest(self.rr_model, X, **{
                "estimate.variance": False, })
        tau_hat = np.array(tau_hat[0]).flatten()
        return -tau_hat


class SurvRF(object):
    def __init__(self, n_trees=100, seed=123):
        self.rfsrc = importr("randomForestSRC")
        self.n_trees = n_trees
        self.seed = seed

    def train(self, X, w, y, t):
        self.X = X
        self.w = w
        self.y = y
        self.t = t
        X = _get_interaction_terms(self.X, w)
        data = _as_df(_add_treatment_feature(X, w))
        data["t"] = t
        data["s"] = y
        ro.globalenv["data"] = data
        r("model = rfsrc(Surv(t, s) ~ ., data = data, ntree = %d, seed = %d)" %
          (self.n_trees, self.seed))

    def predict(self, cens_time):
        times = r("model$time.interest")
        cens_idx = min(np.searchsorted(times, cens_time), len(times) - 1)
        w = np.ones_like(self.w)
        X = _get_interaction_terms(self.X, w)
        df = _as_df(_add_treatment_feature(X, w))
        df["t"] = self.t
        df["s"] = self.y
        ro.globalenv["test.data"] = df
        r("preds = predict(model, newdata = test.data, outcome = 'test')")
        py1 = r("preds$survival.oob")
        py1_final = 1 - py1[:, cens_idx]
        w = np.zeros_like(self.w)
        X = _get_interaction_terms(self.X, w)
        df = _as_df(_add_treatment_feature(X, w))
        df["t"] = self.t
        df["s"] = self.y
        ro.globalenv["test.data"] = df
        r("preds = predict(model, newdata = test.data, outcome = 'test')")
        py0 = r("preds$survival.oob")
        py0_final = 1 - py0[:, cens_idx]
        return py0_final - py1_final
