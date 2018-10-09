### Predicting Individual Patient Treatment Effects from Randomized Trial Data

Last update: October 2018.

---

#### X-learner with random forests

R code for the X-learner [1] with random forest base learners can be found in
`lib/xlearner-rf.R`. By default, it makes out-of-bag predictions, though 
this can be modified by changing the `predict_oob` flag. 

#### Evaluation

Our evaluation code lies in `evaluate.py`, with Python implementations of:

1. C-statistic-for-benefit [2]
2. Decision value of restricted mean survival time (RMST) [3,4]
3. Calibration curve for predicted versus observed absolute risk reduction

For all statistics, we calculate bootstrap confidence intervals through 
stratified resampling of the dataset.

#### Replication 

In order to replicate results with the default 250 bootstrap samples, run:

```
python3 baselines.py --dataset combined

python3 run.py --model xlearner --dataset combined
python3 run.py --model cox --dataset combined

python3 evaluate.py --model xlearner --dataset combined
python3 evaluate.py --model cox --dataset combined

python3 optimism.py --model cox --dataset combined
```

Code to reproduce plots can be found in our notebook `plots.ipynb`.

#### References

[1] Künzel, Sören R., Jasjeet S. Sekhon, Peter J. Bickel, and Bin Yu. “Meta-Learners for Estimating Heterogeneous Treatment Effects Using Machine Learning.” ArXiv:1706.03461 [Math, Stat], June 12, 2017. http://arxiv.org/abs/1706.03461.

[2] Klaveren, David van, Ewout W. Steyerberg, Patrick W. Serruys, and David M. Kent. “The Proposed ‘Concordance-Statistic for Benefit’ Provided a Useful Metric When Modeling Heterogeneous Treatment Effects.” Journal of Clinical Epidemiology 94 (February 2018): 59–68. https://doi.org/10.1016/j.jclinepi.2017.10.021.

[3] Schuler, Alejandro, and Nigam Shah. “General-Purpose Validation and Model Selection When Estimating Individual Treatment Effects.” ArXiv:1804.05146 [Cs, Stat], April 13, 2018. http://arxiv.org/abs/1804.05146.

[4] Royston, Patrick, and Mahesh KB Parmar. “Restricted Mean Survival Time: An Alternative to the Hazard Ratio for the Design and Analysis of Randomized Trials with a Time-to-Event Outcome.” BMC Medical Research Methodology 13 (December 7, 2013): 152. https://doi.org/10.1186/1471-2288-13-152.
