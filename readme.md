### Predicting Individual Patient Treatment Effects from Randomized Trial Data

R code for the X-learner with random forest base learners can be found in
`lib/xlearner-rf.R`. By default, it makes out-of-bag predictions.

In order to replicate results with the default 250 bootstrap samples, run:

```
python3 run.py --model xlearner --dataset combined
python3 run.py --model coxph --dataset combined

python3 evaluate.py --model xlearner --dataset combined
python3 evaluate.py --model coxph --dataset combined
```

