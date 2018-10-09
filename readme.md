### Predicting Individual Patient Treatment Effects from Randomized Trial Data

---

### X-learner with random forests

R code for the X-learner with random forest base learners can be found in
`lib/xlearner-rf.R`. By default, it makes out-of-bag predictions. 

### Replication 

In order to replicate results with the default 250 bootstrap samples, run:

```
python3 baselines.py --dataset combined

python3 run.py --model xlearner --dataset combined
python3 run.py --model cox --dataset combined

python3 evaluate.py --model xlearner --dataset combined
python3 evaluate.py --model cox --dataset combined

python3 optimism.py --model cox --dataset combined
```

