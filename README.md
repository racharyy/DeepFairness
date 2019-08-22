# DeepFairness



## Run command (All commands must be issued from the DeepFairness folder)

python -m deep_fairness.runner --conf confs/experiment.yaml


Data size
=========
orig_concat_data: N x 786 (768 + 1 + 7 + 10) ---> N x 14 (true label)

cf_concat_data: N*11(all possible cf protected att.)*10(sampled unobserved skill) x 786

Loss for cf_concat_data: 

