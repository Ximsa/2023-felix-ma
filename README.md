# 2023-Felix-MA

## Requirements
We used Python 3.9 for this project.

Python requirements are listed in [requirements.txt](https://gitlab.informatik.uni-ulm.de/dbis/data-science-and-big-data-analytics/teaching/2023-felix-ma/-/blob/main/requirements.txt?ref_type=heads)

To perform plotting and evaluation of the results, gnuplot has to be installed.

## Project structure

| File     | Description |
| -------- | ------- |
| datasets.py  | Loads and processed datasets |
| job_omp.sh | Used for slurm job submission |
| models.py | Contains model definitions |
| plot_averages.gnuplot | Plots the results of the current directory and calculates means and standard deviations |
| plot_single.gnuplot | Plots individual results of a run |
| requirements.txt | Python requirements |
| run_config.py | Main python file, loads and runs given config file|
| sampling.py | Contains sampling strategies for the active learner |
| submit.sh | (slurm) Submits every config from config/ with settings from job_omp-sh |
| train.py | Contains training and evaluation functions |
| util.py | Miscellaneous utility functions |
## Getting started

To run experiments you have to define run configurations as seen in config/.

Run configurations use the [edn](https://github.com/edn-format/edn) file format. e.g.:

```clj
{"model_names" ["GPN-GCN", "GCN", "LP"] ;; see models.py for available model names
 "dataset_names" ["Cora", "CiteSeer"] ;; see datasets.py for available dataset names
 "samplers" ["own"] ;; see sampling.py for availible samplers. Subsamplers get ignored if sampler is not "own"
 "num_steps" 20 ;; number of sampler runs
 "samples_per_step" 1 ;; in multiples of #classes
 "seed" 3133742069 ;; seed tah generates individual seeds for each run
 "repeats" 10 ;; number of repeats with different seeds
 "hyperparameters"
 {"subsampler" ["random" "medoids" "entropy" "pagerank" "own"] ;; see sampling.py for available subsamplers
  "label_propagation_uncertainty_treshold" [0 0.2] ;; 0 disables label propagation, 0-1 determines the threshold to keep propagate labels, > 1 overrides all labels with label propagation
  "hidden_dim_size" [64 128] ;; hiden dimension sizes to try
  "dropout" [0.25 0.5 0.75] ;; model dropout
  "distance_loss_weight" [0.5 1 2] ;; GPN only: balance between loss and regularisation term
  "learning_rate" [0.005 0.001] ;; learning rates to try
  "corruption" [0 0.1]}} ;; relative amount of wrong labeled vertices in labeled and unlabeled set.

```
To run a config, simply type
```python
python run_config.py [config]
```


The results are saved in the results/\[dataset\] folder as a csv table, where individual runs are seperated by 2 empty lines for gnuplot to recognize it as data blocks.

To plot the results and calulate means and standart deviation run
```sh
gnuplot -c plot_averages.gnuplot [metric]
```

This plots all csv's in a directory. Availible metrics can be taken from the table header in a csv. If you want to use a different "Terminal" for gnuplot, set the "GNUTERM" environment variable to the respective terminal, eg.:
```sh
GNUTERM=dumb gnuplot -c plot_averages.gnuplot "Test accuracy"
```
