# 2023-Felix-MA

## Requirements

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

Run configurations use the [edn](https://github.com/edn-format/edn) file format.

| Field     | Description |
| -------- | ------- |
| model_names | Model names as defined in models.py |
| samplers | Samplers as defined by sampling.py |
| num_steps | How often to sample |
| sampler_per_step | how many sampler to draw in multiples of amount of classes |
| seed | seed that generates seeds for each repeat |
| repeats | how often to repeat the experiments |
| subsampler | subsamplers from sampling.py to try |
| label_propagation_uncertainty_treshold | label prpagation: 0 disables label propagation, non-zero filters "uncertain" labels |
|hidden_dim_size| hidden dim sizes of used models |
| dropout | dropout used in models |
| distance_loss_weight | (GPN only) weighing of the loss and regularisation terms |
|learning_rate| learning rates used |

To run a config, simply type
```
python run_config.py [config]
```


The results are saved in the results/\[dataset\] folder as a csv table, where individual runs are seperated by 2 empty lines for gnuplot to recognize it as data blocks.

To plot the results and calulate means and standart deviation run
```
gnuplot -c plot_averages.gnuplot [metric]
```

This plots all csv's in a directory. Availible metrics can be taken from the table header in a csv. If you want to use a different "Terminal" for gnuplot, set the "GNUTERM" environment variable to the respective terminal, eg.:
```
GNUTERM=dumb gnuplot -c plot_averages.gnuplot "Test accuracy"
```