# hetgp-eval
Heteroscedastic GP evaluations

## Implemented Methods
* Most Likely Heteroscedastic GP (MLHGP): https://dl.acm.org/doi/10.1145/1273496.1273546
* Improved MLHGP (IMLHGP): https://ieeexplore.ieee.org/document/9103623
* Nearest-neighbor point estimate HGP: https://arc.aiaa.org/doi/10.2514/6.2021-1589
* Our KS-HGP

## installation
Python 3.9 conda mac install:
```bash
conda install python==3.9.19
pip install -r req_macos.txt
```


## Bayesian Optimization

### Running the Experiment

To run the default settings, use the following command:

```bash
python bo_experiment_run.py
```

To specify custom configurations and output, use:

```bash
python bo_experiment_run --config <your_experiment_config.json> --output <your_result.png>
```

To run only the visualization using an existing result pickle file, use:

```bash
python bo_experiment_run --result_file <your_pickle_result.pkl>
```

Note: When using the `--result_file` option, the configuration will be read directly from the pickle file, and the `--config` argument will be ignored.

---

#### Configuration Settings

The settings are configured using a JSON file. By default, the script uses `bo_experiment_config.json`.

Each method's configuration is structured as follows:

```json
"method_name": {
    "model": "model_name",
    "acq_f": {
        "name": "acquisition_function_name",
        "params": {
            "acq_params_1": value,
            "acq_params_2": value
        }
    },
    "bo_optimizer": "acquisition_function_optimizer"
}
```

#### Options

- **`"model"`**: Specifies the model to use. Options include:
  - `GP`
  - `KSMLHGP`
  - `KSIMLHGP`
  - `MLHGP`
  - `IMLHGP`
  - `NNPEHGP`

- **`"acq_f"`**: Specifies the acquisition function. Choices are:
  - `EI`
  - `LCB`
  - `RAHBO`
  - `ANPEI`
  - `AugmentedEI`

- **`"params"`**: Parameters specific to the chosen acquisition function. These vary depending on the function selected.

- **`"bo_optimizer"`**: Specifies the optimizer for the acquisition function. Options include:
  - `Random`
  - `CMAES`
