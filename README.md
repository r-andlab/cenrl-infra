# CenRL
[![DOI](https://zenodo.org/badge/1078492218.svg)](https://doi.org/10.5281/zenodo.17381989)

CenRL is a reinforcement learning (RL) framework for optimizing and automating censorship measurements. CenRL intelligently selects and measures potential censorship targets through sequential decision making. This design enables real-time, dynamic decision-making efficiently utilizing the limited measurement resources available and adapting to the evolving landscape of censorship. CenRL formulates the censorship measurement task as a multi-armed bandit (MAB) problem, where an intelligent entity is given the goal of optimizing censorship detection within a limited time period. CenRL provides two functionalities: (Task 1:) maximizing the discovery of blocked websites within a network and (Task 2 (dyn):) rapidly and automatically detecting changes in blocking over time within a dynamic environment. For both tasks, CenRL operates on a large input list of websites to test, such as the Tranco list of popular websites. CenRL uses several features about websites for its action space: category, subdomain, TLD, website rank, and parent entity. These features are carefully selected to capture censorship patterns. This repository provides both controlled evaluation environments (ground truths gfwatch (CHINA), russia (RUSSIA), and kazakhstan (KAZAKHSTAN)), as well as APIs for real-world censorship measurements. For more information, refer to [our paper](https://ramakrishnansr.com/assets/cenrl.pdf).


## Integrating CenRL Into Real-world Measurements

Follow the setup in [Prerequisites](#prerequisites), then do the following:

### 1. Create the Real-world Model
CenRL can be integrated into real-world measurement platforms leveraging our RL models as an API. We provide an example at [ucb_naive_api.py](api/ucb_naive_api.py) on how to use the CenRL UCBNaive implementation and integrate with an active censorship measurement platform.

1. Start a new class [UCBNaiveAPI](api/ucb_naive_api.py) that inherits from a model, e.g., [UCBNaive](models/ucb/ucb_naive.py)
2. Do any necessary initialization.
3. Implement the take_measurements function to call the platform's measurement function directly and use the selected_target (e.g., the website to be tested for censorship). See a test version [UCBNaiveAPITest](api/ucb_naive_api_test.py).

### 2. Create and Run the Eval Script for the Real-world Model

Now, create a corresponding bash script to run the real-world model. We provide an example at [eval_ucb_naive_real_world.py](scripts/eval_full_real_world/evaluate_ucb_naive_real_world.sh). Make sure to select your hyperparameters. Tune them using [controlled environments](#cenrl-controlled-environments), if necessary. 

### 3. Outputs of the Eval Script for the Real-world Model

The `evaluate_ucb_naive_real_world.sh` script will output its results in `real-world/outputs/`.
For real-world experiments, the relevant files are the following:
- `action_space.graphml`: the constructed action space used in the experiment
- `ucb_naive_api_*.csv`: the entire results per time step for all episodes in csv format
- `plots/ucb_naive_api_grid.pdf`: the plot of occurrences of censorship found vs. time steps

### Quick Test for Real-world Model

For demonstration purposes, you can run [UCBNaiveAPITest](api/ucb_naive_api_test.py) using:

`HOME="/path/to/parent-of-cenrl" bash scripts/eval_full_real_world/evaluate_ucb_naive_real_world.sh -e 1 -m 1000 -f "categories" -s "top10k" -a "ucb_naive_api_test" -r "false"`

See more in [eval_ucb_naive_real_world.py](scripts/eval_full_real_world/evaluate_ucb_naive_real_world.sh), but the parameters are:
- `-e`: number of episodes of the experiment to run
- `-m`: number of measurements per episode (time steps)
- `-f`: which category is used to build the action space
- `-s`: which test list file to build the action space
- `-a`: which RL model to use within the `api` directory. This matches the name of the model file.
- `-r`: once an arm is chosen, whether we select a domain to take_measurement using its rank to increase its selection probability (i.e., higher ranking domains are more likely to be selected for measurement).


## CenRL Controlled Environments

Controlled environments are a way to tune hyperparameters, test out RL components, such as RL models, policies, reward functions, and action spaces. This enables users of CenRL to then apply the findings and hyperparameters in their [real-world integration](#integrating-cenrl-into-real-world-measurements).

In this section, we provide instructions on how to use the controlled environment for [UCB](/scripts/eval_full/evaluate_ucb_naive.sh), but instructions also apply to similar scripts in that directory. 


### 1. Understanding Controlled Environments

Controlled environments simulate how real world censorship environments can behave. 
- `Ground Truth File`: What is censored is determined by a given blocklist (we call this the ground truth file). An example is provided in [gfwatch blocklist](inputs/gfwatch/gfwatch-blocklist.csv).
- `Test List File`: The scope of what the user of CenRL cares in learning what is being censored is determined by a test list file. An example is provided in [Tranco-Top10K](inputs/tranco/tranco_categories_subdomain_tld_entities_top10k.csv). In other words, using the Tranco-Top10K means that the user only cares about the domains in that list. 
- `Action Space and Feature`s`: The Test List File is utilized to construct the [action space](/models/base/action_space.py). The set of arms that CenRL will pull/test to learn which domains are being censored. The feature (e.g., categories) determine how the items in the Test List File will be grouped together within an arm. A feature must exists as a column name in the Test List File.
- `RL Model`: Models implement RL based components and algorithms, such as its policies, reward function, etc. An example for [UCB](/models/ucb/ucb_naive.py) is given. 

### 2. Scripts for Controlled Environments

Follow the setup in [Prerequisites](#prerequisites).

[Scripts for controlled environments](/scripts/eval_full/) are provided for several RL algorithms. Taking [UCB script](/scripts/eval_full/evaluate_ucb_naive.sh) as an example, it provides several nested for loops to test out hyperparameters, and then plot them together for comparison.

*WARNING*: running the below script will take awhile due to nested loops trying several different hyperparameters.

An example run can be (replace the HOME path with the parent directory of CenRL) (it should under 5 minutes to run):
`HOME="/path/to/parent-of-cenrl" bash evaluate_ucb_naive.sh -e 2 -m 10000 -g gfwatch -f "categories" -s "top10k" -a "ucb_naive" -r "false"`

Important parameters include:
- `-e`: number of episodes of the experiment to run
- `-m`: number of measurements per episode (time steps)
- `-g`: which ground truth file is used to simulate the controlled environment
- `-f`: which category is used to build the action space
- `-s`: which test list file to build the action space
- `-a`: which RL model to use within the `models` directory. This matches the name of the model file.
- `-r`: once an arm is chosen, whether we select a domain to take_measurement using its rank to increase its selection probability (i.e., higher ranking domains are more likely to be selected for measurement).

### 3. Outputs of the Script for the Controlled Environments

Taking [UCB script](/scripts/eval_full/evaluate_ucb_naive.sh) as an example, it will output its results in `models/outputs/`.
The relevant files are the following:
- `action_space.graphml`: the constructed action space used in the experiment
- `ucb_naive_*.csv`: the entire results per time step for all episodes in csv format. One CSV file per hyperparameter configuration.
- `plots/ucb_naive_grid.pdf`: the plot of occurrences of censorship found vs. time steps for all experiments for comparison. Ordered by their area under the curve. This shows which hyperparameters perform better.
- `plots/coverage_*_grid.pdf`: the plot of % of coverage of ground truth file vs. time steps for all experiments for comparison. Ordered by their area under the curve.This shows which hyperparameters perform better in terms of finding all possible censored domains from the ground truth file.

### Quick Test for Controlled Environments

For demonstration purposes, we provide a test eval file that finishes quickly by reducing the number of hyperparameters being tested at [UCB](/scripts/eval_full/evaluate_ucb_naive_test.sh). Run it using (replace the HOME path with the parent directory of CenRL):
`HOME="/path/to/home" bash scripts/eval_full/evaluate_ucb_naive_test.sh -e 2 -m 1000 -g gfwatch -f "categories" -s "top10k" -a "ucb_naive" -r "false"`


## Prerequisites

Requires Python 3.11. (You may need update your pip install using `pip install --upgrade pip`)

1. Create a virtual environment using 

`python3 -m venv <name of virtual environment>`

2. Activate the environment using

`source <name of virtual environment>/bin/activate`

3. Install requisite packages using 

`pip3 install -r requirements.txt`

Example Expected output:
> Successfully installed cachetools-5.3.0 certifi-2022.12.7 charset-normalizer-3.1.0 cloudpickle-2.2.1 db-dtypes-1.0.5 google-api-core-2.11.0 google-auth-2.16.2 google-cloud-bigquery-3.7.0 google-cloud-core-2.3.2 google-crc32c-1.5.0 google-resumable-media-2.4.1 googleapis-common-protos-1.58.0 grpcio-1.51.3 grpcio-status-1.51.3 gym-0.26.2 gym-notices-0.0.8 idna-3.4 numpy-1.26.4 packaging-23.0 pandas-1.5.2 proto-plus-1.22.2 protobuf-4.22.1 pyarrow-11.0.0 pyasn1-0.4.8 pyasn1-modules-0.2.8 python-dateutil-2.8.2 pytz-2022.7.1 requests-2.28.2 rsa-4.9 six-1.16.0 tqdm-4.65.0 urllib3-1.26.15

4. Install CenRL module locally

- go to root directory of this project
- `pip3 install -e .`

Example Expected output:
> Running setup.py develop for cenrl 
Successfully installed braveblock-0.5.1 cenrl-1.0a1 contourpy-1.3.3 cycler-0.12.1 fonttools-4.60.1 kiwisolver-1.4.9 matplotlib-3.10.6 networkx-3.5 pillow-11.3.0 pyparsing-3.2.5 seaborn-0.13.2


5. Run CenRL. See [Integrating CenRL into Real World Measurements](#integrating-cenrl-into-real-world-measurements).

Once you are done with using CenRL, you can deactivate the virtual environment using `deactivate`

## File organization

`inputs/` contains example data that can be used as input to the models. Data is grouped under folders depending on the source of the data. Other data sources can be added. The `tranco` dataset can be used as the input list of websites, while the `gfwatch`, `russia`, and `kazakhstan` datasets can be used as  simulated environments for controlled experiments. 
`models/` contains specific models as well as preprocessor functionality and config files.
- `models/base` contains common code and base classes that will be used by other files
- `models/epsilon_greedy` contains code for epsilon greedy
- `models/thompson_sampling` contains code for thompson sampling
- `models/ucb` contains code for UCB
`\api` contains specific models to integrate with measurement platforms with CenRL models.
`scripts/` contains helper scripts to evaluate the models. 
- `scripts/eval_full` contains eval scripts for hyperparameter tuning for controlled environments
- `scripts/eval_full_real_world` contains eval scripts to run CenRL integrated with a real-world measurement platform.

## Disclaimer
Russing CenRL measurements from your machine may place you at risk if you use it within a highly restrictive networks. Therefore, please exercise caution while using the tool, and understand the risks of running CenRL before using it on your machine. Please refer to [our paper](https://ramakrishnansr.com/assets/cenrl.pdf) for more information.

## Citation
If you use the CenRL tool or data, please cite the following publication:
```
@inproceedings{le2025cenrl,
  title={CenRL: A Framework for Performing Intelligent Censorship Measurements},
  author={Le, Hieu and Huremagic, Armin and Wang, Kevin and Ensafi, Roya and Sundara Raman, Ram},
  booktitle={IEEE Security & Privacy 2026 (IEEE S&P 26)},
  year={2026}
}
```


## Contact
For any questions, reach out to rsundar2@ucsc.edu, levanhieu@gmail.com, musicer@umich.edu, agix@umich.edu, and ensafi@umich.edu.

- Real-world Data Notice: The data collected during our real-world experiments are snapshots of detected blocking at specific moments in time and not meant to be used to replicate results. Feel free to contact us for access to the data or more information.

