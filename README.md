# Arithmetic Multilayer Perceptron
We implement a multilayer perceptron (MLP) that learns arithmetic operations: addition, subtraction, multiplication, division, and modulo. This implementation was used in the CogSci 2019 paper titled "Problem Difficulty in Arithmetic Cognition: Humans and Connectionist Models".

# About Files and Directories

- `cogsci_final_experiment.sh`: The shell script used in the final camera-ready paper. This script trains 3000 MLPs.
- `cogsci_experiment.sh`: The shell script used in the submission paper. This script trains 100 MLPs.
- `config.py`: To set hyperparameters in the experiment.
- `data_utils.py`: A Python script helping manipulate data.
- `mlp_run.py`: A Python script that trains MLPs. Written with Tensorflow. This is the main script.
- `rm_records.py`: A Python script helping remove all `run_info` of a certain experiment.
- `run_info_utils.py`: A Python script helping read and write `run_info`.
- `utils.py`: A Python script helping `mlp_run.py`.

