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

# Acknowledgement

This work was partly supported by the Institute for Information & Communications Technology Promotion (R0126-16-1072-SW.StarLab, 2017-0-01772-VTT, 2018-0-00622-RMI, 2019-0-01367-BabyMind) and Korea Evaluation Institute of Industrial Technology (10060086-RISF) grant funded by the Korea government (MSIP, DAPA).

# Citation

When you reuse this implementation, cite the following.

```
@inproceedings{ChoLHZ19,
  author    = {Sungjae Cho and Jaeseo Lim and Chris Hickey and Byoung{-}Tak Zhang},
  title     = {Problem Difficulty in Arithmetic Cognition: Humans and Connectionist Models},
  booktitle = {Proceedings of the 41th Annual Meeting of the Cognitive Science Society},
  pages     = {1506--1512},
  year      = {2019}
}
```

# License

MIT License
