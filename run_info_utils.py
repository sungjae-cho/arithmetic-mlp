import pickle
import pandas as pd
import operator
import config
from pprint import pprint
from os import listdir
from os.path import isfile, join
from collections import Counter


def get_run_info_dir_path(experiment_name):
    return '{}/{}'.format(config.dir_run_info_experiments(), experiment_name)

def get_run_info_path(run_id, experiment_name):
    run_info_dir_path = get_run_info_dir_path(experiment_name)
    return '{}/run-{}.pickle'.format(run_info_dir_path, run_id)

def get_run_info(run_id, experiment_name):
    run_info_path = get_run_info_path(run_id, experiment_name)
    with open (run_info_path, 'rb') as f:
        run_info = pickle.load(f)
    return run_info

def print_run_info(run_id, experiment_name):
    run_info = get_run_info(run_id, experiment_name)
    pprint(run_info)

def is_all_correct(run_info):
    if run_info['init_all_correct_epoch'] != -1:
        return True
    else:
        return False

def get_all_run_info_files(experiment_name):
    run_info_dir_path = get_run_info_dir_path(experiment_name)
    run_info_files = [join(run_info_dir_path, f) for f in listdir(run_info_dir_path) if isfile(join(run_info_dir_path, f))]
    return run_info_files

def read_run_info_file(run_info_path):
    with open (run_info_path, 'rb') as f:
        run_info = pickle.load(f)
    return run_info


def import_all_run_info(experiment_name, operator=None, all_correct=None):
    all_run_info = list()
    run_info_files = get_all_run_info_files(experiment_name)
    for run_info_file in run_info_files:
        run_info = read_run_info_file(run_info_file)

        if all_correct == None:
            if operator == None:
                all_run_info.append(run_info)
            if operator == run_info['operator']:
                all_run_info.append(run_info)
        if all_correct == True and run_info['init_all_correct_epoch'] != -1:
            if operator == None:
                all_run_info.append(run_info)
            if operator == run_info['operator']:
                all_run_info.append(run_info)
        if all_correct == False and run_info['init_all_correct_epoch'] == -1:
            if operator == None:
                all_run_info.append(run_info)
            if operator == run_info['operator']:
                all_run_info.append(run_info)

    return all_run_info

def import_all_all_correct_run_info(experiment_name):
    all_run_info = list()
    run_info_files = get_all_run_info_files(experiment_name)
    for run_info_file in run_info_files:
        run_info = read_run_info_file(run_info_file)
        all_run_info.append(run_info)
    return all_run_info

def extract_number(string):
    str_number = str()
    for char in string:
        if ord('0') <= ord(char) and ord(char) <= ord('9'):
            str_number = str_number + char
    return int(str_number)


def get_epoch_keys(run_info):
    epoch_keys = list()
    for key in run_info.keys():
        if key.find('epoch') != -1:
            epoch_keys.append(key)
    return sorted(epoch_keys)


def get_carry_epoch_keys(run_info):
    epoch_keys = list()
    for key in run_info.keys():
        if key.find('init_all_correct_carry') != -1:
            epoch_keys.append(key)
        if key.find('init_complete_all_correct_carry') != -1:
            epoch_keys.append(key)

    return sorted(epoch_keys)


def get_digit_epoch_keys(run_info):
    epoch_keys = list()
    for key in run_info.keys():
        if key.find('init_all_correct_digit') != -1:
            epoch_keys.append(key)
        if key.find('init_complete_all_correct_digit') != -1:
            epoch_keys.append(key)

    return sorted(epoch_keys)


def epoch_order_count(all_run_info):
    init_all_correct_carry_epoch = list()
    init_complete_all_correct_carry_epoch = list()
    init_all_correct_digit_epoch = list()
    init_complete_all_correct_digit_epoch = list()

    for run_info in all_run_info:
        init_all_correct_carry_epoch_sorted = list()
        init_complete_all_correct_carry_epoch_sorted = list()
        init_all_correct_digit_epoch_sorted = list()
        init_complete_all_correct_digit_epoch_sorted = list()

        for key in run_info.keys():

            if key.find('init_all_correct_carry') != -1:
                carries = extract_number(key) # 0~sth
                epoch = run_info[key]
                init_all_correct_carry_epoch_sorted.append((carries, epoch))

            if key.find('init_complete_all_correct_carry') != -1:
                carries = extract_number(key) # 0~sth
                epoch = run_info[key]
                init_complete_all_correct_carry_epoch_sorted.append((carries, epoch))

            if key.find('init_all_correct_digit') != -1:
                digit_loc = extract_number(key) # 1~sth
                epoch = run_info[key]
                init_all_correct_digit_epoch_sorted.append((digit_loc, epoch))

            if key.find('init_complete_all_correct_digit') != -1:
                digit_loc = extract_number(key) # 1~sth
                epoch = run_info[key]
                init_complete_all_correct_digit_epoch_sorted.append((digit_loc, epoch))

        # End of key iteration
        init_all_correct_carry_epoch_sorted = sorted(init_all_correct_carry_epoch_sorted, key=lambda tup: tup[1])   # sort by epoch
        init_complete_all_correct_carry_epoch_sorted = sorted(init_complete_all_correct_carry_epoch_sorted, key=lambda tup: tup[1])   # sort by epoch
        init_all_correct_digit_epoch_sorted = sorted(init_all_correct_digit_epoch_sorted, key=lambda tup: tup[1])   # sort by epoch
        init_complete_all_correct_digit_epoch_sorted = sorted(init_complete_all_correct_digit_epoch_sorted, key=lambda tup: tup[1])   # sort by epoch

        init_all_correct_carry_epoch_sorted_carry_only = list()
        init_complete_all_correct_carry_epoch_sorted_carry_only = list()
        init_all_correct_digit_epoch_sorted_digit_only = list()
        init_complete_all_correct_digit_epoch_sorted_digit_only = list()

        for carries, _ in init_all_correct_carry_epoch_sorted:
            init_all_correct_carry_epoch_sorted_carry_only.append(carries)
        for carries, _ in init_complete_all_correct_carry_epoch_sorted:
            init_complete_all_correct_carry_epoch_sorted_carry_only.append(carries)
        for digit_loc, _ in init_all_correct_digit_epoch_sorted:
            init_all_correct_digit_epoch_sorted_digit_only.append(digit_loc)
        for digit_loc, _ in init_complete_all_correct_digit_epoch_sorted:
            init_complete_all_correct_digit_epoch_sorted_digit_only.append(digit_loc)

        init_all_correct_carry_epoch.append(init_all_correct_carry_epoch_sorted_carry_only)
        init_complete_all_correct_carry_epoch.append(init_complete_all_correct_carry_epoch_sorted_carry_only)
        init_all_correct_digit_epoch.append(init_all_correct_digit_epoch_sorted_digit_only)
        init_complete_all_correct_digit_epoch.append(init_complete_all_correct_digit_epoch_sorted_digit_only)

    str_init_all_correct_carry_epoch = list()
    str_init_complete_all_correct_carry_epoch = list()
    str_init_all_correct_digit_epoch = list()
    str_init_complete_all_correct_digit_epoch = list()

    for element in init_all_correct_carry_epoch:
        str_init_complete_all_correct_carry_epoch.append(str(element))

    for element in init_complete_all_correct_carry_epoch:
        str_init_all_correct_carry_epoch.append(str(element))

    for element in init_all_correct_digit_epoch:
        str_init_all_correct_digit_epoch.append(str(element))

    for element in init_complete_all_correct_digit_epoch:
        str_init_complete_all_correct_digit_epoch.append(str(element))

    result = {
        'init_complete_all_correct_carry_epoch':Counter(str_init_complete_all_correct_carry_epoch),
        'init_all_correct_carry_epoch':Counter(str_init_all_correct_carry_epoch),
        'init_all_correct_digit_epoch':Counter(str_init_all_correct_digit_epoch),
        'init_complete_all_correct_digit_epoch':Counter(str_init_complete_all_correct_digit_epoch)
    }

    return result
