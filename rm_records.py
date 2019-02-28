import sys
import shutil
import config


def main():
    rm_records()

def rm_records():
    experiment_name = sys.argv[1]
    path_list = list()
    path_list.append(config.dir_run_info_experiments())
    path_list.append(config.dir_measure_log())
    path_list.append(config.dir_saved_models())
    path_list.append(config.dir_logs())

    for path in path_list:
        shutil.rmtree('{}/{}'.format(path, experiment_name), ignore_errors=True)


if __name__ == "__main__":
    # execute only if run as a script
    main()
