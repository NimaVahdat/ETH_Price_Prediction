import os.path as osp

import yaml


def load_config_file(file_path):
    with open(file_path, "r") as yaml_file:
        try:
            data = yaml.safe_load(yaml_file)
            return data
        except yaml.YAMLError as e:
            print("Error Loading Config File:", e)
            return None
