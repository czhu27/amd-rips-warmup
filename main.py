import os
import datetime
import argparse
import copy

import yaml

from helpers import Configs

from train import train

def make_configs(changes_configs, default_configs):
	# Merge the two configs
	configs = default_configs.copy()
	configs.update(changes_configs)

	# Convert dict to object
	configs = Configs(**configs)
	
	return configs

def get_filename(path):
	return os.path.basename(path).split(".")[0]

def run_trials(configs):
	run_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	configs.output_dir = configs.output_dir + "/" + "run_" + run_name

	# For each run, update output directory and seed
	for trial_id in range(configs.trials):
		trial_configs = copy.deepcopy(configs)
		trial_configs.output_dir = configs.output_dir + "/" + f"trial_{trial_id}"
		trial_configs.seed = configs.seed + trial_id

		# RUN TRIAL!
		train(trial_configs)

def grid_search(search_file, default_configs):
	print("Running a grid search.")
	print("YAML file: ", search_file)
	search_configs = yaml.safe_load(open(search_file))
	assert len(search_configs) == 1, "Only supports grid search in one argument"
	key = list(search_configs.keys())[0]
	values = search_configs[key]
	all_configs = []
	for i, value in enumerate(values):
		changes_configs = {key: value}
		configs = make_configs(changes_configs, default_configs)

		search_file_name = get_filename(search_file)
		if isinstance(value, (list, tuple)):
			value_name = '_'.join(str(i) for i in value)
		else:
			value_name = str(value)
		configs.output_dir = configs.output_root + "/search/" + search_file_name + f"/{key}={value_name}"

		all_configs.append(configs)
		
	for configs in all_configs:
		run_trials(configs)

def single_configuration(changes_file, default_configs):
	print("Running a single configs file.")
	print("YAML file: ", changes_file)
	changes_configs = yaml.safe_load(open(changes_file))
	configs = make_configs(changes_configs, default_configs)

	changes_file_name = get_filename(changes_file)
	configs.output_dir = configs.output_root + "/single/" + changes_file_name

	#with tf.device('/cpu:0'):
	run_trials(configs)
	

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-t", "--toy", action="store_true")
	parser.add_argument("-w", "--wave", action="store_true")
	parser.add_argument("-g", "--grid-search", type=str, default=None)
	parser.add_argument("-s", "--single-run", type=str, default="configs/single/test.yaml")
	args = parser.parse_args()

	# Load default dict from yaml file
	if args.toy:
		print("Training on toy problem")
		default_configs = yaml.safe_load(open("configs/default/toy.yaml"))
	elif args.wave:
		print("Training on wave equation")
		# Default to wave
		default_configs = yaml.safe_load(open("configs/default/wave.yaml"))
	else:
		raise ValueError("Specify which problem to train on, please (toy or wave)")

	if args.grid_search is not None:
		grid_search(args.grid_search, default_configs)
	
	else:
		single_configuration(args.single_run, default_configs)