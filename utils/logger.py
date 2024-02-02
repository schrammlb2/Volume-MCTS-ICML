import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import ipdb
import os
from scipy.stats import beta

color_list = ['r', 'g', 'b', 'c', 'm', 'k']
color_list = [x for x in mcolors.TABLEAU_COLORS]

epochs_to_interactions = True

def bayes_interval(n_successes, n_failures, z: float = .95) -> tuple:
	prior_alpha = .5
	prior_beta = .5

	rv = beta.interval(z, prior_alpha + n_successes, prior_beta + n_failures)
	return rv#(round(rv[0], 4), round(rv[1], 4))

def plot_ci(x_list, y_list, color='r', label=''):
	means = [np.mean(ys) for ys in y_list]
	plot_confidence_interval = True
	if plot_confidence_interval:
		stds  = [np.std(ys)/(len(ys))**0.5 for ys in y_list]
		y_plus  = [mean + 2*std for mean, std in zip(means, stds)]
		y_minus = [mean - 2*std for mean, std in zip(means, stds)]
		plt.fill_between(x_list, y_plus, y_minus, alpha=.4, color=color)

	plt.plot(x_list, means, label=label, color=color)

def plot_bayes_interval(x_list, y_list, color='r', label=''):
	means = [np.mean(ys) for ys in y_list]
	plot_confidence_interval = True
	if plot_confidence_interval:
		y_plus = []
		y_minus = []
		for successes in y_list:
			n_successes = sum(successes)
			n_failures  = len(successes) - sum(successes)
			interval = bayes_interval(n_successes, n_failures)
			y_minus.append(interval[0])
			y_plus.append(interval[1])
		# import ipdb
		# ipdb.set_trace()
		plt.fill_between(x_list, y_plus, y_minus, alpha=.4, color=color)

	plt.plot(x_list, means, label=label, color=color)

def method_renamer(name):
	original_name=name
	name = name.replace("continuous_external_training_HER_her:False_az:True_full_traj_her:False", "AlphaZero with AZ training")
	name = name.replace("continuous_external_training_HER_her:True_az:False_full_traj_her:False", "AlphaZero with HER training")
	name = name.replace("external_training_HER_her:False_az:True_full_traj_her:False", "Volume-MCTS with AZ training")
	name = name.replace("external_training_HER_her:True_az:False_full_traj_her:False", "Volume-MCTS with HER training")

	name = name.replace("open_loop_continuous_off_policy_HER", "OL AZ")
	name = name.replace("continuous_off_policy_HER", "CL AZ")
	name = name.replace("off_policy_HER", "OL Volume-MCTS")
	# name = name.replace("with_az_training", " after training")
	name = name.replace("open_loop", "OL")
	name = name.replace("close_loop", "CL")
	name = name.replace("closed_loop", "CL")
	name = name.replace("off_policy", "V-MCTS")
	name = name.replace("with_az_training", " AT")
	# name = name.replace("off_policy", "")
	name = name.replace("_HER", "")
	



	name = name.replace("one_shot_2", "Volume-MCTS")
	name = name.replace("open_loop_continuous", "Open-loop AlphaZero")
	name = name.replace("continuous", "AlphaZero")
	name = name.replace("never_give_up", "AlphaZero w/ CBE")
	# name = name.replace("with_training", "after training")
	name = name.replace("with_training", "")
	# if "after training" not in name: 
	# 	name += " before training"
	print(
		f"Original name: {original_name}\n"
		f"New name: {name}\n\n"
	)
	# import ipdb
	# ipdb.set_trace()

	name = name.replace("_", " ")
	return name

def environment_renamer(name):
	if epochs_to_interactions:
		name = name.replace("epoch", "environment interations")
	name = name.replace("planning steps", "environment interactions")

	name = name.replace("local_maze", "Maze")
	name = name.replace("dubins", "Dubins")
	name = name.replace("_multi_seed", " ")
	name = name.replace("_", " ")
	name = name.replace("-", " ")
	name = " ".join((s.capitalize() for s in name.split(" ")))
	return name


class Logger:
	def __init__(self, variable_list, env_name=None, filename="log", relative_directory=""):
		# self.variable_list = variable_list
		# for var in variable_list:
		# 	assert type(var) is str
		# if x_axis_variable_name == None:
		# 	assert type(x_axis_name) == str
		# 	assert x_axis_variable_name in variable_list
		# else:
		# 	self.x_var = self.variable_list[0]

		self.all_data = []
		self.filename = filename + ".pkl"
		self.variable_list = variable_list
		self.env_name = env_name
		directory = os.getcwd().split("outputs")[0]
		self.base_dir = directory
		if directory[-1] != "/":
			self.base_dir += "/"
		self.base_dir += "data/"
		self.base_dir += relative_directory 


	def log(self, dictionary):
		self.all_data.append(dictionary)

	def process(self, x_axis_name):
		processed_log = {}
		self.x_axis_name = x_axis_name
		for item in self.all_data:
			if x_axis_name in item:
				x_val = item[x_axis_name]
				if x_val not in processed_log:
					processed_log[x_val] = [item]
				else:
					processed_log[x_val].append(item)

		if self.env_name == None or f"/{self.env_name}/" in self.filename:
			filename = self.base_dir + self.filename
		else: 
			target_directory = self.base_dir + self.env_name + "/"
			if not os.path.isdir(target_directory):
				os.mkdir(target_directory)
			filename = target_directory + self.filename

		# with open(self.base_dir + self.filename, "wb") as file:
		with open(filename, "wb") as file:
			pickle.dump(processed_log, file)

		self.processed_log = processed_log

	# def show(self, show_result = True, label='', color='r'):
	def show(self, show_result = False, label='', color='r', smoothing=0):
		if self.x_axis_name != None: 
			for variable in self.variable_list:
				if variable == self.x_axis_name:
					continue

				xs = sorted(list(self.processed_log.keys()))
				# ys = [[entry[variable] for entry in self.processed_log[x]] for x in xs]
				original_ys = []
				for x in xs:
					ys_element = []
					for entry in self.processed_log[x]:
						ys_element.append(entry[variable])
					original_ys.append(ys_element)



				# width = 2
				# ys = []
				# for i, x in enumerate(xs):
				# 	lower_bound = max(0, i-width)
				# 	upper_bound = min(len(xs), i+width+1)
				# 	ys_element = []
				# 	for j in range(lower_bound, upper_bound):
				# 		for entry in self.processed_log[xs[j]]:
				# 			ys_element.append(entry[variable])

				# 	ys.append(ys_element)

				width = smoothing
				ys = []
				for i, x in enumerate(xs):
					lower_bound = max(0, i-width)
					# upper_bound = min(len(xs), i+width+1)
					upper_bound = i#min(len(xs), i+width+1)

					ys_element = []
					for k, elem in enumerate(original_ys[i]):
						mean_element = []
						# for j in range(lower_bound, upper_bound):
						for j in range(lower_bound, i+1):
							mean_element.append(original_ys[j][k])
						ys_element.append(sum(mean_element)/len(mean_element))

					ys.append(ys_element)

				if self.x_axis_name == "epoch" and epochs_to_interactions:
					# from hydra import compose, initialize
					from omegaconf import OmegaConf

					# with initialize(config_path="config/env/Quadcopter"):
					#     cfg = compose(config_name="config")
					path = f"config/env/{self.env_name}.yaml"
					cfg = OmegaConf.load(path)
					num_interactions = cfg.env.num_train_episodes*cfg.env.n_rollouts_per_step
					# num_interactions = 3000
					xs = [x*num_interactions for x in xs]


				if self.variable_list[-1] == "success":
					plot_bayes_interval(xs, ys, label=label, color=color)
				else:
					plot_ci(xs, ys, label=label, color=color)

				x_axis_name = environment_renamer(self.x_axis_name)
				plt.xlabel(x_axis_name)
				plt.ylabel(variable)
				if show_result:
					plt.show()
		else:
			ys = [
				entry[variable] 
				for variable in self.variable_list if variable != None
				for x in sorted(list(self.processed_log.keys()))
				for entry in self.processed_log[x] 
			]


			if self.variable_list[-1] == "success":
				n_successes = sum(ys)
				n_failures = len(ys) - n_successes
				interval = bayes_interval(n_successes, n_failures)
				mean = (interval[1] + interval[0])/2
				error = (interval[1] - interval[0])/2
			else:				
				mean = np.mean(ys)
				std  = np.std(ys)/(len(ys))**0.5
				error = 2*std
			plt.bar(label, mean, yerr=error, align='center', alpha=0.5, capsize=10)
			# plt.xlabel(environment_renamer(self.x_axis_name))
			plt.ylabel(self.variable_list[-1])
			if show_result:
				plt.show()


	def plot_all(self, x_axis_name, include_training=None, smoothing=0):
		self.x_axis_name = x_axis_name
		import os
		directory = self.base_dir + self.env_name + "/"
		print(directory)

		files = os.listdir(directory)

		cleaned_file_list = sorted([file_name for file_name in files if (".pkl" in file_name)])
		ignore_subset_list = ["with_HER"]
		cleaned_file_list = [
			f for f in cleaned_file_list 
			if not any([(ignore in f) for ignore in ignore_subset_list])
		]
		if include_training != None:
			if include_training == True:
				cleaned_file_list = filter(lambda x: "with_training" in x, cleaned_file_list)
				training_extension = "After_Training"
			elif include_training == False:				
				cleaned_file_list = filter(lambda x: not ("with_training" in x), cleaned_file_list)
				training_extension = "Before_Training"
			else: 				
				raise Exception("include_training must be True, False, or None")
		else: 
			training_extension = ""
	
		for i, file_name in enumerate(cleaned_file_list):
			with open(directory + file_name, "rb") as file:
				self.processed_log = pickle.load(file)
			try: 
				color = color_list[i]
			except IndexError:
				if i > len(color_list) - 1:
					print("Too many values for the current color list. Use a larger set of colors")
				raise IndexError
			label = method_renamer(file_name[:-4])
			self.show(show_result=False, label=label, color=color, smoothing=smoothing)
		
		plt.legend()
		title = environment_renamer(f"{self.env_name} {training_extension}")
		plt.title(title)

		filename = f"{directory}{self.env_name}_{self.variable_list[-1]}_{training_extension}.pdf"
		plt.savefig(filename)
		plt.show()
