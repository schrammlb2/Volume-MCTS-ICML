import numpy as np
from alphazero.helpers import argmax
from alphazero.search.kd_states import KDNodeContinuous, KDActionContinuous, PWAction
import ipdb

class Object(object): 
	pass

def bisection_search(f, lower_bound, upper_bound, depth, tolerance):
	#Finds a zero of the function f
	mid = (upper_bound + lower_bound)/2
	f_mid = f(mid)
	if abs(f_mid) < tolerance:
		return mid
	# f_plus = f(upper_bound)
	# f_minus = f(lower_bound)
	# print(f"Bounds: {lower_bound, upper_bound} \t X: {mid} \t f(X): {f_mid}")
	# print(f"\tf_-:{f_minus} \tf_+:{f_plus} ")
	if depth <= 0: 
		return mid
	if f_mid > 0:
		return bisection_search(f, mid, upper_bound, depth=depth-1, tolerance=tolerance)
	else: 	
		return bisection_search(f, lower_bound, mid, depth=depth-1, tolerance=tolerance)

def newton_search(f, lower_bound, upper_bound, f_lower_bound=None, f_upper_bound=None, 
		depth=10, tolerance=0.01):
	#NOTE: 
	#	f_lower_bound is f(lower_bound), not lower_bound(f(x))
	# 	Since f is monotonically decreasing on this range,
	#		this means that f_lower_bound > f_upper_bound always
	delta_f = f_upper_bound - f_lower_bound
	delta_x = upper_bound - lower_bound
	try: 
		assert f_upper_bound < 0
		assert f_lower_bound > 0
	except:
		ipdb.set_trace()
	# if delta_f < tolerance: return lower_bound
	assert delta_x >= 0
	if delta_x == 0: return lower

	try: assert delta_f < 0
	except: ipdb.set_trace()
	
	# tolerance
	mid = lower_bound - delta_x/delta_f*f_lower_bound
	assert mid >= lower_bound
	assert mid <= upper_bound
	# mid = (upper_bound + lower_bound)/2
	f_mid = f(mid)
	if abs(f_mid) < tolerance:
		return mid
	if depth <= 0: 
		return mid
	if f_mid > 0:
		return newton_search(f, mid, upper_bound, 
			f_lower_bound=f_mid, f_upper_bound=f_upper_bound, 
			depth=depth-1, tolerance=tolerance)
	else: 	
		return newton_search(f, lower_bound, mid, 
			f_lower_bound=f_lower_bound, f_upper_bound=f_mid, 
			depth=depth-1, tolerance=tolerance)

def _search(f, lower_bound, upper_bound, f_lower_bound=None, f_upper_bound=None, depth=5):
	tolerance = 0.01
	return bisection_search(f, lower_bound, upper_bound, depth=depth, tolerance = tolerance)
	# return newton_search(f, lower_bound, upper_bound,
	# 	f_lower_bound=f_lower_bound, f_upper_bound=f_upper_bound,
	# 	depth=depth, tolerance = tolerance)

def search(total_prob, alpha_lower_bound, alpha_upper_bound, depth=5):
	prob_low = total_prob(alpha_upper_bound)
	prob_high = total_prob(alpha_lower_bound)
	if prob_low <= 0.001 and prob_low >= -0.001: 
		alpha = alpha_upper_bound
	elif prob_high <= 0.001 and prob_high >= -0.001: 
		alpha = alpha_lower_bound
	else:
		if prob_low > 0:
			alpha = alpha_upper_bound
		elif prob_high < 0:
			alpha = alpha_lower_bound
		else: 
			alpha = _search(total_prob, alpha_lower_bound, alpha_upper_bound,
				f_lower_bound=prob_high, f_upper_bound=prob_low, depth=depth)
	est_total_prob = total_prob(alpha) + 1
	if prob_high**2 < (est_total_prob-1)**2:
		est_total_prob = prob_high + 1
		alpha = alpha_lower_bound
	elif prob_low**2 < (est_total_prob-1)**2:
		est_total_prob = prob_low + 1
		alpha = alpha_upper_bound
	
	return alpha, est_total_prob

def calculate_policy(actions, lam):
	if len(actions) == 1:
		return np.array([1.0], dtype=np.float32)

	# import ipdb
	# ipdb.set_trace()
	for action in actions:
		assert action.Q.shape == ()


	Q_mean = np.mean([action.Q for action in actions])
	Q_std = np.std([action.Q for action in actions])
	assert min([action.Q for action in actions]) > -1000
	def normalize(action):
		return action.Q#(action.Q - Q_mean)/(Q_std+0.01)
	single_prob = lambda action, alpha: lam/(alpha - normalize(action))*(1/len(actions))
	total_prob = lambda alpha: sum([single_prob(action, alpha) for action in actions])-1

	alpha_upper_bound = max([normalize(action) + lam for action in actions])
	alpha_lower_bound = max([normalize(action) + lam/len(actions) for action in actions]) 
	# try: 
	alpha, est_total_prob = search(total_prob, alpha_lower_bound, alpha_upper_bound, depth=10)
	assert est_total_prob < 1.1
	assert est_total_prob > 0.9
	# except: 
	# 	import ipdb
	# 	ipdb.set_trace()
	action_list = [(single_prob(action, alpha)/est_total_prob).reshape(-1) for action in actions]
	policy = np.concatenate(action_list)
	# if action
	# policy = np.array(action_list)
	return policy
	# print(alpha)
	# print(total_prob(alpha))
	# import ipdb
	# ipdb.set_trace()
	# print("hello")

def calculate_policy_with_volume(actions, lam):
	# print("Using the outdated volume model, please use volume_2")
	assert False, "Using the outdated volume model, please use volume_2"
	# total_volume = sum(
	# 	(action.children_density()/np.exp(action.log_prob) for action in actions)
	# )
	# if len(actions) == 1:
	# 	return np.array([1.0], dtype=np.float32)
	# Q_mean = np.mean([action.Q for action in actions])
	# Q_std = np.std([action.Q for action in actions])
	# assert min([action.Q for action in actions]) > -1000
	# def normalize(action):
	# 	return action.Q#(action.Q - Q_mean)/(Q_std+0.01)
	# single_prob = lambda action, alpha: lam/(alpha - normalize(action))*(
	# 	action.children_density()/(total_volume*np.exp(action.log_prob)))
	# total_prob = lambda alpha: sum([single_prob(action, alpha) for action in actions])-1

	# try:
	# 	alpha_upper_bound = max([normalize(action) + lam for action in actions])
	# 	alpha_lower_bound = max([normalize(action) + lam*(
	# 		action.children_density()/(total_volume*np.exp(action.log_prob))) for action in actions]) 
	# 	prob_low = total_prob(alpha_upper_bound)
	# 	prob_high = total_prob(alpha_lower_bound)
	# 	assert prob_high < 1000000000000000000
	# except: 
	# 	#policy is not efficiently normalizable
	# 	#However if this is the case it is almost certainly dominated by a single action
	# 	action_index = argmax(np.stack([single_prob(action, alpha_lower_bound) for action in actions]))
	# 	# return actions[action_index]
	# 	epsilon = 0.01
	# 	policy = np.array([
	# 			epsilon if i == action_index  else 1
	# 			for i in range(len(actions))
	# 		])
	# 	policy /= policy.sum()
	# 	return policy
	# 	# import ipdb
	# 	# ipdb.set_trace()

	# # if prob_high > 1000:
	# # 	alpha_lower_bound += 0.0000001
	# # 	prob_high = total_prob(alpha_lower_bound)
	# try: 
	# 	if prob_low <= 0.001 and prob_low >= -0.001: 
	# 		alpha = alpha_upper_bound
	# 	elif prob_high <= 0.001 and prob_high >= -0.001: 
	# 		alpha = alpha_lower_bound
	# 	else:
	# 		if prob_low > 0:
	# 			alpha = alpha_upper_bound
	# 		elif prob_high < 0:
	# 			alpha = alpha_lower_bound
	# 		else: 
	# 			# alpha = search(total_prob, 
	# 			# 	alpha_lower_bound, alpha_upper_bound, depth=15)
	# 			alpha = search(total_prob, 
	# 				alpha_lower_bound, alpha_upper_bound, depth=25)
	# 	est_total_prob = total_prob(alpha) + 1
	# 	if prob_high**2 < (est_total_prob-1)**2:
	# 		est_total_prob = prob_high + 1
	# 		alpha = alpha_lower_bound
	# 	elif prob_low**2 < (est_total_prob-1)**2:
	# 		est_total_prob = prob_low + 1
	# 		alpha = alpha_upper_bound
	# 	assert est_total_prob < 2
	# 	assert est_total_prob > 0.5
	# except: 
	# 	import ipdb
	# 	ipdb.set_trace()
	# policy = np.concatenate([(single_prob(action, alpha)/est_total_prob).reshape(-1) for action in actions])
	# if type(policy) == list:
	# 	import ipdb
	# 	ipdb.set_trace()
	# return policy
	# print(alpha)
	# print(total_prob(alpha))
	# # import ipdb
	# # ipdb.set_trace()
	# print("hello")


def calculate_policy_with_volume_2(node, n, gamma_d, with_pw=False, global_total_volume=1, 
	lambda_coeff = 100):
	# assert False
	Force_local = True
	# Force_local = False
	# Force_no_volume = True
	Force_no_volume = False
	if Force_no_volume:
		action_weight = 1
	else: 
		action_weight = 0.0001#5
	if Force_local: 		
		n = node.n
		global_total_volume = node.children_density()
		gamma_d=1
	try: 
		assert type(node) is KDNodeContinuous
		assert type(n) is int
	except: 
		import ipdb
		ipdb.set_trace()

	# lam = 10*n**(-0.5) 
	lam = lambda_coeff*n**(-0.5)
	# lam = 10 
	# lam = n**(-0.1) 
	# lam = 1
	if with_pw:
		actions = node.child_actions + [PWAction(node)]
	else: 
		actions = node.child_actions
	# ucb = lambda action: gamma_d/len(actions) + 1/128*action.children_density()
	if Force_no_volume:
		ucb = lambda action: gamma_d/len(actions)
	elif Force_local: 
		ucb = lambda action: (1/len(actions)*action_weight + action.children_density()/node.children_density())
	else:
		ucb = lambda action: (node.base_prob/node.prob)*(gamma_d/len(actions)*action_weight + action.children_density()/global_total_volume)
		# ucb = lambda action: (node.n/node.prob)*(gamma_d/len(actions)*action_weight 
		# 	+ action.children_unweighted_density()/global_total_volume)
		# ucb = lambda action: (gamma_d/len(actions)*action_weight + (node.base_prob/node.prob)*action.children_density()/global_total_volume)
		# ucb = lambda action: (1/len(actions)*action_weight + (node.base_prob/node.prob)*action.children_density()/global_total_volume)
	# ucb = lambda action: (node.base_prob/node.prob)*(action.children_density()/global_total_volume)
	total_volume = sum((ucb(action) for action in actions))
	if len(actions) == 1:
		return np.array([1.0], dtype=np.float32)
	Q_mean = np.mean([action.Q for action in actions])
	Q_std = np.std([action.Q for action in actions])
	assert min([action.Q for action in actions]) > -1000

	def normalize(action):
		return gamma_d*action.Q#(action.Q - Q_mean)/(Q_std+0.01)
	
	# ucb = lambda action: (node.base_prob)*(gamma_d/len(actions)*action_weight + action.children_density()/global_total_volume)
	# action_weight = 0#1.05
	# ucb = lambda action: (
	# 	# node.local_density()/(node.n+1)*gamma_d/len(actions)*action_weight 
	# 	node.local_density()*gamma_d/len(actions)*action_weight 
	# 	+ action.children_density()/global_total_volume
	# )
	# def normalize(action):
	# 	return node.prob*gamma_d*action.Q

	single_prob = lambda action, alpha: lam/(alpha - normalize(action))*ucb(action)/total_volume
	# single_prob = lambda action, alpha: 1/(alpha - normalize(action))*(ucb(action)/total_volume
	total_prob = lambda alpha: sum([single_prob(action, alpha) for action in actions])-1

	try:
		alpha_upper_bound = max([normalize(action) + lam for action in actions])
		# alpha_lower_bound = max([normalize(action) + lam*(
		# 	gamma_d/len(actions) + action.children_density())/total_volume for action in actions]) 
		alpha_lower_bound = max([normalize(action) + lam*ucb(action)/total_volume
			 for action in actions]) 
		prob_low = total_prob(alpha_upper_bound)
		prob_high = total_prob(alpha_lower_bound)
		assert prob_high < 1000000000000000000
	except: 
		# import ipdb
		# ipdb.set_trace()
		#policy is not efficiently normalizable
		#However if this is the case it is almost certainly dominated by a single action
		action_index = argmax(np.stack(
			[single_prob(action, alpha_lower_bound) for action in actions]
		))
		# return actions[action_index]
		epsilon = 0.01
		policy = np.array([
				epsilon if i == action_index  else 1
				for i in range(len(actions))
			])
		policy /= policy.sum()
		return policy
		# import ipdb
		# ipdb.set_trace()

	# if prob_high > 1000:
	# 	alpha_lower_bound += 0.0000001
	# 	prob_high = total_prob(alpha_lower_bound)
	try: 
		alpha, est_total_prob = search(total_prob, alpha_lower_bound, alpha_upper_bound, depth=10)
		# assert est_total_prob < 2
		# assert est_total_prob > 0.5
	except: 
		import ipdb
		ipdb.set_trace()
	policy = np.concatenate([
		(single_prob(action, alpha)/est_total_prob).reshape(-1) 
		for action in actions
	])
	if type(policy) == list:
		import ipdb
		ipdb.set_trace()

	# alt_policy = calculate_policy(actions, node.n**(-0.5)).reshape(-1)
	# if np.sum((policy - alt_policy)**2) > 0.01:
	# 	import ipdb
	# 	ipdb.set_trace()

	return policy
	print(alpha)
	print(total_prob(alpha))
	# import ipdb
	# ipdb.set_trace()
	print("hello")


def calculate_policy_with_volume_without_volume(node, n, gamma_d, with_pw=False, global_total_volume=1, 
	lambda_coeff = 100):
	# assert False
	Force_local = True
	# Force_local = False
	# Force_no_volume = True
	Force_no_volume = False
	if Force_no_volume:
		action_weight = 1
	else: 
		action_weight = 0.0001#5
	if Force_local: 		
		n = node.n
		global_total_volume = node.children_density()
		gamma_d=1
	try: 
		assert type(node) is KDNodeContinuous
		assert type(n) is int
	except: 
		import ipdb
		ipdb.set_trace()

	lam = lambda_coeff*n**(-0.5)
	if with_pw:
		actions = node.child_actions + [PWAction(node)]
	else: 
		actions = node.child_actions
	if Force_no_volume:
		ucb = lambda action: gamma_d/len(actions)
	elif Force_local: 
		ucb = lambda action: (1/len(actions)*action_weight + action.children_density()/node.children_density())
	else:
		ucb = lambda action: (node.base_prob/node.prob)*(gamma_d/len(actions)*action_weight + action.children_density()/global_total_volume)

	total_volume = sum((ucb(action) for action in actions))
	if len(actions) == 1:
		return np.array([1.0], dtype=np.float32)
	Q_mean = np.mean([action.Q for action in actions])
	Q_std = np.std([action.Q for action in actions])
	assert min([action.Q for action in actions]) > -1000

	def normalize(action):
		return gamma_d*action.Q

	single_prob = lambda action, alpha: lam/(alpha - normalize(action))*ucb(action)/total_volume
	total_prob = lambda alpha: sum([single_prob(action, alpha) for action in actions])-1

	try:
		alpha_upper_bound = max([normalize(action) + lam for action in actions])
		alpha_lower_bound = max([normalize(action) + lam*ucb(action)/total_volume
			 for action in actions]) 
		prob_low = total_prob(alpha_upper_bound)
		prob_high = total_prob(alpha_lower_bound)
		assert prob_high < 1000000000000000000
	except: 
		action_index = argmax(np.stack(
			[single_prob(action, alpha_lower_bound) for action in actions]
		))
		epsilon = 0.01
		policy = np.array([
				epsilon if i == action_index  else 1
				for i in range(len(actions))
			])
		policy /= policy.sum()
		return policy
	try: 
		alpha, est_total_prob = search(total_prob, alpha_lower_bound, alpha_upper_bound, depth=10)
	except: 
		import ipdb
		ipdb.set_trace()
	policy = np.concatenate([
		(single_prob(action, alpha)/est_total_prob).reshape(-1) 
		for action in actions
	])
	if type(policy) == list:
		import ipdb
		ipdb.set_trace()

	return policy


def calculate_one_shot_policy(node, global_n, epoch, gamma_d, with_pw=False, global_total_volume=1):
	# assert False
	# Force_local = True
	Force_local = False
	# Force_no_volume = True
	Force_no_volume = False
	if Force_no_volume:
		action_weight = 1
	else: 
		action_weight = 0.0#5
	if Force_local: 		
		n = node.n
		global_total_volume = node.children_density()
		gamma_d=1
	else: 
		n = global_n
	try: 
		assert type(node) is KDNodeContinuous
		assert type(n) is int
	except: 
		import ipdb
		ipdb.set_trace()

	# lam = 10*(1+epoch)**0.5*n**(-0.5) 
	lam = 1/(1+epoch)**(-0.5)#*n**(-0.5) 
	# lam = n**(-0.1) 
	# lam = 1
	if with_pw:
		# actions = node.child_actions + [PWAction(node)]
		actions = node.child_actions + [PWAction(node, forced_volume=0.02)]
	else: 
		actions = node.child_actions
	# ucb = lambda action: gamma_d/len(actions) + 1/128*action.children_density()
	if Force_no_volume:
		ucb = lambda action: gamma_d/len(actions)
	elif Force_local: 
		ucb = lambda action: (1/len(actions)*action_weight + action.children_unweighted_density()/node.children_unweighted_density())
	else:
		# ucb = lambda action: (node.base_prob/node.prob)*(gamma_d/len(actions)*action_weight + action.children_density()/global_total_volume)
		ucb = lambda action: ((node.n+1)**(-0.5)/len(actions)*action_weight + 
			lam/node.prob*action.children_unweighted_density()/global_total_volume)
		# ucb = lambda action: ((node.n+1)**(-0.5)*action_weight + 
		# 	lam/node.prob*action.children_density()/global_total_volume)
		# ucb = lambda action: (node.n/node.prob)*(gamma_d/len(actions)*action_weight 
		# 	+ action.children_unweighted_density()/global_total_volume)
		# ucb = lambda action: (gamma_d/len(actions)*action_weight + (node.base_prob/node.prob)*action.children_density()/global_total_volume)
		# ucb = lambda action: (1/len(actions)*action_weight + (node.base_prob/node.prob)*action.children_density()/global_total_volume)
	# ucb = lambda action: (node.base_prob/node.prob)*(action.children_density()/global_total_volume)
	total_volume = sum((ucb(action) for action in actions))
	if len(actions) == 1:
		return np.array([1.0], dtype=np.float32)
	Q_mean = np.mean([action.Q for action in actions])
	Q_std = np.std([action.Q for action in actions])
	assert min([action.Q for action in actions]) > -1000

	def normalize(action):
		return node.base_prob*action.Q#(action.Q - Q_mean)/(Q_std+0.01)
	# def normalize(action):
	# 	if type(action) == PWAction:
	# 		return action.Q
	# 	else:
	# 		return gamma_d*action.kernel_value/10#(action.Q - Q_mean)/(Q_std+0.01)

	single_prob = lambda action, alpha: 1/(alpha - normalize(action))*ucb(action)/total_volume
	# single_prob = lambda action, alpha: 1/(alpha - normalize(action))*(ucb(action)/total_volume
	total_prob = lambda alpha: sum([single_prob(action, alpha) for action in actions])-1

	try:
		alpha_upper_bound = max([normalize(action) + lam for action in actions])
		# alpha_lower_bound = max([normalize(action) + lam*(
		# 	gamma_d/len(actions) + action.children_density())/total_volume for action in actions]) 
		alpha_lower_bound = max([normalize(action) + lam*ucb(action)/total_volume
			 for action in actions]) 
		prob_low = total_prob(alpha_upper_bound)
		prob_high = total_prob(alpha_lower_bound)
		assert prob_high < 1000000000000000000
	except: 
		# import ipdb
		# ipdb.set_trace()
		#policy is not efficiently normalizable
		#However if this is the case it is almost certainly dominated by a single action
		action_index = argmax(np.stack(
			[single_prob(action, alpha_lower_bound) for action in actions]
		))
		# return actions[action_index]
		epsilon = 0.01
		policy = np.array([
				epsilon if i == action_index  else 1
				for i in range(len(actions))
			])
		policy /= policy.sum()
		return policy
		# import ipdb
		# ipdb.set_trace()

	# if prob_high > 1000:
	# 	alpha_lower_bound += 0.0000001
	# 	prob_high = total_prob(alpha_lower_bound)
	try: 
		alpha, est_total_prob = search(total_prob, alpha_lower_bound, alpha_upper_bound, depth=10)
		# assert est_total_prob < 2
		# assert est_total_prob > 0.5
	except: 
		import ipdb
		ipdb.set_trace()
	policy = np.concatenate([
		(single_prob(action, alpha)/est_total_prob).reshape(-1) 
		for action in actions
	])
	if type(policy) == list:
		import ipdb
		ipdb.set_trace()

	# alt_policy = calculate_policy(actions, node.n**(-0.5)).reshape(-1)
	# if np.sum((policy - alt_policy)**2) > 0.01:
	# 	import ipdb
	# 	ipdb.set_trace()

	return policy
	print(alpha)
	print(total_prob(alpha))
	# import ipdb
	# ipdb.set_trace()
	print("hello")

if __name__ == '__main__':
	a1 = Object()
	a1.total_volume = 1
	a1.value = 1

	a2 = Object()
	a2.total_volume = 0.1
	a2.value = 2
	actions = [a1, a2]

	lam = 1
	
	calculate_policy(actions, lam)