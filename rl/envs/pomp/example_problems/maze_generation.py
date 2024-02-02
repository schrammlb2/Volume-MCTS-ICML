import numpy as np
import ipdb

np.random.seed(1)

def with_p(chance):
	if np.random.rand() < chance: 
		return True
	else: 
		return False

def rand_offset(shape):
	if np.random.rand() < 0.5:
		sign = -1
	else: 
		sign = 1
	off = np.zeros(len(shape), dtype='int64')
	off[np.random.randint(len(shape))] = sign
	return off

def rand_cell(nodes, found): 
	loc = np.random.randint(nodes.shape)
	while nodes[tuple(loc)] != found: 
		loc = np.random.randint(nodes.shape)
	return loc

def valid_node(node, shape):
	shape_range = range(len(shape))	
	if all([node[i] >= 0 and node[i] < shape[i] for i in shape_range]):
		return True
	else: 
		return False

def rand_step(node, shape):
	while True:
		offset = rand_offset(shape)
		next_node = node + offset
		assert next_node.dtype == np.dtype('int64')
		if valid_node(next_node, shape):
			return next_node


def do_one_iter(nodes, edges, shape, log=False): 
	log=False
	prob_clear_path = 0.40
	s = rand_cell(nodes, found=True)
	if log: print(f"s: {s}")
	path = [tuple(s)]
	g = rand_cell(nodes, found=False)
	if log: print(f"g: {g}")
	while not (s==g).all():
		next_s = rand_step(s, shape)
		if log: print(f"next s: {next_s}")
		if nodes[tuple(next_s)] and (with_p(prob_clear_path)):
			path = [tuple(next_s)]
		elif tuple(next_s) in path:
			ind = path.index(tuple(next_s))
			path = path[:ind] + [tuple(next_s)]
		else:
			path.append(tuple(next_s))
		# ipdb.set_trace()
		s = next_s

	# if not nodes[tuple(path[0])]:
	# 	ipdb.set_trace()
	# if any([nodes[tuple(loc)] for loc in path[1:]]):
	# 	ipdb.set_trace()


	return path

def build_maze(n=4, d=2):
	shape = (n,)*d
	shape_range = range(len(shape))

	nodes = np.full(shape, False)
	edges = np.full(shape + shape, False)
	# nodes[rand_cell(nodes, found=False)] = True
	nodes[0,0] = True

	while not nodes.all():
		path = do_one_iter(nodes, edges, shape)
		# print("New Branch")
		for p in path: 
			nodes[tuple(p)] = True
			# print(tuple(p))

		for n1, n2 in zip(path[:-1], path[1:]): 
			edges[tuple(n1) + tuple(n2)] = True
			edges[tuple(n2) + tuple(n1)] = True



	# print(edges)
	return nodes, edges

# build_maze()