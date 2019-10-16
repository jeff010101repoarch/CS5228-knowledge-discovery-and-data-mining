import numpy as np
import os

DATATYPE = np.float32

class MyCluster():
	'''
	The cluster object used in clustering.
	Just use the properties and methods.
	You can also change anything if you need.
	-------
	Properties:
	items: list
		list of the id of items in the cluster
		E.g. the items are stored in the list '_items', then [1, 2, 4] reprents the cluster of _items[1], _items[2], and _items[4]
	size: int
		number of current items in the cluster
	used: bool
		If True, the cluster remains; if False, the cluster has already been merged into another cluster
	-------
	Methods:
	append(self, item):
		Add a single item into the cluster. Input is the id of the item.
	extend(self, items):
		Add a list of items into the cluster. Input is a list of ids of the items.
	release(self):
		Release the cluster when it is merged into another cluster.
	'''
	def __init__(self):
		self._items = []
		self._used = True

	@property
	def size(self):
		return len(self._items)

	@property
	def used(self):
		return self._used

	@property
	def items(self):
		return self._items

	def append(self, item):
		self._items.extend([item])
	
	def extend(self, items):
		self._items.extend(items)
		
	def release(self):
		self._used = False
		self._items = []


def euclidean_distance(p, q):
	'''
	Method to calculate euclidean distance between two points.
	'''
	return np.sqrt(np.sum((p - q)**2))


def single_linkage(points, clusters, p, q):
	'''
	TO DO
	Compute the proximity of single linkage between two clusters.
	Just use euclidean_distance(points[x], points[y]) to calculate the distance between x and y
	-------
	Inputs:
	points: list
		the list of items
	clusters: list
		the list of clusters
	p, q: int
		the id of input clusters
	-------
	Returns: a float, the proximity of single linkage between clusters[p] and clusters[q]
	'''
	pass


def complete_linkage(points, clusters, p, q):
	'''
	TO DO
	Compute the proximity of complete linkage between two clusters.
	Just use euclidean_distance(points[x], points[y]) to calculate the distance between x and y
	-------
	Inputs:
	points: list
		the list of items
	clusters: list
		the list of clusters
	p, q: int
		the id of input clusters
	-------
	Returns: a float, the proximity of complete linkage between clusters[p] and clusters[q]
	'''
	pass


class MyAgglomerativeClustering():
	'''
	Agglomerative Clustering
	Recursively merges the pair of clusters that minimally increases the proximity of a given linkage.
	Use the attributes to implement the methods.
	You can change anything if you need.
	-------
	Attributes:
	_n_clusters: int
		the number of clusters that the algorithm should find. Default=1.
	_n_items: int
		the number of items
	_items: list
		the list of items
	_clusters: list
		the list of MyCluster
		HINT: You don't need to remove a cluster even it is already merged into another one. Instead, just use its release() method.
	_proximity_matrix: array[2 * _n_items, 2 * _n_items]
		the proximity between each pair of cluster.
		e.g. the proximity between _clusters[i] and _clusters[j] is _proximity_matrix[i, j]
	_linkage: string
		to tell the algorithm which kind of linkage is used to define the proximity
	_linkage_func: callable
		You should call _linkage_func in your implementation to compute the proximity instead of calling single_linkage or complete_linkage.
		It will be automatically navigated to the corresponding function based on _linkage.
		Note the parameters of _linkage_func should be the same as single_linkage and complete_linkage.
		E.g. call self._linkage_func(points, clusters, p, q) in this class.
	linkage_choices: dict
		to map _linkage to _linkage_func
	-------
	Methods:
	
	'''
	def __init__(self, n_clusters=1, linkage='single'):
		'''
		Construction
		You don't need to change this method.
		'''
		self._n_clusters = n_clusters
		self._n_items = None
		self._items = None
		self._clusters = None
		self._proximity_matrix = None
		self._history = None
		self._linkage = linkage
		linkage_choices = {'single': single_linkage,
						   'complete': complete_linkage}
		self._linkage_func = linkage_choices[self._linkage]


	def init_cluster(self, inputs):
		'''
		Initialization
		You don't need to change this method.
		'''
		list = []
		self._n_items = len(inputs)
		for i, item in enumerate(inputs):
			new_cluster = MyCluster()
			new_cluster.append(i)
			list.append(new_cluster)

		self._proximity_matrix = np.zeros((2 * self._n_items, 2 * self._n_items), dtype=DATATYPE)
		return inputs, list

	
	def find_clusters_to_merge(self):
		'''
		TO DO
		Select two clusters with the smallest proximity.
		HINT: You may need to check 'self._proximity_matrix' to find the smallest proximity.
		-------
		Inputs:
		None
		-------
		Returns:
		p, q: int, int
			the id of two clusters that should be merged
		'''
		pass


	def merge_cluster(self, p, q):
		'''
		TO DO
		Merge the pair of clusters (p, q).
		-------
		Inputs:
		p: int
			the cluster id
		q: int
			the cluster id
		-------
		Returns: int
			the id of the new cluster by merging p and q.
		'''
		pass


	def update_proximity(self, new_cluster):
		'''
		TO DO
		Update the proximity matrix using the new cluster.
		Call self._linkage_func(points, clusters, p, q) to compute the proximity as mentioned above.
		-------
		Inputs:
		new_cluster: int
			the id of the new cluster
		-------
		Returns:
			None
		'''
		pass


	def fit(self, X):
		'''
		Workflow of clustering.
		You don't need to change this method as it is already done.
		Just implement find_clusters_to_merge(), merge_cluster(), update_proximity(), which are used in this method.
		-------
		Returns: self._history that will be compared with the correct solution.
		'''
		# initialize the item list and the cluster list
		self._items, self._clusters = self.init_cluster(X)
		self._history = []

		# initialize the proximity matrix by euclidean distances between each pair of items
		for i in range(len(self._clusters) - 1):
			for j in range(i + 1, len(self._clusters)):
				x = self._clusters[i]
				y = self._clusters[j]
				self._proximity_matrix[i, j] = euclidean_distance(self._items[x._items[0]], self._items[y._items[0]])

		# loop until only self._n_clusters clusters remain
		while len(self._clusters) < self._n_items * 2 - self._n_clusters:
			# merge the two closest clusters
			p, q = self.find_clusters_to_merge()
			new_cluster = self.merge_cluster(p, q)
			# update the proximity matrix
			self.update_proximity(new_cluster)

			# record the merged pair
			self._history.extend([[p, q]])

		return self._history


def compare_solution(history, filename):
	answer = np.genfromtxt(filename, delimiter=',', dtype=np.int32).tolist()

	if len(history) != len(answer):
		return False

	for a, b in zip(history, answer):
		if len(a) != len(b):
			return False
		for x in b:
			if x not in a:
				return False

	return True

# Test
if __name__=='__main__':
	for i in range(1):
		testdata = np.genfromtxt("Test_data" + os.sep + "clustering_" + str(i) +".csv", delimiter=',', dtype=DATATYPE)
		
		single_linkage_clustering = MyAgglomerativeClustering(n_clusters=1, linkage='single')
		complete_linkage_clustering = MyAgglomerativeClustering(n_clusters=1, linkage='complete')
		
		history_single = single_linkage_clustering.fit(testdata)
		file_single_answer = "Test_data" + os.sep + "clustering_single_" + str(i) +".csv"

		history_complete = complete_linkage_clustering.fit(testdata)
		file_complete_answer = "Test_data" + os.sep + "clustering_complete_" + str(i) +".csv"

		print(compare_solution(history_single, file_single_answer), compare_solution(history_complete, file_complete_answer))
