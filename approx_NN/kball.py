
import numpy as np
np.random.seed(0)
X = np.random.random((10, 3))  # 10 points in 3 dimensions
tree = BallTree(X, leaf_size=2)

K = 4
query_matrix = np.array([.5,.5,.5]).reshape(1,3)
dist, ind = tree.query(query_matrix, k=K)

thresh = .3
#rank
[ind.squeeze()[i] for i in range(K) if dist.squeeze()[i] < thresh]

#pickle tree
s = pickle.dumps(tree)
tree_copy = pickle.loads(s)

#query for neighbors in radius
tree.query_radius(X[:1], r=0.3, count_only=True)
