from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=20, n_classes=4,
                           n_informative=20, n_redundant=0,
                           random_state=0, shuffle=False, n_clusters_per_class=4)
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X, y)


print("hello")