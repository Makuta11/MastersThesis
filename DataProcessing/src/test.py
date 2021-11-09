#%%
import numpy as np

from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier

X, y = make_multilabel_classification(n_classes=3, random_state=0)

clf = MultiOutputClassifier(KNeighborsClassifier()).fit(X, y)

clf.predict(X[-2:])


# %%
