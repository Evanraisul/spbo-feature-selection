import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from FS.pso_svm import jfs   # change this to switch algorithm 
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

import os
# Dataset path resolved from the repo layout instead of a hardcoded local
# Windows path, and one shared seed so every run uses the same split.
ROOT     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_CSV = os.path.join(ROOT, 'datasets', 'Final.csv')
SEED     = 42


# load data
data  = pd.read_csv(DATA_CSV)
data  = data.values

feat  = np.asarray(data[:, 0:41])
label = np.asarray(data[:, 41])

#######################--------------------------------------------------------
#PCA

pca = PCA(n_components=20, random_state=SEED)  # Choose the number of principal components to keep
X_pca = pca.fit_transform(feat)

pca_data = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2','PC3','PC4','PC5','PC6', 'PC7','PC8','PC9','PC10','PC11', 'PC12','PC13','PC14','PC15','PC16', 'PC17','PC18','PC19','PC20'])
feat=np.asarray(pca_data)

#//////------------------------------------------------------------------------

# split data into train & validation (70 -- 30)
xtrain, xtest, ytrain, ytest = train_test_split(feat, label, test_size=0.3,
                                                random_state=SEED)
fold = {'xt':xtrain, 'yt':ytrain, 'xv':xtest, 'yv':ytest}
#print(fold)

# parameter
#k    = 5     # k-value in KNN
N    = 200    # number of particles
T    = 100   # maximum number of iterations

c1  = 2         # cognitive factor
c2  = 2         # social factor 
w   = 0.9       # inertia weight
opts = {'fold':fold, 'N':N, 'T':T, 'w':w, 'c1':c1, 'c2':c2, 'seed':SEED}

# perform feature selection
fmdl = jfs(feat, label, opts)
sf   = fmdl['sf']

# model with selected features
num_train = np.size(xtrain, 0)
num_valid = np.size(xtest, 0)
x_train   = xtrain[:, sf]
y_train   = ytrain.reshape(num_train)  # Solve bug
x_valid   = xtest[:, sf]
y_valid   = ytest.reshape(num_valid)  # Solve bug

mdl       = SVC()
mdl.fit(x_train, y_train)

# accuracy
y_pred    = mdl.predict(x_valid)
Acc       = np.sum(y_valid == y_pred)  / num_valid
print("Accuracy:", 100 * Acc)

# Precision = TP/(TP + FP)
# Recall = TP/(TP + FN)
# F1 -score = 2 x [(Precision x Recall) / (Precision + Recall)]
# Support -> rows with a matching key in the main dataset

print(classification_report(y_valid, y_pred))

# number of selected features
num_feat = fmdl['nf']
print("No of Feature:", num_feat)

# plot convergence
curve   = fmdl['c']
curve   = curve.reshape(np.size(curve,1))
x       = np.arange(0, opts['T'], 1.0) + 1.0

fig, ax = plt.subplots()
ax.plot(x, curve, 'o-')
ax.set_xlabel('Number of Iterations')
ax.set_ylabel('Fitness')
ax.set_title('PSO')
ax.grid()
plt.show()

