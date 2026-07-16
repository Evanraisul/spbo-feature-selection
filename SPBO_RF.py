import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from final_spbo_rf import jfs   # change this to switch algorithm 
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

#from sklearn.metrics import precision_score
#from sklearn.metrics import recall_score
#from sklearn.metrics import f1_score
#from sklearn import metrics
#from sklearn.preprocessing import label_binarize
#from sklearn.multiclass import OneVsRestClassifier
#from sklearn.metrics import roc_curve, auc
#from sklearn.metrics import confusion_matrix
#import seaborn as sns


# load data
data  = pd.read_csv('PSO/Final.csv')
data  = data.values
feat  = np.asarray(data[:, 0:41])
label = np.asarray(data[:, 41])

X=feat
y=label
# split data into train & validation (70 -- 30)
xtrain, xtest, ytrain, ytest = train_test_split(feat, label, test_size=0.3)
fold = {'xt':xtrain, 'yt':ytrain, 'xv':xtest, 'yv':ytest}



#----------------------- Evan ------------------------------------------------
x_train=feat
y_train=label

mdl       = RandomForestClassifier(n_estimators=20, random_state=42)
mdl.fit(x_train, y_train)


#-------------------------------------------------------------

# accuracy
# Make predictions on the testing set
y_pred = mdl.predict(xtest)

# Calculate the accuracy of the model
Acc = accuracy_score(ytest, y_pred)
print("Total Accuracy:", 100 * Acc)

#------------------------------------------------------------------------------------
# parameter
k    = 5     # k-value in KNN
#N    = 7    # number of particles
#T    = 10   # maximum number of iterations

#c1  = 2         # cognitive factor
#c2  = 2         # social factor 
#w   = 0.9       # inertia weight

#population Size
N=80

#min Values
mn=(-5,-5)

#max Values
mx=(5,5)

#Generation/ Iteration
T=10

opts = {'k':k,'fold':fold, 'N':N, 'T':T, 'mn':mn, 'mx':mx}

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

mdl       = RandomForestClassifier(n_estimators=10, random_state=42)
mdl.fit(x_train, y_train)

##------------------------------------------------------------------



# Create the confusion matrix

###_--------------------------------------------------------
#------------------------------------------------------------------------------
#ROC Curve

#--------------------------------------------------------------------------------
# Accuracy

#Train
x_pred    = mdl.predict(x_train)
Acc       = np.sum(y_train == x_pred)  / num_train
print("SubSet Train Accuracy:", 100 * Acc)

print('Training')
print(classification_report(y_train, x_pred))


print()
print()

#Test
y_pred    = mdl.predict(x_valid)
Acc       = np.sum(y_valid == y_pred)  / num_valid
print("SubSet Test Accuracy:", 100 * Acc)

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
ax.set_title('SPBO')
ax.grid()
plt.show()

