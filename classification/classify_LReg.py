#importing necessary libraries 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder 
import matplotlib.pyplot as plt 
from mlxtend.plotting import plot_confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import sys
from sklearn import tree 
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.utils.fixes import signature
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
#load drug & target
ds=pd.read_csv("Traini.csv")

#number of drugs =277, rows =33000, training = 25,000, test = 8000
# X -> features, y -> label
LE= LabelEncoder()
ds['Target'] = LE.fit_transform(ds['Target'])
X= pd.DataFrame(ds['Target'])
y= ds['Category']
# dividing X, y into train and test data 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state =4) 

from sklearn import linear_model  
reg = linear_model.LinearRegression() 
  
# train the model using the training sets 
reg = reg.fit(X_train, y_train) 
y_pred = reg.predict(X_test) 


print("F1 score")
print(f1_score(y_test, y_pred, average="macro"))
print("Precision Score")
print(precision_score(y_test, y_pred, average="macro"))
print("Recall")
print(recall_score(y_test, y_pred, average="macro"))  

  
# creating a confusion matrix 
cm = confusion_matrix(y_test, y_pred)   
print(cm)

#fig, ax = plot_confusion_matrix(conf_mat=cm,
#                                colorbar=True,
#                                show_absolute=False,
#                                show_normed=True)
#plt.savefig("cm.png",bbox_inches='tight')

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(0):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
print("Average Roc-General")
print(roc_auc["micro"])


