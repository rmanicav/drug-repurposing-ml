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
from sklearn.preprocessing import label_binarize

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
  

dtree_model = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=3, min_samples_leaf=5)  
dtree_model =dtree_model.fit(X_train, y_train) 
y_pred = dtree_model.predict(X_test) 


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
print("Average Roc:",roc_auc["micro"])

fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=0)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# Print ROC curve
plt.plot(fpr,tpr,label="Decision Tree")
print("DT-complete")


#####################################KNN###################################################

from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, y_train)   
y_pred = knn.predict(X_test) 
fpr1, tpr1, thresholds = roc_curve(y_test, y_pred, pos_label=0)

# Print ROC curve
plt.plot(fpr1,tpr1,label="KNN")
print("KNN complete")

################################################################Naive
from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB() 
gnb.fit(X_train, y_train) 
  
# making predictions on the testing set 
y_pred = gnb.predict(X_test) 
fpr2, tpr2, thresholds = roc_curve(y_test, y_pred, pos_label=0)
# Print ROC curve
plt.plot(fpr2,tpr2,label="Naive Bayes")


##########################################RF
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier() 
classifier=classifier.fit(X_train,y_train) 
y_pred = classifier.predict(X_test) 
# Print ROC curve
fpr3, tpr3, thresholds = roc_curve(y_test, y_pred, pos_label=0)
plt.plot(fpr3,tpr3,label="Random Forest")

#########################SVM
from sklearn.svm import SVC 
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train) 
y_pred = svm_model_linear.predict(X_test)   
fpr4, tpr4, thresholds = roc_curve(y_test, y_pred, pos_label=0)
plt.plot(fpr4,tpr4,label="SVM")
plt.savefig("All_ROC.png",bbox_inches='tight')

plt.savefig("All_ROC.png",bbox_inches='tight')

