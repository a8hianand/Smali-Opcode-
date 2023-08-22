
import pandas as pd
import os
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split


df = pd.read_csv('C:/Users/Abhishek Anand/Desktop/extra/Drebin_smali_opcodes/Drebin_smali_opcode_count_with_family_complete.csv')

df=df.drop(df.columns[0], axis=1)

x=df.drop('Family', axis=1)

y=df['Family']

from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()
label_encoder.fit(y)
y=label_encoder.transform(y)
Family=label_encoder.classes_


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3,random_state=0)



# # Random forest classifer

import resultpresentation as result
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing


classes_list = ['BaseBridge', 'DroidKungFu','FakeDoc', 'FakeInstaller', 'Geinimi', 'GinMaster', 'Iconosys', 'Kmin', 'Opfake', 'Plankton']


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100,max_depth=None,min_samples_split=2, random_state=0)
rf.fit(X_train,y_train)
y_pred_rf = rf.predict(X_test)
print("random")
print(metrics.classification_report(y_test, y_pred_rf))
print("Accuracy score =", accuracy_score(y_test, y_pred_rf))
plt.clf()
cnf_matrix = metrics.confusion_matrix(y_test, y_pred_rf)
result.plot_confusion_matrix(cnf_matrix, classes = classes_list, normalize = True)
plt.savefig("click_confusion_matrix_rf_bal.pdf", format = 'pdf', dpi =1000)
plt.clf()

### Label Encoding
encoder = preprocessing.LabelEncoder()
encoder.fit(y_train)
y_train = encoder.transform(y_train)
y_test = encoder.transform(y_test)

result.plot_AUC_ROC(y_test, y_pred_rf)
plt.savefig("click_ROC_rf_bal.pdf", format = 'pdf', dpi =1000)

