# -*- coding: utf-8 -*-
"""Meningitis Virus Detection.ipynb

## **Understanding the Data**
"""

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from math import sqrt
import warnings
warnings.filterwarnings('ignore')

"""#### Read Dataset"""

# Reading the Data
path1 = '/content/drive/My Drive/AB Project Mid/Meningtins Classification.xlsx'
df1 = pd.read_excel(path1, sheet_name='HEALTHY')

path2 = '/content/drive/My Drive/AB Project Mid/Meningtins Classification.xlsx'
df2 = pd.read_excel(path2, sheet_name='Ill BACTERIAL MENINGITIS')

path3 = '/content/drive/My Drive/AB Project Mid/Meningtins Classification.xlsx'
df3 = pd.read_excel(path3, sheet_name='Ill VIRUS MENINGITIS')

#0: healthy, 1: BACTERIAL, 2: VIRUS
df1['label'] = 0
df2['label'] = 1
df3['label'] = 2

# Concatenate 3 dataframes
df = pd.concat([df1, df2, df3], ignore_index=True)

"""### Data Cleaning"""

# Drop columns not needed
df = df.drop(columns=['Unnamed: 0', 'Leukocytes in CSL', 'Lactates in CSL', 'GUL / GUK', 'Procalcitonin', 'CRP'])

df.head()

# Check data type
df.info()

# Check statistics
df.describe()

# Check missing values
df.isnull().sum()

df = df.dropna()

"""## **Exploratory Data Analysis**"""

# Check balance data
df['label'].value_counts()

df['label'].value_counts().plot(kind='bar') #plot bar chart to check imbalance data

df['label'].value_counts().plot.pie(autopct='%.2f') #plot pie chart

# Plot histogram to represent of the distribution of data
fig = plt.figure(figsize = (16,10))
ax = fig.gca()
df.hist(ax = ax)

"""#### Correlation Matrix"""

df1 = df.copy().drop(['Color CSL', 'Look'], axis = 1)

cols = df1.columns

plt.figure(figsize = (12, 8), dpi = 100)

corr = df1.corr()

mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True


sns.heatmap(corr,
            mask = mask,
            cmap = 'autumn',
            vmax=.3,
            annot = True,
            linewidths = 0.5,
            fmt = ".2f",
            alpha = 0.7)

hfont = {'fontname':'monospace'}
plt.xticks(**hfont)
plt.yticks(**hfont)

plt.title('Correlation Matrix',
          family = 'monospace',
          fontsize = 20,
          weight = 'semibold',
          color = 'blue')

plt.show()

"""## **Data Preprocessing**"""

from sklearn import preprocessing

object_col = df.select_dtypes(include="object").columns

le = preprocessing.LabelEncoder()

for col in object_col:
    df[col] = le.fit_transform(df[col])

y = df['label']
X = df.drop('label', axis=1)

"""### Data Splitting"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6)

"""### Feature scaling"""

# Standardization
from sklearn.preprocessing import StandardScaler

scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)

"""## **Models**"""

random_forest_classifier = RandomForestClassifier(n_estimators = 300)
random_forest_classifier.fit(X_train, y_train)

svm = SVC()
svm.fit(X_train, y_train)

print("Accuracy score of random forest model is: ", random_forest_classifier.score(X_test, y_test))
print("Accuracy score of SVM model is: ", svm.score(X_test, y_test))

svm_pred = svm.predict(X_test)
cm_svm = confusion_matrix(y_test, svm_pred)

random_forest_classifier_pred = random_forest_classifier.predict(X_test)
cm_forest = confusion_matrix(y_test, random_forest_classifier_pred)

"""### Evaluation"""

print("MSE of random forest model is: ", mean_squared_error(y_test, random_forest_classifier_pred))
print("MSE of SVM model is: ", mean_squared_error(y_test, svm_pred))
print('\n')
print("RMSE of random forest model is: ", sqrt(mean_squared_error(y_test, random_forest_classifier_pred)))
print("RMSE of SVM model is: ", sqrt(mean_squared_error(y_test, svm_pred)))

"""#### Confusion matrix"""

target_names = ['HEALTHY', 'BACTERIAL', 'VIRUS']

ax = plt.subplot()
sns.heatmap(cm_forest, annot=True, fmt='g', ax=ax)
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix - Random Forest')
ax.xaxis.set_ticklabels(target_names)
ax.yaxis.set_ticklabels(target_names)

ax = plt.subplot()
sns.heatmap(cm_svm, annot=True, fmt='g', ax=ax)
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix SVM')
ax.xaxis.set_ticklabels(target_names)
ax.yaxis.set_ticklabels(target_names)

"""#### Classification Report"""

print("Report Model - Random Forest\n")
print(classification_report(random_forest_classifier_pred, y_test, target_names=target_names))

print("Report Model - SVM\n")
print(classification_report(svm_pred, y_test, target_names=target_names))
