import sklearn as sk
import pandas as pd
import matplotlib.pyplot as plt

train_df = pd.read_csv("training_data.txt")
train_label_df = pd.read_csv("training_labels.txt")
test_df = pd.read_csv("testing_data.txt")
test_label_df = pd.read_csv("testing_labels.txt")

X_train = train_df.iloc[:,:22].values
y_train = train_label_df.iloc[:].values
y_train = y_train.reshape(155)
X_test = test_df.iloc[:,:22].values
y_test = test_label_df.iloc[:].values
y_test = y_test.reshape(40)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

graph_X = []
graph_Y = []
for i in range(2,22):
    logistic_pipe = Pipeline([('scl', StandardScaler()),('pca', PCA(n_components=i)),('clf',LogisticRegression(random_state=1))])
    logistic_pipe.fit(X_train,y_train)
    graph_X.append(i)
    graph_Y.append(logistic_pipe.score(X_test,y_test))

plt.xticks(graph_X)
plt.xlabel("Number of components")
plt.ylabel("Accuracy")
plt.plot(graph_X,graph_Y)
plt.show()

# Using K-Fold
import numpy as np
from sklearn.cross_validation import StratifiedKFold
kfold = StratifiedKFold(y=y_train.reshape(155),n_folds=5,random_state=1)
scores = []
logistic_pipe_kfold = Pipeline([('scl', StandardScaler()),('pca', PCA(n_components=5)),('clf',LogisticRegression(random_state=1))])

for k, (train, test) in enumerate(kfold):
    logistic_pipe_kfold.fit(X_train[train], y_train[train])
    score = logistic_pipe_kfold.score(X_train[test], y_train[test])
    scores.append(score)

for i,score in enumerate(scores):
    print("%s: %.3f" %(i+1,score))

np.mean(scores)
np.std(scores)

#Using SVM
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

pipe_svc = Pipeline([('scl',StandardScaler()),('clf',SVC(random_state=1))])
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'clf__C':param_range,'clf__kernel':['linear']},{'clf__C':param_range,'clf__gamma':param_range,'clf__kernel':['rbf']}]

gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)
svm_gs = gs.fit(X_train,y_train)
svm_gs.best_score_
svm_gs.best_params_
svm_gs.score(X_test,y_test)
y_pred = svm_gs.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)

import seaborn as sns
ax = sns.heatmap(confmat,annot=True)

plt.show()

#Using Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import confusion_matrix

dtree_gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0),param_grid=[{'max_depth': [1,2,3,4,5,6,7,None]}],scoring='accuracy',cv=5,n_jobs=-1)
scores = cross_val_score(dtree_gs,X_train,y_train,scoring='accuracy',cv=5)

print(scores)

np.mean(scores), np.std(scores)
dtree_gs.fit(X_train,y_train)
dtree_gs.best_params_
dtree_gs.score(X_test,y_test)

y_pred = dtree_gs.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
confmat

import seaborn as sns

sns.set()
ax = sns.heatmap(confmat, annot=True)
plt.show()

#Using KNN
from sklearn.neighbors import KNeighborsClassifier
pipe_knn = Pipeline([('scl',StandardScaler()),('clf',KNeighborsClassifier())])
param_grid = [{'clf__n_neighbors':[3,4,5,6,7,8,9,10],'clf__p':[1,2]}]

gs_knn = GridSearchCV(estimator=pipe_knn, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)
gs_knn.fit(X_train,y_train)
gs_knn.best_params_
gs_knn.best_score_

y_pred = gs_knn.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
gs_knn.score(X_test,y_test)
ax = sns.heatmap(confmat, annot=True)
plt.show()

#Using ANN
from keras.models import Sequentialuential
from keras.layers import Dense

model = Sequential()
model.add(Dense(12,input_dim=22,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
model.fit(X_train,y_train, epochs=100, batch_size=10)

scores = model.evaluate(X_test,y_test)
scores[1]

y_pred = model.predict(X_test)
y_pred = [round(y[0]) for y in y_pred]

confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
ax = sns.heatmap(confmat, annot=True)

plt.show()



if __name__ == '__main__':
    pass
