from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
# load the iris datasets
dataset = datasets.load_iris()
# fit a logistic regression model to the data
l=len(dataset.data)
l80=l*.80

trainX=dataset.data[0:l80]
testX=dataset.data[l80:]
trainY=dataset.target[0:l80]
testY=dataset.target[l80:]

#print(trainX, trainY, testX, testY)

model = LogisticRegression()
model.fit(trainX, trainY)


    # make predictions
expected = testY
predicted = model.predict(testX)
# summarize the fit of the model
#print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
#print dataset