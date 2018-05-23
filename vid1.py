# https://github.com/llSourcell/gender_classification_challenge/blob/master/demo.py
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import neural_network
from sklearn import neighbors
from sklearn import naive_bayes
from sklearn import model_selection
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

clf = tree.DecisionTreeClassifier()

# CHALLENGE - create 3 more classifiers...

clf1 = naive_bayes.GaussianNB()
clf2 = neighbors.KNeighborsClassifier()
clf3 = neural_network.MLPClassifier()
clf4 = SVC()

models = []
models.append(('DecisionTree', clf))
models.append(('Gaussian', clf1))
models.append(('KNeighbors', clf2))
models.append(('MLP', clf3))
models.append(('SVM', clf4))

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39], [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

# CHALLENGE - ...and train them on our data
scores = []
for name, model in models:
    model.fit(X, Y)
    accuracy = accuracy_score(Y, model.predict(X))
    msg = '{} = {}'.format(accuracy, name)
    #print(msg)
    scores.append(msg)

scores.sort(reverse=True)
print(scores)

# CHALLENGE compare their reusults and print the best one!


"""
# https://machinelearningmastery.com/compare-machine-learning-algorithms-python-scikit-learn/

seed = 9
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
"""