import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from glob import glob

# def plot_labels_and_loss():


#     logistic = LogisticRegression()
#     logistic.fit(X=X, y=y)
    
#     # and plot the result
#     plt.figure(1, figsize=(4, 3))
#     #plt.logistic()
#     plt.scatter(X.ravel(), y, color='black', zorder=20)
#     #X_test = np.linspace(-5, 10, 300)

def render_labels(data_admit, data_reject, admitted, rejected):
    plt.figure(figsize=(6, 6))

    plt.scatter(data_admit[:,0], data_admit[:,1],
                c='b', marker='+', label='accept')
    plt.scatter(data_reject[:,0], data_reject[:,1],
                c='y', marker='o', label='retransmit')
    plt.xlabel('Joint Entropy Loss');
    plt.ylabel('Compression Level');
    #plt.axes().set_aspect('equal', 'datalim')
   
accept_dir = '/Users/hannahrae/src/mcam-artifacts/v2/labeler/data/supervised_labels/accept_test'
retransmit_dir = '/Users/hannahrae/src/mcam-artifacts/v2/labeler/data/supervised_labels/retransmit_test'

y_a = []
y_r = []
X_a = []
X_r = []
X = []
y = []

for example in glob(accept_dir + '/*'):
    ex_name = example.split('/')[-1][:-4]
    i, l, c = ex_name.split('_')
    x_i = [float(l), float(c)]
    y_a.append([1])
    X_a.append(x_i)
    X.append(x_i)
    y.append([1])

for example in glob(retransmit_dir + '/*'):
    ex_name = example.split('/')[-1][:-4]
    i, l, c = ex_name.split('_')
    x_i = [float(l), float(c)]
    y_r.append([0])
    X_r.append(x_i)
    X.append(x_i)
    y.append([0])

y_r = np.array(y_r)
y_a = np.array(y_a)
X_a = np.array(X_a)
X_r = np.array(X_r)
X = np.array(X)
y = np.array(y)

# Train the classifier
# classifier = OneVsRestClassifier(LogisticRegression(penalty='l1')).fit(X, y)
# print 'Coefficents: ', classifier.coef_
# print 'Intercept" ', classifier.intercept_

# coef = classifier.coef_
# intercept = classifier.intercept_

# print type(coef)
# print type(intercept)
# print coef.shape
# print intercept.shape

# Use a known classifier with hard-coded coefficients
coef = np.array([[-1.20730291, 0.15657625]])
intercept = np.array([[0.]])
print type(coef)
print type(intercept)
print coef.shape
print intercept.shape

ex1 = np.arange(8, 13, 0.0001)
ex2 = -(coef[:, 0] * ex1 + intercept[:, 0]) / coef[:,1]

render_labels(X_a, X_r, y_a, y_r)
plt.plot(ex1, ex2, color='r', label='decision boundary')
plt.title('Logistic Regression Classification on Test Set')
plt.legend()
plt.show()

# if __name__ == '__main__':
#     plot_labels_and_loss()