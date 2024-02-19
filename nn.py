import subprocess as sp
import numpy as np
import mltools as ml
from scipy.optimize import minimize
from scipy.io import loadmat

sp.call('cls', shell=True) # clear the screen

print('Loading and Visualizing Data ...\n')

#Loading Training Data:
dataSet = loadmat('ex4data1.mat')
X = dataSet['X']
y = dataSet['y']

y[y==10]=0 #There are no images of '10' in dataset there are images of '0' instead, so 10 is equivalent to 0.

#Random selection of 100 data points to display:
rows = X.shape[0]
randomRows = np.random.permutation(rows)
selection = X[randomRows[0:100]]
ml.displayData(selection)

#defining neural network shape:
inputLayerSize  = X.shape[1]
hiddenLayerSize = 25
labels = list(set(y.flatten())) #a list from unique values from y
labelsSize = len(labels)

#Random initialization of Neural Network Parameters:

initTheta1 = ml.randInitWeights(inputLayerSize, hiddenLayerSize)
initTheta2 = ml.randInitWeights(hiddenLayerSize, labelsSize)
initNNParams = np.append(initTheta1.flatten(), initTheta2.flatten()) #unroll parameters

print('Training neural network...')
lam = 1 #lambda of cost function
#labels = list(set(y.flatten())) #a list from unique values from y

#Cost function to optimize. (lam and lambda are not the same):
function = lambda t: ml.nnCostFunction(t, inputLayerSize, hiddenLayerSize, labels, X, y, lam)

#optimization:
optim = minimize(fun = function, x0 = initNNParams, method = 'TNC', jac = True, options = {'disp': True})
cost = optim.fun
nn_params = optim.x

#roll parameters
theta1 = nn_params[:hiddenLayerSize * (inputLayerSize + 1)].reshape(hiddenLayerSize, (inputLayerSize + 1))
theta2 = nn_params[hiddenLayerSize * (inputLayerSize + 1):].reshape(labelsSize, (hiddenLayerSize + 1))

#Visualizing Neural Network
ml.displayData(theta1[:,1:])

#Caculating Accuracy
p = ml.predict_nn(theta1, theta2, X) # predictions of the model.
p = np.asarray(labels)[p].reshape(-1,1) #Formatting p to compare with y
accuracy = np.mean((p == y)*1)*100
print('\nTraining Set Accuracy: ', accuracy, '\n')

#validate model

print('\nValidating Model...\n')

for i in range(rows):
    try:
        Xi = X[randomRows[i],:].reshape(1,-1)
        ml.displayData(Xi)
        p = ml.predict_nn(theta1, theta2, Xi)
        print('\033[F\rImage is', p[0], '. \n\'ctrl-c\' to finish', end='',flush=True)
    except:
        break
