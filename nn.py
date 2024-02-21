import subprocess as sp
import numpy as np
import mltools as ml
from scipy.optimize import minimize
from scipy.io import loadmat
import sys
import time

sp.call('cls', shell=True) # clear the screen

#Calling available datasets from OS:
print('Available datasets:\n')
sp.call('dir .\datasets', shell=True)

print('Loading and Visualizing Data ...\n')

dataSetAvailable = 0
while (dataSetAvailable == 0):
    try:
        datasetFile = input('\nType the file name of the dataset: ')
        X, y = ml.load(datasetFile)
        dataSetAvailable = 1
    except KeyboardInterrupt:
        sys.exit()
    except:
        print('Dataset not available.')

#Random selection of 100 data points to display:
rows = X.shape[0]
randomRows = np.random.permutation(rows)
selection = X[randomRows[0:100]]

display = 1 #Activate visualization by default 
try:
    ml.displayData(selection)
except:
    display = 0
    print('Non-displayable data')

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
#Cost function to optimize. (lam and lambda are not the same):
function = lambda t: ml.nnCostFunction(t, inputLayerSize, hiddenLayerSize, labels, X, y, lam)

#optimization:
optim = minimize(fun = function, x0 = initNNParams, method = 'TNC', jac = True, options = {'disp': True})
cost = optim.fun
nn_params = optim.x

#roll parameters:
theta1 = nn_params[:hiddenLayerSize * (inputLayerSize + 1)].reshape(hiddenLayerSize, (inputLayerSize + 1))
theta2 = nn_params[hiddenLayerSize * (inputLayerSize + 1):].reshape(labelsSize, (hiddenLayerSize + 1))

#Visualizing Neural Network:

if (display == 1): ml.displayData(theta1[:,1:])

#Caculating Accuracy:
p = ml.predict_nn(theta1, theta2, X) # predictions of the model.
p = np.asarray(labels)[p].reshape(-1,1) #Formatting p to compare with y
accuracy = np.mean((p == y)*1)*100
print('\nTraining Set Accuracy: ', accuracy, '\n')

#validating model:

print('\nValidating Model...\n')

for i in range(rows):
    try:
        Xi = X[randomRows[i],:].reshape(1,-1) #Selecting sample
        if (display == 1): 
            ml.displayData(Xi)
            print('\033[F\033[F')
        else: 
            print('\033[F\033[F\033[F\rNon-displayable data. Parameters are: \n', Xi, end='',flush=True)
        p = ml.predict_nn(theta1, theta2, Xi)
        print('\rData objective is: ', labels[p[0]], '. \n\'ctrl-c\' to finish', end='',flush=True)
        time.sleep(1)
    except KeyboardInterrupt:
        sys.exit()
