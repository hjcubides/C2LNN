import subprocess as sp
import numpy as np
import mltools as ml
from scipy.optimize import minimize
from scipy.io import loadmat
import sys
import time
import matplotlib.pyplot as plt

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

#Randomizing of the dataset to handle sorted cases.
rows = X.shape[0]
randomRows = np.random.permutation(rows)
X = X[randomRows]
y = y[randomRows]

#Selection of 100 data points to display:
selection = X[0:100] 
display = 1 #Activate visualization by default 

try:
    ml.displayData(selection, 'Random Selection')
except:
    display = 0
    print('Non-displayable data')

#Defining neural network shape:
inputLayerSize  = X.shape[1]
hiddenLayerSize = 25
labels = list(set(y.flatten())) #a list from unique values from y
labelsSize = len(labels)

#Defining the data set segmentation (70-15-15):
trainingLimit = round(len(X)*0.70) #Data limit to use for training
validationLimit = round(len(X)*0.85) #Upper limit of data to use for validation. Lower limit is the training limit.

#Random initialization of Neural Network Parameters:

initTheta1 = ml.randInitWeights(inputLayerSize, hiddenLayerSize)
initTheta2 = ml.randInitWeights(hiddenLayerSize, labelsSize)
initNNParams = np.append(initTheta1.flatten(), initTheta2.flatten()) #unroll parameters

print('Training neural network...')
lam = 1 #lambda of cost function
#Cost function to optimize. (lam and lambda are not the same):
function = lambda t: ml.nnCostFunction(t, inputLayerSize, hiddenLayerSize, labels, X[:trainingLimit], y[:trainingLimit], lam)

#optimization:
optim = minimize(fun = function, x0 = initNNParams, method = 'TNC', jac = True, options = {'disp': True, 'maxfun': 1000})
cost = optim.fun
nn_params = optim.x

#roll parameters:
theta1 = nn_params[:hiddenLayerSize * (inputLayerSize + 1)].reshape(hiddenLayerSize, (inputLayerSize + 1))
theta2 = nn_params[hiddenLayerSize * (inputLayerSize + 1):].reshape(labelsSize, (hiddenLayerSize + 1))

#Visualizing trained model:

if (display == 1): 
    print('\nVisualizing Neural Network:\n')
    ml.displayData(theta1[:,1:], 'Trained Model')

#Caculating Accuracy:
p = ml.predict_nn(theta1, theta2, X[:trainingLimit]) # predictions of the model.
p = np.asarray(labels)[p].reshape(-1,1) #Formatting p to compare with y
accuracy = np.mean((p == y[:trainingLimit])*1)*100
print('\nTraining Set Accuracy: ', accuracy, '\n')

p = ml.predict_nn(theta1, theta2, X[trainingLimit+1:validationLimit]) # predictions of the model.
p = np.asarray(labels)[p].reshape(-1,1) #Formatting p to compare with y
accuracy = np.mean((p == y[trainingLimit+1:validationLimit])*1)*100
print('\nValidation Set Accuracy: ', accuracy, '\n')

#Testing:

testSet = X[validationLimit+1:]

print('\nTesting Model...\n')

plt.ion() #enable interactive mode

for i in range(len(testSet)):
    try:
        Xi = testSet[i,:].reshape(1,-1) #Selecting sample
        p = ml.predict_nn(theta1, theta2, Xi)
        if (display == 1): 
            titleText = 'Input image is ' + str(labels[p[0]])
            ml.displayData(Xi, titleText)
            plt.pause(1)
            print('\033[F\033[F')
        else: 
            print('\033[F\033[F\033[F\rNon-displayable data. Parameters are: \n', Xi, end='',flush=True)
        print('\rData objective is: ', labels[p[0]], '. \n\'ctrl-c\' to finish', end='',flush=True)
        time.sleep(1)
    except KeyboardInterrupt:
        sys.exit()
