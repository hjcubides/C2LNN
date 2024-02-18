import subprocess as sp
import numpy as np
import mltools as ml
from scipy.optimize import minimize
from scipy.io import loadmat

sp.call('cls', shell=True) # clear the screen

#defining neural network shape:
input_layer_size  = 400
hidden_layer_size = 25
num_labels = 10

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

#Random initialization of Neural Network Parameters:

init_theta1 = ml.randInitWeights(input_layer_size, hidden_layer_size)
init_theta2 = ml.randInitWeights(hidden_layer_size, num_labels)
init_nn_params = np.append(init_theta1.flatten(), init_theta2.flatten()) #unroll parameters

print('Training neural network...')
lam = 1 #lambda of cost function
labels = list(set(y.flatten())) #a list from unique values from y

#Cost function to optimize. (lam and lambda are not the same):
function = lambda t: ml.nnCostFunction(t, input_layer_size, hidden_layer_size, labels, X, y, lam)

#optimization:
optim = minimize(fun = function, x0 = init_nn_params, method = 'TNC', jac = True, options = {'disp': True})
cost = optim.fun
nn_params = optim.x

#roll parameters
theta1 = nn_params[:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, (input_layer_size + 1))
theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape(num_labels, (hidden_layer_size + 1))

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
