import numpy as np
import matplotlib.pyplot as plt

#Compute the sigmoid function
def sigmoid(z):
    g = 1/(1+np.power(np.exp(1),-z))
    return(g)

#computes the gradient of the sigmoid function evaluated at z
def sigmoidGradient(z):
    g = sigmoid(z)*(1-sigmoid(z))
    return(g)

#randomly initializes the weights of a layer with L_in incoming connections and L_out outgoing connections.
def randInitWeights(L_in, L_out):
    epsilon_init = 0.12
    W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init
    return(W)

#cost function for a two layer neural network
def nnCostFunction(parameters, inputlayerSize, hiddenLayerSize, labels, X, y, lam): #lam is lambda
    #variables initilization:
    theta1 = parameters[:hiddenLayerSize * (inputlayerSize + 1)].reshape(hiddenLayerSize, (inputlayerSize + 1))
    theta2 = parameters[hiddenLayerSize * (inputlayerSize + 1):].reshape(len(labels), (hiddenLayerSize + 1))
    theta1_gradient = np.zeros(theta1.shape)
    theta2_gradient = np.zeros(theta2.shape)
    m = X.shape[0] #rows
    n = X.shape[1] #columns
    cost = 0
    X = np.c_[np.ones((m,1)),X] #Adding bias column to X

    z = np.dot(X, theta1.T) #hypothesis

    #Feedforward:
    hiddenLayer = sigmoid(z)
    hiddenLayer = np.c_[np.ones((m,1)),hiddenLayer] #Adding bias column to hidden layer.
    outputLayer = sigmoid(np.dot(hiddenLayer, theta2.T))
    '''
    fixing of nodes >= 1.
    if a node of the output layer is >= 1 calc of t2 will get an error...
    ...cause np.log will try to calculate the log of a <= 0 number which is NaN.
    So all nodes >= 1 wil be set to 0.9999
    '''
    if (sum(sum((outputLayer >= 1)*1)) > 0):
        print('Fixing nodes...')
        outputLayer[outputLayer >= 1] = 0.9999

    ny = (y == labels)*1 #this produce a 2-dimensional array (matrix) of length_y*length_labels
    regularization = (lam/(2*m))*(np.sum(np.sum(np.power(theta1[:,1:],2), axis = 1))+np.sum(np.sum(np.power(theta2[:,1:],2), axis = 1)))
    t1 = -ny*np.log(outputLayer)
    t2 = (1-ny)*np.log(1-outputLayer)
    cost = (sum(np.sum(t1-t2, axis=1))/m)+regularization


    #backpropagation:
    delta_3 = outputLayer-ny #error term for output layer

    z = np.c_[np.ones((m,1)), z] #adding bias column to z

    delta_2 = np.dot(theta2.T, delta_3.T).T*sigmoidGradient(z) ##error term for hidden layer
    theta1_gradient += np.dot(delta_2[:,1:].T, X)
    theta2_gradient += np.dot(delta_3.T, hiddenLayer)
    theta1_gradient /= m
    theta2_gradient /= m
    theta1_gradient[:,1:] += (lam/m)*theta1[:,1:]
    theta2_gradient[:,1:] += (lam/m)*theta2[:,1:]
    grad = np.append(theta1_gradient.flatten(), theta2_gradient.flatten()).reshape(-1,1)
    return(cost, grad)

#visualization tool
def displayData(X, cmap='gray', ax = plt):
    (m, n) = X.shape
    exampleWidth = int(np.round(np.sqrt(n)))
    exampleHeight = int((n/exampleWidth))
    itemsRowsToDisplay = int(np.floor(np.sqrt(m)))
    itemsColumsToDisplay = int(np.ceil(m/itemsRowsToDisplay))
    pad = 1
    itemsArrayToDisplay = np.ones((pad+itemsRowsToDisplay*(exampleHeight+pad),pad+itemsColumsToDisplay*(exampleWidth+pad)))
    cursorExample = 0
    for j in range(itemsRowsToDisplay):
        for i in range(itemsColumsToDisplay):
            if cursorExample > m-1:
                break
            patchMaxValue = np.max(np.abs(X[cursorExample,:])) #used to limit the values of the patch up to 1
            r = [pad+j*(exampleHeight + pad)+a for a in range(exampleHeight+1)]
            c = [pad+i*(exampleWidth + pad)+b for b in range(exampleWidth+1)]
            itemsArrayToDisplay[min(r):max(r),min(c):max(c)] = X[cursorExample,:].reshape(exampleHeight, exampleWidth).T/patchMaxValue
            cursorExample += 1
        if cursorExample > m-1:
            break

    ax.imshow(itemsArrayToDisplay, cmap=cmap)
    plt.show()

#Model Prediction
def predict_nn(theta1, theta2, X):
    m = X.shape[0]
    X = np.c_[np.ones((m,1)),X] #Adding bias column to X
    z = np.dot(X, theta1.T) #hypothesis
    hiddenLayer = sigmoid(z)
    hiddenLayer = np.c_[(np.ones((m,1))),hiddenLayer] #Adding bias column to hiddenLayer
    outputLayer = sigmoid(np.dot(hiddenLayer, theta2.T))
    mx = np.argmax(outputLayer, axis=1)
    return(mx)
