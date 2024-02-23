# C2LNN
Two layer neural network that performs classification.

Neural network of two layers. Parameters in input layer dynamically adapt to trainig set (X) columns. Hidden layer have 25 nodes and the output layer dynamically adapt to unique values in objective (y) of the training set.

Data set is splitted with the typical proportion 70-15-15 for training, validation and testing purposes respectively. Additionally data set is randomized to handle sorted cases.

- nn.py is the main code.
- mltools.py has functions of machine learning.
- Datasets collection are in datasets folder.
