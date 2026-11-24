import numpy as np

class ArtificialNeuralNetwork:



    def __init__(self, architecture, activations):

        """
            Initialises a configurable feedforward neural network
            architecture: List containing the number of neurons in each layer Example: [4, 8, 6, 3]
            activations: List containing the activation function for each hidden and output layer
                         Example: ["relu", "tanh", "sigmoid", "leaky relu", "elu", "mish", "softplus"]
         """

        #Checks if the network has an input and output layer
        if len(architecture) < 2:
            raise ValueError("The network must have at least two layers.")          

        #Checks that every layer has atleast one neuron
        for n in architecture:
            if n <= 0:
                raise ValueError("Each layer must contain at least one neuron.")   

        #Each layer needs an activation function after input layer
        if len(activations) != len(architecture) - 1:
            raise ValueError("Each non-input layer needs an activation function.") 

        self.architecture = architecture  

        #Create a list to store activation function
        self.activations = []                                                     
 
        for a in activations:
            self.activations.append(a.lower())


       #Create lists to store weights and biases between layers
        self.weights = []
        self.biases = []

        #Create weights and biases for each connection between layers
        for i in range(len(architecture) - 1):

            #Get number of neurons in the current layer
            input_nodes = architecture[i]         

            #Get number of neurons in the next layer
            output_nodes = architecture[i + 1]         

            #Randomly initialise weights connecting the two layers
            #The small value (0.1) helps keep the initial weights small
            weight = np.random.randn(input_nodes, output_nodes) * 0.1


            bias = np.zeros(output_nodes) # either 1 or 0 (current)

            #Save the weights and biases for this layer
            self.weights.append(weight)
            self.biases.append(bias)



    def activate(self, x, function):

        """
        Applies the selected activation function to the input.
        """

        #Sigmoid converts values into a range between 0 and 1
        if function == "sigmoid":
            x = np.clip(x, -500, 500) 
            return 1 / (1 + np.exp(-x))

        #ReLU changes negative values to zero and keeps positive values
        elif function == "relu":
            return np.maximum(0, x)

        #Tanh converts values into a range between -1 and 1
        elif function == "tanh":
            return np.tanh(x)

        #Leaky ReLU allows small negative values 
        elif function == "leaky relu":
            return np.where(x > 0, x, 0.01 * x)

        #ELU keeps positive values unchanged and smoothly transforms negative values 
        elif function == "elu":
            return np.where(x > 0, x, np.exp(np.minimum(x, 0)) - 1)

        # Mish is a smooth activation function that allows small negative outputs
        elif function == "mish":
            return x * np.tanh(np.logaddexp(0, x))

        # softplus smoothly converts inputs into positive values
        elif function == "softplus":
             return np.logaddexp(0, x)

        #if an unsupported activation function is used
        else:
            raise ValueError(f"Unknown activation function: {function}")

  

    def forward_propagation(self, inputs):

        """
        Passes input data through each layer of the network.
        Calculates the weighted sum, adds the bias, and applies
        the activation function at each layer.
        Returns the final output of the network.
        """

        #Convert the input data into a NumPy array containing numbers
        inputs = np.asarray(inputs, dtype=float)

        #If only one input sample is provided, reshape it into a 2D array
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)

        #Check that the input has the correct number of features
        if inputs.ndim != 2 or inputs.shape[1] != self.architecture[0]:
            raise ValueError("Input dimensions do not match the input layer.")

        current_output = inputs

        #Go through each layer using its weights, biases and activation function
        for i in range(len(self.weights)):

            """ Multiply the current output by the weights, add the biases,
                and apply the activation function.
                The result becomes the input to the next layer. """
            current_output = self.activate(
            current_output @ self.weights[i] + self.biases[i],
            self.activations[i]
      )
                

        return current_output

 

    def calculate_parameters(self):

        """
        Calculates the total number of weights and biases.
        This determines the number of values each PSO particle
        needs to represent.
        """

        total_weights = 0
        total_biases = 0

        #Count all the individual weight values in the network
        for w in self.weights:
            total_weights += w.size

        #Count all the individual bias values in the network
        for b in self.biases:
            total_biases += b.size

        return total_weights + total_biases



    def get_parameters(self):

        """
        Converts all weights and biases into a single
        one-dimensional array for PSO.
        """

        parameters = []

       #Go through each layer and collect its weights and biases
        for i in range(len(self.weights)):
             
             #Flatten the weight matrix into a one-dimensional array
             parameters.extend(self.weights[i].flatten())

             #Add the bias values to the same list
             parameters.extend(self.biases[i].flatten())

 
        return np.array(parameters)



    def set_parameters(self, parameters):

        """
        Assigns a one-dimensional parameter array to the
        network's weights and biases.
        This allows PSO particles to configure the ANN.
        """

        parameters = np.asarray(parameters, dtype=float)

        #Make sure the supplied array contains the correct number of values
        if parameters.size != self.calculate_parameters():
            raise ValueError("Incorrect number of ANN parameters.")

        index = 0

        #Go through each layer and update its weights and biases
        for i in range(len(self.weights)):

            weight_size = self.weights[i].size
            bias_size = self.biases[i].size

            # Extract and reshape weights
            self.weights[i] = parameters[index:index + weight_size].reshape(self.weights[i].shape)

            index += weight_size

            # Extract biases
            self.biases[i] = parameters[index:index + bias_size]

            index += bias_size


    def accuracy(self, X, y):
        """
        Calculates classification accuracy for a dataset.
        """
        # Get the ANN's predicted probabilities
        predictions = self.forward_propagation(X).flatten()

        # Convert probabilities into 0 or 1 using a threshold of 0.5
        classes = (predictions >= 0.5).astype(int)

        # Convert the true labels into one dimensional array
        y = np.array(y).flatten()

        # Calculate the proportion of correctly classified samples
        return np.mean(y == classes)


    def calculate_loss(self, X, y):
        """
        Calculates binary cross-entropy loss for a dataset.
        """

        predictions = self.forward_propagation(X).flatten()

        y = np.array(y).flatten()

        # Prevent taking the logarithm of zero
        predictions = np.clip(predictions, 1e-7, 1 - 1e-7)

        # Calculate the average Binary Cross-Entropy loss
        loss = -np.mean( y * np.log(predictions) + (1 - y) * np.log(1 - predictions))

        return loss


    def fitness(self, position, X_train, y_train):
        """
        Assigns PSO particle parameters to the ANN
        and calculates training accuracy as fitness.
        """
        # Apply the particle's position to the ANN parameters
        self.set_parameters(position)

        # Use training accuracy as the particle's fitness
        return self.accuracy(X_train, y_train)
