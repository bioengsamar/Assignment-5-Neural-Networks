import numpy as np

class NeuralNetwork:
    def __init__(self, no_of_in_nodes, no_of_out_nodes, no_of_hidden_nodes,learning_rate=0.01,gamma=0.0,n_epoch=100):
        #parameters
        self.inputsize= no_of_in_nodes
        self.outputsize= no_of_out_nodes
        self.hiddensize= no_of_hidden_nodes
        self.learning_rate= learning_rate
        self.gamma= gamma
        self.epochs= n_epoch
        
        # initialize the weights
        np.random.seed(1)
        self.w1= np.random.randn(self.inputsize, self.hiddensize) # weight matrix from input to hidden layer
        self.w2= np.random.randn(self.hiddensize, self.outputsize) # weight matrix from hidden to output layer

    def forward_propagation(self, x):
        self.z = np.dot(x, self.w1) #dot product of x (input) & first set of weights 
        self.z2= self.sigmoid(self.z) #activation function 
        self.z3= np.dot(self.z2, self.w2) #dot product of hidden layer (z2) & second set of weights 
        output= self.sigmoid(self.z3)
        return output
        
    def sigmoid(self, s, derivative=False):
        if derivative == True:
            return s * (1 - s)
        return 1/(1+ np.exp(-s))
    
    def one_hot_encoding(self, y):
        y_encoding=[]
        #print(y)
        for i in y:
            #print(type(i))
            expected = [0 for i in range(self.outputsize)]
            expected[i] = 1 #one hot encoding
            y_encoding.append(expected)
        return y_encoding
    
    def back_propagation(self, x, y, output):
        self.output_error= self.one_hot_encoding(y) - output # error in output
        self.output_delta= self.output_error * self.sigmoid(output, derivative=True) #output layer
        
        self.z2_error= self.output_delta.dot(self.w2.T) #how much our hidden layer weights contribute to output error
        self.z2_delta=self.z2_error * self.sigmoid(self.z2, derivative=True) #applying derivative of sigmoid to z2_error
        self.update_weights(x, self.z2_delta, self.output_delta)
        
       
    def update_weights(self, x, z2_delta, output_delta):
        output_delta_prev=np.zeros(shape=(len(x), self.outputsize))
        z2_delta_prev=np.zeros(shape=(len(x), self.hiddensize))
        self.w1 += self.learning_rate * x.T.dot(z2_delta)+ self.gamma * x.T.dot(z2_delta_prev) # updating first set (input -> hidden) weights with or without momentum
        self.w2 += self.learning_rate * self.z2.T.dot(output_delta)+ self.gamma * self.z2.T.dot(output_delta_prev)# updating second set (hidden -> output) weights with or without momentum
        output_delta_prev=output_delta
        z2_delta_prev=z2_delta
        
    def fit(self, x, y):
        losses=[]
        accuracy_train=[]
        for epoch in range(self.epochs):
            #print(epoch)
            output = self.forward_propagation(x)
            self.back_propagation(x, y, output)
            loss=np.mean(np.square(self.one_hot_encoding(y) - output))
            losses.append(loss)
            accuracy = 1 - np.mean(abs(self.one_hot_encoding(y) - output))
            accuracy_train.append(accuracy)
        return losses, accuracy_train
                
    
    def predict(self, row):
    	outputs = self.forward_propagation(row)
    	return list(outputs).index(max(outputs))
    
    def score(self, actual, predicted):
    	correct = 0
    	for i in range(len(actual)):
    		if actual[i] == predicted[i]:
    			correct += 1
    	return correct / float(len(actual)) * 100.0
   
    