import seaborn as sns
from sklearn import preprocessing
from neural_networks import NeuralNetwork
import matplotlib.pylab as plt
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(cols):
    iris = sns.load_dataset("iris")
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(iris["species"])
    x= iris
    x= x[cols]
    return x.values, y

def model(X_train, X_test, y_train, y_test):
    costs=[]
    scores_test=[]
    accuracy_train=[]
    different_learning_rates=[0.1,0.06,0.03,0.01]
    for i in  different_learning_rates:
        
        ANN = NeuralNetwork(no_of_in_nodes=4, no_of_out_nodes=3, no_of_hidden_nodes=4,
                          learning_rate=i,n_epoch=50)
        
        #ANN_momentum = NeuralNetwork(no_of_in_nodes=4, no_of_out_nodes=3, no_of_hidden_nodes=2,
                           #learning_rate=i,gamma=0.7,n_epoch=50)
        
        cost, accuracy=ANN.fit(X_train,y_train)
        costs.append(cost)
        accuracy_train.append(accuracy)
        predictions=[]
        for row in X_test:
            #print(type(list(row)))
            predicted=ANN.predict(row)
            predictions.append(predicted)
        scores_test.append(ANN.score(y_test, predictions))
    return accuracy_train, scores_test
        
def plot(costs):
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(costs[0])), costs[0],  label='learning_rate=0.1')
    ax.plot(np.arange(len(costs[1])), costs[1], label='learning_rate=0.06')
    ax.plot(np.arange(len(costs[2])), costs[2],  label='learning_rate=0.03')
    ax.plot(np.arange(len(costs[3])), costs[3], label='learning_rate=0.01')
    ax.legend(loc='upper left', shadow=True)
    plt.xlabel("epochs")
    plt.ylabel("scores")
    plt.show()
    fig.savefig('iris_score.jpg')

if __name__ == '__main__':
    cols = ["sepal_length","sepal_width","petal_length", "petal_width"]
    x, y = load_data(cols)
    # scale the data
    x= (x - x.mean()) / x.std() 
    #split data into 70% train and 30% test
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=42)
    
    accuracy_train, scores_test=model(X_train, X_test, y_train, y_test)
    plot(accuracy_train) #or plot costs
    print(scores_test)
    
    