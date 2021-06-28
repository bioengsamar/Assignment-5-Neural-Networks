import pandas as pd
from neural_networks import NeuralNetwork
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt
import numpy as np
from sklearn import preprocessing

# Load a excel file
def load_excel(filename):
    data = pd.read_excel(filename)
    data = data.drop('Train_or_Test', 1)
    data = data.drop('Speaker_Number', 1)
    data = data.drop('Sex', 1)
    x=data.iloc[:,0:10]
    y=data.iloc[:,10]
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    return x, y

def model(X_train, X_test, y_train, y_test):
    costs=[]
    scores_test=[]
    accuracy_train=[]
    different_learning_rates=[0.03,0.01,0.05,0.06]
    for i in different_learning_rates:
        ANN = NeuralNetwork(no_of_in_nodes=10, no_of_out_nodes=11, no_of_hidden_nodes=20,
                           learning_rate=i,n_epoch=1000)
        #ANN_momentum = NeuralNetwork(no_of_in_nodes=4, no_of_out_nodes=3, no_of_hidden_nodes=2,
                           #learning_rate=i,gamma=0.7,n_epoch=50)
        cost, accuracy=ANN.fit(X_train,y_train)
        costs.append(cost)
        accuracy_train.append(accuracy)
        predictions=[]
        for row in X_test:
            predicted=ANN.predict(row)
            predictions.append(predicted)
        scores_test.append(ANN.score(y_test, predictions))
    return accuracy_train, scores_test
        
def plot(costs):
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(costs[0])), costs[0],  label='learning_rate=0.03')
    ax.plot(np.arange(len(costs[1])), costs[1], label='learning_rate=0.01')
    ax.plot(np.arange(len(costs[2])), costs[2],  label='learning_rate=0.05')
    ax.plot(np.arange(len(costs[3])), costs[3], label='learning_rate=0.06')
    ax.legend(loc='lower right', shadow=True)
    plt.xlabel("epochs")
    plt.ylabel("scores")
    plt.show()
    fig.savefig('vowel_score.jpg')

if __name__ == "__main__":
    path="dataset_58_vowel.xlsx"
    x, y=load_excel(path)
    #split data into 75% train and 25% test
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25,random_state=42)
    
    accuracy_train, scores_test= model(X_train, X_test, y_train, y_test)
    plot(accuracy_train) #or plot costs
    print(scores_test)
    
    
    