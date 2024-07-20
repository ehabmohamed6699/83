import numpy as np
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("Dry_Bean_Dataset.csv")
# df = pd.read_excel("Dry_Bean_Dataset.xlsx")

df.head(5)

# sns.scatterplot(data=df, x=df["Area"], y=df["Perimeter"], hue=df["Class"])
# plt.title('Scatter Plot with class')
# plt.xlabel('Area')
# plt.ylabel('Perimeter')
# plt.show()
#
# sns.scatterplot(data=df, x=df["Area"], y=df["MajorAxisLength"], hue=df["Class"])
# plt.title('Scatter Plot with class')
# plt.xlabel('Area')
# plt.ylabel('MajorAxisLength')
# plt.show()
#
# sns.scatterplot(data=df, x=df["Area"], y=df["MinorAxisLength"], hue=df["Class"])
# plt.title('Scatter Plot with class')
# plt.xlabel('Area')
# plt.ylabel('MinorAxisLength')
# plt.show()
#
# sns.scatterplot(data=df, x=df["Area"], y=df["roundnes"], hue=df["Class"])
# plt.title('Scatter Plot with class')
# plt.xlabel('Area')
# plt.ylabel('roundnes')
# plt.show()
#
# sns.scatterplot(data=df, x=df["Perimeter"], y=df["MajorAxisLength"], hue=df["Class"])
# plt.title('Scatter Plot with class')
# plt.xlabel('Perimeter')
# plt.ylabel('MajorAxisLength')
# plt.show()
#
# sns.scatterplot(data=df, x=df["Perimeter"], y=df["MinorAxisLength"], hue=df["Class"])
# plt.title('Scatter Plot with class')
# plt.xlabel('Perimeter')
# plt.ylabel('MinorAxisLength')
# plt.show()
#
# sns.scatterplot(data=df, x=df["Perimeter"], y=df["roundnes"], hue=df["Class"])
# plt.title('Scatter Plot with class')
# plt.xlabel('Perimeter')
# plt.ylabel('roundnes')
# plt.show()
#
# sns.scatterplot(data=df, x=df["MajorAxisLength"], y=df["MinorAxisLength"], hue=df["Class"])
# plt.title('Scatter Plot with class')
# plt.xlabel('MajorAxisLength')
# plt.ylabel('MinorAxisLength')
# plt.show()
#
# sns.scatterplot(data=df, x=df["MajorAxisLength"], y=df["roundnes"], hue=df["Class"])
# plt.title('Scatter Plot with class')
# plt.xlabel('MajorAxisLength')
# plt.ylabel('roundnes')
# plt.show()
#
# sns.scatterplot(data=df, x=df["MinorAxisLength"], y=df["roundnes"], hue=df["Class"])
# plt.title('Scatter Plot with class')
# plt.xlabel('MinorAxisLength')
# plt.ylabel('roundnes')
# plt.show()
# # df
#
# sns.displot(data=df, x=df["Area"], hue=df["Class"])
# # sns.histplot(data=df, x='Area', hue='Class', element='step', common_norm=False, stat='count')
# plt.title('Class')
# plt.xlabel('Area')
# plt.show()
#
# sns.displot(data=df, x=df["Perimeter"], hue=df["Class"])
# plt.title('Class')
# plt.xlabel('Perimeter')
# plt.show()
#
# sns.displot(data=df, x=df["MajorAxisLength"], hue=df["Class"])
# plt.title('Class')
# plt.xlabel('MajorAxisLength')
# plt.show()
#
# sns.countplot(data=df, x=df["MinorAxisLength"], hue=df["Class"])
# plt.title('Class')
# plt.xlabel('MinorAxisLength')
# plt.show()
#
# sns.countplot(data=df, x=df["roundnes"], hue=df["Class"])
# plt.title('Class')
# plt.xlabel('roundnes')
# plt.show()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dervative_sigmoid(x):
     return x * (1 - x)

def tanh(x):
    return np.tanh(x)

def dervative_tanh(x):
    return 1.0 - np.tanh(x)**2

def feed_forward(x, paramters, layers, activition):
    mylist = []
    Input = x
    for i in range(1, len(layers)):
        input_prev = Input
        if (activition == "sigmoid"):

            z = np.dot(paramters['w' + str(i)], input_prev) + paramters['b' + str(i)]


            Input = sigmoid(z)
        elif (activition == "tanh"):

            z = np.dot(paramters['w' + str(i)], input_prev) + paramters['b' + str(i)]


            Input = tanh(z)
        mylist.append((z, input_prev, paramters['w' + str(i)], paramters['b' + str(i)]))

    return mylist, Input




def Backward(Y, Y_predicted, activation, layers, mylist):
    #computing gradients
    grads = {}
    Num_layers = len(layers) - 1

    encodeY = np.array([[1, 0, 0]]).T
    # encodepredict = np.array([[1 ,0, 0]]).T
    if Y == 1:
        encodeY = np.array([[0, 1, 0]]).T
    elif Y == -1:
        encodeY = np.array([[0, 0, 1]]).T

  
     
    if (activation == "sigmoid"):
        z, input_prev, w, b = mylist[Num_layers - 1]
        s = (Y_predicted - encodeY) * dervative_sigmoid(z)
        grads["dw" + str(Num_layers)] = np.dot(s, input_prev.T)
        grads["db" + str(Num_layers)] = s
        grads["da" + str(Num_layers - 1)] = np.dot(w.T, s)
        for i in reversed(range(Num_layers - 1)):
            z, input_prev, w, b = mylist[i]
            s = grads["da" + str(i + 1)] * dervative_sigmoid(z)
            grads["dw" + str(i + 1)] = np.dot(s, input_prev.T)
            grads["db" + str(i + 1)] = s
            grads["da" + str(i)] = np.dot(w.T, s)
    elif (activation == "tanh"):
        z, input_prev, w, b = mylist[Num_layers - 1]
        s = (Y_predicted - encodeY) * dervative_tanh(z)
        grads["dw" + str(Num_layers)] = np.dot(s, input_prev.T)
        grads["db" + str(Num_layers)] = s
        grads["da" + str(Num_layers - 1)] = np.dot(w.T, s)
        for i in reversed(range(Num_layers - 1)):
            z, input_prev, w, b = mylist[i]
            s = grads["da" + str(i + 1)] * dervative_tanh(z)
            grads["dw" + str(i + 1)] = np.dot(s, input_prev.T)
            grads["db" + str(i + 1)] = s
            grads["da" + str(i)] = np.dot(w.T, s)

    return grads