import numpy as np 
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

np.random.seed(45)

EPOCHS = 5000
LR = 1e-1

def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0] - pt[1]) / 1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1) 
    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy():
    inputs = []
    labels = []

    for i in range(11):
        inputs.append([0.1 * i, 0.1 * i])
        labels.append(0)
        if 0.1 * i == 0.5:
            continue        
        inputs.append([0.1 * i, 1 -0.1 * i])
        labels.append(1)
    
    return np.array(inputs), np.array(labels).reshape(21, 1)

def show_result(x, y, pred_y):
    plt.subplot(1, 2, 1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.subplot(1, 2, 2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    
    plt.show()

def learning_curve(epoch, loss):
    plt.plot(epoch, loss)
    plt.title('Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def derivative_sigmoid(x):
    return np.multiply(x, 1.0 - x) 

def ReLU(x):
    return np.where(x < 0, 0.07 * x, x)

def derivative_ReLU(x):
    return np.where(x < 0, 0.07, 1)

def initialize_network(n_input, hidden_dimension, n_output):
    net = []
    hidden_layer1 = np.random.uniform(-1, 1, (n_input, hidden_dimension))
    net.append(hidden_layer1)
    hidden_layer2 = np.random.uniform(-1, 1, (hidden_dimension, hidden_dimension))
    net.append(hidden_layer2)
    output_layer = np.random.uniform(-1, 1, (hidden_dimension, n_output))
    net.append(output_layer)
    return net

def forwarding_pass(net, input_data):
    inputs = input_data
    outputs = []
    for layer in net:
        inputs = sigmoid(inputs @ layer)
        outputs.append(inputs)
    return inputs, outputs

def loss_function(predict_result, ground_truth):
    mse_loss = np.sum(np.power((predict_result - ground_truth), 2)) / len(ground_truth)
    return mse_loss

def backpropagation(net, y, outputs, pred, x):
    update_net = net.copy()
    # net[2]
    C_a2 = 2 * pred - 2 * y    
    a2_z2 = np.diag(derivative_sigmoid(outputs[2]).reshape(-1))
    z2_W2 = outputs[1].T
    C_W2 = z2_W2 @ a2_z2 @ C_a2
    
    # net[1]
    C_W1 = np.zeros((len(net[1]), 1))
    C_z1 = np.zeros((1, 1))
    for i in range(len(net[2])):
        z2_a1 = np.array([net[2][i]]).reshape(1, 1)
        a1_z1 = np.array([derivative_sigmoid(outputs[1]).reshape(-1)[i]]).reshape(1, 1)
        z1_W1 = outputs[0].T
        C_z1 = np.concatenate((C_z1, a1_z1 @ z2_a1 @ a2_z2 @ C_a2), axis=1)
        C_W1 = np.concatenate((C_W1, z1_W1 @ a1_z1 @ z2_a1 @ a2_z2.T @ C_a2.T), axis=1)

    C_W1 = C_W1[:, 1:]
    C_z1 = C_z1[:, 1:]
    
    # net[0]

    C_W0 = np.zeros((2, 1))
    for i in range(len(net[1])):
        z1_a0 = np.array([net[1][i]]).reshape(len(net[1]), 1)
        a0_z0 = np.array([derivative_sigmoid(outputs[0]).reshape(-1)[i]]).reshape(1, 1)
        z0_W0 = x.T
        C_W0 = np.concatenate((C_W0, z0_W0 @ a0_z0 @ z1_a0.T @ C_z1.T), axis=1)

    C_W0 = C_W0[:, 1:]
   
    update_net[0] -= LR * C_W0
    update_net[1] -= LR * C_W1
    update_net[2] -= LR * C_W2
    return update_net

def linear_task():
    epochs = []
    losses = []
    x, y = generate_linear(n=100)
    net = initialize_network(2, 10, 1)
    for i in range(EPOCHS):
        loss = 0
        for j in range(len(x)):
            pred, outputs = forwarding_pass(net, x[j].reshape(1, 2))
            net = backpropagation(net, y[j].reshape(1, 1), outputs, pred, x[j].reshape(1, 2)) 
            loss += loss_function(pred, y[j].reshape(1, 1))
        epochs.append(i+1)
        losses.append(loss / len(y))
        print('epoch {} loss : {}'.format(i+1, loss / len(y)))

    pred, _ = forwarding_pass(net, x)
    print(pred)
    pred = np.where(pred > 0.5, 1, 0)
    num_corrects = (pred == y).sum()
    print('Accuracy ', num_corrects / len(pred))    
    show_result(x, y, pred)
    learning_curve(epochs, losses)

def XOR_task():
    epochs = []
    losses = []
    x, y = generate_XOR_easy()
    net = initialize_network(2, 10, 1)
    for i in range(EPOCHS):
        loss = 0
        for j in range(len(x)):
            pred, outputs = forwarding_pass(net, x[j].reshape(1, 2))
            net = backpropagation(net, y[j].reshape(1, 1), outputs, pred, x[j].reshape(1, 2)) 
            loss += loss_function(pred, y[j].reshape(1, 1))
        epochs.append(i+1)
        losses.append(loss / len(y))
        print('epoch {} loss : {}'.format(i+1, loss / len(y)))

    pred, _ = forwarding_pass(net, x)
    print(pred)
    pred = np.where(pred > 0.5, 1, 0)
    num_corrects = (pred == y).sum()
    print('Accuracy ', num_corrects / len(pred))    
    show_result(x, y, pred)
    learning_curve(epochs, losses)

if __name__ == '__main__':
    # linear_task()
    XOR_task()