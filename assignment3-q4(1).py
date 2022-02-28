import torch
import pandas as pd
import matplotlib.pyplot as plt


# Plot training loss curve
def plot_loss(x, loss):
    plt.plot(x, loss)
    plt.savefig('plots/train_loss_q4.png')


# Load training and testing data
def load_data():
    x1 = torch.tensor(pd.read_csv('train_q4.csv')['x1']).unsqueeze(1).float()
    x2 = torch.tensor(pd.read_csv('train_q4.csv')['x2']).unsqueeze(1).float()

    train_x = torch.cat((x1, x2), dim=1)
    train_y = torch.tensor(pd.read_csv('train_q4.csv')['y'])

    x1 = torch.tensor(pd.read_csv('test_q4.csv')['x1']).unsqueeze(1).float()
    x2 = torch.tensor(pd.read_csv('test_q4.csv')['x2']).unsqueeze(1).float()
    test_x = torch.cat((x1, x2), dim=1)
    test_y = torch.tensor(pd.read_csv('test_q4.csv')['y'])

    return train_x, train_y, test_x, test_y


# Randomly initialize weights
def initialize():
    w1 = torch.tensor(
        [[ 0.74, 0.10,  0.98],
        [-2.04, -1.40, -0.31]], device='cpu', requires_grad=True)
    w2 = torch.tensor(torch.tensor(
        [[ 1.37, -0.90, -0.80],
        [-0.08,  0.94,  0.47],
        [-0.30,  0.57,  0.93]]).t().tolist(), device='cpu', requires_grad=True)
    # w1 = torch.rand((2, 3), device='cpu', requires_grad=True)
    # w2 = torch.tensor(torch.rand((3, 3)).t().tolist(), device='cpu', requires_grad=True)

    return w1, w2


# Learning rate decay
def lr_decay(lr, iter):
    return lr * pow(0.3, iter // 20)


# Test model
def test(test_x, test_y, w1, w2):
    y_pred = test_x.mm(w1).clamp(min=0).mm(w2)
    y_pred = torch.softmax(y_pred, dim=1).max(1)[1]

    print('Test Accuracy:', 100. * y_pred.eq(test_y).sum().item() / len(test_y), '%')


iterations = 50
init_learning_rate = 1e-1

train_x, train_y, test_x, test_y = load_data()
w1, w2 = initialize()
train_loss = []


for t in range(iterations):
    learning_rate = lr_decay(init_learning_rate, t)
    
    # Forward pass: compute predicted y
    y_pred = train_x.mm(w1).clamp(min=0).mm(w2)

    # Compute and print loss; loss is a scalar, and is stored in a PyTorch Tensor
    # of shape (); we can get its value as a Python number with loss.item().
    loss = torch.nn.functional.nll_loss(torch.nn.LogSoftmax(dim=1)(y_pred), train_y)
    train_loss.append(loss.item())

    # Backprop to compute gradients of w1 and w2 with respect to loss
    loss.backward()

    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # Manually zero the gradients after running the backward pass
        w1.grad.zero_()
        w2.grad.zero_()


print(w1)
print(w2.t())

plot_loss([_ for _ in range(iterations)], train_loss)

test(test_x, test_y, w1, w2)