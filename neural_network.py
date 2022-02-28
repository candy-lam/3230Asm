import torch
import pandas as pd
import matplotlib.pyplot as plt


def plot_loss(x, loss):
    plt.plot(x, loss)
    plt.savefig('plots/train_loss_q3.png')


# Load training and testing data
def load_data():
    train_x = torch.tensor(pd.read_csv('train_q3.csv')['x']).unsqueeze(1).float()
    train_y = torch.tensor(pd.read_csv('train_q3.csv')['y']).unsqueeze(1).float()

    test_x = torch.tensor(pd.read_csv('test_q3.csv')['x']).unsqueeze(1).float()
    test_y = torch.tensor(pd.read_csv('test_q3.csv')['y']).unsqueeze(1).float()

    return train_x, train_y, test_x, test_y


# Randomly initialize weights
def initialize():
    w1 = torch.tensor([[0.12, 0.26,  -0.15]], device='cpu', requires_grad=True)
    w2 = torch.tensor(torch.tensor([[0.11, 0.13, 0.07]]).t().tolist(), device='cpu', requires_grad=True)

    return w1, w2


def lr_decay(lr, iter):
    return lr * pow(0.3, iter // 20)


# Test model
def test(test_x, test_y, w1, w2, threshold=0.5):
    y_pred = test_x.mm(w1).clamp(min=0).mm(w2)
    y_pred = torch.sigmoid(y_pred) > threshold

    print('Test Accuracy:', 100. * y_pred.eq(test_y).sum().item() / len(test_y), '%')


iterations = 50
init_learning_rate = 0.5


train_x, train_y, test_x, test_y = load_data()
w1, w2 = initialize()
train_loss = []


for t in range(iterations):
    learning_rate = lr_decay(init_learning_rate, t)
    
    # Forward pass: compute predicted y
    y_pred = train_x.mm(w1).clamp(min=0).mm(w2)

    # Compute and print loss; loss is a scalar, and is stored in a PyTorch Tensor
    # of shape (); we can get its value as a Python number with loss.item().
    loss = torch.nn.functional.binary_cross_entropy_with_logits(y_pred, train_y, reduction='mean')
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
