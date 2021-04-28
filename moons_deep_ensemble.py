import torch
import torch.utils.data
from torch.nn import functional as F

import numpy as np
import sklearn.datasets
import copy

from models import MLP3, mlp_step

def mlp_eval_step(model, x, y):
    model.eval()
    with torch.no_grad():
        z = model(x)
        y_pred = F.softmax(z, dim=1)
        probs = y_pred.detach().numpy()
        loss = F.binary_cross_entropy(y_pred, F.one_hot(torch.from_numpy(y)).float())
        y_pred = np.argmax(probs, axis=1)
        accuracy = np.mean(y == y_pred)
    return accuracy, loss.detach().item(), probs

def mlp_eval_combined(model, x_combined):
    model.eval()
    with torch.no_grad():
        z = model(x_combined)
        probs = F.softmax(z, dim=1)
    return probs.detach().numpy()

# Moons
noise = 0.1
batch_size = 64
x_train, y_train = sklearn.datasets.make_moons(n_samples=1500, noise=noise)
mean = np.mean(x_train, axis=0)
std = np.std(x_train, axis=0)
print('mean, std', mean, std)
x_train = (x_train-mean)/std/np.sqrt(2)
x_test, y_test = sklearn.datasets.make_moons(n_samples=500, noise=noise)
x_test = (x_test-mean)/std/np.sqrt(2)
x_train = torch.tensor(x_train, dtype=torch.float)
x_test = torch.tensor(x_test, dtype=torch.float)
ds_train = torch.utils.data.TensorDataset(x_train, F.one_hot(torch.from_numpy(y_train)).float())
dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)


seed = 0
inputs = 2
features = 64
ensemble = 10
learningRate = 0.001
l2Penalty = 1.0e-3
epochs = 101
models = []
optimizers = []
np.random.seed(seed)
torch.manual_seed(seed)

# 1st, train models
for i in range(ensemble):
    print('model ', i)
    # model = Model_bilinear(20)
    model = MLP3(inputs, features)
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate,
                                 weight_decay=l2Penalty)
    optimizers.append(optimizer)

    ms = []
    for epoch in range(0, epochs, 5):
        for i, batch in enumerate(dl_train):
            mlp_step(model, optimizer, batch)
        acc, loss, _ = mlp_eval_step(model, x_train, y_train)
        acc_test, loss_test, _ = mlp_eval_step(model, x_test, y_test)
        print('epoch {}, acc {:.3f}, test acc {:.3f}'.format(epoch, acc, acc_test))
        ms.append(copy.deepcopy(model))
    models.append(ms)


# 2nd find the best episode which has the best test act
bestAcc = 0.95
modelsAtBestTestAcc = []
for i in range(0, epochs, 5):
    probs = []
    for j in range(ensemble):
        acc, loss, prob = mlp_eval_step(models[j][i//5], x_test, y_test)
        probs.append(prob)
    probs = np.array(probs)
    prob = np.mean(probs, axis=0)
    y_pred = np.argmax(prob, axis=1)
    acc = np.mean(y_test == y_pred)
    print('episode, acc ', i, acc)
    if acc >= bestAcc:
        bestAcc = acc
        modelsAtBestTestAcc.clear()
        for j in range(ensemble):
            modelsAtBestTestAcc.append(models[j][i//5])
        print('best acc {:.3f} at episode {}'.format(acc, i))

# 3rd calculate the score distribution at grid
domain = 3
x_lin = np.linspace(-domain + 0.5, domain + 0.5, 100)
y_lin = np.linspace(-domain, domain, 100)
x_lin = (x_lin-mean[0])/std[0]/np.sqrt(2)
y_lin = (y_lin-mean[1])/std[1]/np.sqrt(2)
xx, yy = np.meshgrid(x_lin, y_lin)
x_grid = np.column_stack([xx.flatten(), yy.flatten()])
x_grid = torch.tensor(x_grid, dtype=torch.float)
probs = []
for j in range(ensemble):
    prob = mlp_eval_combined(modelsAtBestTestAcc[j], x_grid)
    probs.append(prob)
probs = np.array(probs)
prob = np.mean(probs, axis=0)
score = -prob * np.log(prob)
score = score.sum(axis=1)
z = score.reshape(xx.shape)
outputDir =  'results/'
np.savez(outputDir + "moons_confidence_map_deep_ensemble.npz", a=x_lin, b=y_lin, c=z, d=x_train.numpy(), e=y_train)
