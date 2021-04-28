import torch.utils.data
from torch.nn import functional as F
import numpy as np

from models import MLP3, mlp_step, mlp_eval_step

dir = 'results/'
f = np.load("combined_dataset_before_feature_selection.npz")

x_train = f['a']
y_train = f['b']
x_validate = f['c']
y_validate = f['d']
print('{} train samples, positive rate {:.3f}'.format(x_train.shape[0], np.mean(y_train)))
print('{} validate samples, positive rate {:.3f}'.format(x_validate.shape[0], np.mean(y_validate)))

dim = x_train.shape[1]
x_train = x_train/np.sqrt(dim)
x_validate = x_validate/np.sqrt(dim)

# network hyper param
batchSize = 64
hiddenUnits = 64
learningRate = 0.0005
l2Penalty = 1.0e-3

seed = 0
runs = 10

trained = False

epochs = 200
np.random.seed(seed)
torch.manual_seed(seed)
inputs = 2
print('input features ', inputs)
x_train0 = x_train.copy()
x_validate0 = x_validate.copy()

# 0 for lag vehicle in target lane
# 1 for front vehicle in target lane
# 2 for front vehicle in original lane
# x = [v_ego, dv0, dv1, dv2, dx0, dx1, dx2, dy0, dy1, dy2]

# mask = np.ones(10).astype(np.bool)
# mask = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0]).astype(np.bool) # drop dy2
# mask = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0]).astype(np.bool) # drop dy1
# mask = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0]).astype(np.bool) # drop dy0
# mask = np.array([0, 1, 1, 1, 1, 1, 1, 0, 0, 0]).astype(np.bool) # drop v_ego
# mask = np.array([0, 1, 1, 1, 1, 1, 0, 0, 0, 0]).astype(np.bool) # drop dx2
# mask = np.array([0, 1, 1, 0, 1, 1, 0, 0, 0, 0]).astype(np.bool) # drop dv2
mask = np.array([0, 1, 0, 0, 1, 1, 0, 0, 0, 0]).astype(np.bool) # drop dv1
for i in range(10):
    maskTemp = mask.copy()
    if not maskTemp[i]: continue
    maskTemp[i] = False
    x_train = x_train0[:, maskTemp]
    x_validate = x_validate0[:, maskTemp]
    ds_train = torch.utils.data.TensorDataset(torch.from_numpy(x_train).float(),
                                              F.one_hot(torch.from_numpy(y_train)).float())
    ds_test = torch.utils.data.TensorDataset(torch.from_numpy(x_validate).float(),
                                             F.one_hot(torch.from_numpy(y_validate)).float())
    ACCs = []
    for run in range(runs):
        print('run ', run)
        dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batchSize, shuffle=True, drop_last=False)
        dl_test = torch.utils.data.DataLoader(ds_test, batch_size=x_validate.shape[0], shuffle=False)
        model = MLP3(inputs, hiddenUnits)
        optimizer = torch.optim.Adam(model.parameters(), lr=learningRate,
                                     weight_decay=l2Penalty)

        losses = []
        accuracies = []
        losses_validate = []
        accuracies_validate = []
        for epoch in range(epochs):
            for _, batch in enumerate(dl_train):
                mlp_step(model, optimizer, batch)
            accuracy, loss = mlp_eval_step(model, x_train, y_train)
            testacc, testloss = mlp_eval_step(model, x_validate, y_validate)

            if epoch % 5 == 0:
                losses.append(loss)
                losses_validate.append(testloss)
                accuracies.append(accuracy)
                accuracies_validate.append(testacc)
                # print('epoch {}, train {:.3f}, test {:.3f}'.format(epoch,accuracy,testacc))

        # plot_save_loss(losses, losses_validate, dir+'/loss_run{}.png'.format(run))
        # plot_save_acc(accuracies, accuracies_validate, dir+'/acc_run{}.png'.format(run))
        ACCs.append(accuracies_validate)
    ACCs = np.array(ACCs)
    ACC_mean = np.mean(ACCs, axis = 0)
    bestAcc = np.amax(ACC_mean)
    print('newly delete ', i, 'th feature! best test acc {:.4f}'.format(bestAcc))
