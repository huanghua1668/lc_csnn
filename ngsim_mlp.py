import torch.utils.data
from torch.nn import functional as F
import numpy as np

from sklearn.metrics import roc_curve, auc

from models import MLP2, MLP3, mlp_step, mlp_eval_step, mlp_eval_combined

dir = 'data/'
f = np.load(dir + "combined_dataset.npz")
# f = np.load(dir + "combined_dataset_trainUs80_testUs101.npz")
# f = np.load(dir + "combined_dataset_trainUs101_testUs80.npz")
outputDir='results/'
# outputDir='results/train_us80_test_us101/'
# outputDir='results/train_us101_test_us80/'

x_train = f['a']
y_train = f['b']
x_test = f['c']
y_test = f['d']
x_ood = f['e']
print('{} train samples, positive rate {:.3f}'.format(x_train.shape[0], np.mean(y_train)))
print('{} test samples, positive rate {:.3f}'.format(x_test.shape[0], np.mean(y_test)))

dim = x_train.shape[1]
x_train = x_train/np.sqrt(dim)
x_test = x_test/np.sqrt(dim)
x_ood = x_ood/np.sqrt(dim)
x_combined = np.concatenate((x_test, x_ood))

label_ood = np.zeros(x_combined.shape[0])
label_ood[x_test.shape[0]:] = 1

ds_train = torch.utils.data.TensorDataset(torch.from_numpy(x_train).float(),
                                          F.one_hot(torch.from_numpy(y_train)).float())

ds_test = torch.utils.data.TensorDataset(torch.from_numpy(x_test).float(),
                                         F.one_hot(torch.from_numpy(y_test)).float())

ds_combined = torch.utils.data.TensorDataset(torch.from_numpy(x_combined).float())

# network hyper param
inputs = 4
batchSize = 64
hiddenUnits = 64
learningRate = 0.001
l2Penalty = 1.0e-3

seed = 0
runs = 10
BIAS = False

trained = False

bestValidationAccs = []
ACCs = []
AUCs = []
epochs = 200
np.random.seed(seed)
torch.manual_seed(seed)
for run in range(runs):
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batchSize, shuffle=True, drop_last=False)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=x_test.shape[0], shuffle=False)
    dl_combined = torch.utils.data.DataLoader(ds_combined, batch_size=x_combined.shape[0], shuffle=False)
    model = MLP3(inputs, hiddenUnits, bias=BIAS)
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate,
                                 weight_decay=l2Penalty)

    losses = []
    accuracies = []
    losses_test = []
    accuracies_test = []
    aucs = []
    for epoch in range(epochs):
        for i, batch in enumerate(dl_train):
            mlp_step(model, optimizer, batch)
        accuracy, loss = mlp_eval_step(model, x_train, y_train)
        testacc, testloss = mlp_eval_step(model, x_test, y_test)

        if epoch % 5 == 0:
            losses.append(loss)
            losses_test.append(testloss)
            accuracies.append(accuracy)
            accuracies_test.append(testacc)
            uncertainties = mlp_eval_combined(model, dl_combined)
            falsePositiveRate, truePositiveRate, _ = roc_curve(label_ood, -uncertainties)
            AUC = auc(falsePositiveRate.astype(np.float32), truePositiveRate.astype(np.float32))
            aucs.append(AUC)
            print('epoch {}, train {:.3f}, test {:.3f}, auc {:.3f}'.format(epoch,accuracy,testacc, AUC))

    # plot_save_loss(losses, losses_test, outputDir+'/loss_run{}.png'.format(run))
    # plot_save_acc(accuracies, accuracies_test, outputDir+'/acc_run{}.png'.format(run))
    AUCs.append(aucs)
    ACCs.append(accuracies_test)

ACCs = np.array(ACCs)
AUCs = np.array(AUCs)
dir = outputDir + '/mlp_mean_std_accs_aucs_net4.npz'
np.savez(dir, a=np.mean(ACCs, axis=0), b=np.std(ACCs, axis=0),
              c=np.mean(AUCs, axis=0), d=np.std(AUCs, axis=0) )
# plt.show()