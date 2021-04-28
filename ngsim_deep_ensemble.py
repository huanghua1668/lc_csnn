import torch
import torch.utils.data
from torch.nn import functional as F
import torch.optim as optim

from sklearn.metrics import roc_curve, auc

import numpy as np

from models import MLP3, mlp_step#, mlp_eval_step, mlp_eval_combined
# don't import the general mlp_eval_step, as it will output acc and loss, here we also need probs

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

if __name__ == "__main__":
    # ngsim data
    dir = 'data/'
    f = np.load(dir + "combined_dataset.npz")
    # f = np.load(dir + "combined_dataset_trainUs80_testUs101.npz")
    # f = np.load(dir + "combined_dataset_trainUs101_testUs80.npz")
    outputDir = 'results/deep_ensemble/'
    # outputDir='results/train_us80_test_us101/deep_ensemble/'
    # outputDir = 'results/train_us101_test_us80/deep_ensemble/'
    x_train = f['a']
    y_train = f['b']
    x_test = f['c']
    y_test = f['d']
    x_ood = f['e']
    print('{} train samples, positive rate {:.3f}'.format(x_train.shape[0], np.mean(y_train)))
    print('{} test samples, positive rate {:.3f}'.format(x_test.shape[0], np.mean(y_test)))
    print('{} ood samples'.format(x_ood.shape[0]))
    x_combined = np.concatenate((x_test, x_ood))
    label_ood = np.zeros(x_combined.shape[0])
    label_ood[x_test.shape[0]:] = 1
    x_train = torch.tensor(x_train, dtype=torch.float)
    x_test = torch.tensor(x_test, dtype=torch.float)
    x_combined = torch.tensor(x_combined, dtype=torch.float)

    ds_train = torch.utils.data.TensorDataset(x_train, F.one_hot(torch.from_numpy(y_train)).float())

    ds_test = torch.utils.data.TensorDataset(x_test, F.one_hot(torch.from_numpy(y_test)).float())

    ds_combined = torch.utils.data.TensorDataset(x_combined)


    # network hyper param
    inputs = 4
    batchSize = 64
    # epochs = 200
    epochs = 201
    hiddenUnits = 64
    learningRate = 0.001
    # learningRate = 0.0001
    l2Penalty = 1.0e-3
    num_classes = 2


    seed = 0
    runs = 10
    ensembles = 10
    models = []
    optimizers = []

    trained = False
    trained = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    scores = None # output score when acc is largest for in-dis samples

    AUCs = []
    ACCs = []
    for i in range(runs):
        bestAcc = 0.8
        accs = []
        aucs = []
        probs = []

        # train and predict
        for j in range(ensembles):
            losses = []
            accuracies = []
            losses_test = []
            accuracies_test = []
            prob = []
            print('run {} ensemble {}'.format(i, j))
            dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batchSize, shuffle=True, drop_last=False)
            dl_test = torch.utils.data.DataLoader(ds_test, batch_size=x_test.shape[0], shuffle=False)
            dl_combined = torch.utils.data.DataLoader(ds_combined, batch_size=x_combined.shape[0], shuffle=False)
            model = MLP3(inputs, hiddenUnits)
            optimizer = optim.Adam(model.parameters(), lr=learningRate,
                                   weight_decay=l2Penalty)
            for epoch in range(epochs):
                for k, batch in enumerate(dl_train):
                    mlp_step(model, optimizer, batch)
                if epoch%5 != 0: continue
                accuracy, loss, _  = mlp_eval_step(model, x_train, y_train)
                losses.append(loss)
                accuracies.append(accuracy)
                accuracy_test, loss_test, probs_test = mlp_eval_step(model, x_test, y_test)
                losses_test.append(loss_test)
                accuracies_test.append(accuracy_test)
                print('epoch {}, train acc {:.3f}, train loss {:.3f}, test acc {:.3f}, test loss {:.3f}'
                      .format(epoch, accuracy, loss, accuracy_test, loss_test))
                probs_combined = mlp_eval_combined(model, x_combined)
                prob.append(probs_combined)
            prob = np.array(prob)
            if i == 0 and j == 0: print('prob shape', prob.shape)
            probs.append(prob)
            # dir = '/home/hh/data/loss_deep_ensemble_run{}_ensemble{}.npz'.format(i, j)
            # np.savez(dir, a=np.array(losses), b=np.array(losses_test))
            # dir = '/home/hh/data/acc_deep_ensemble_run{}_ensemble{}.npz'.format(i, j)
            # np.savez(dir, a=np.array(accuracies), b=np.array(accuracies_test))
        probs = np.array(probs)
        if i == 0: print('probs shape', probs.shape)
        prob = np.mean(probs, axis=0)
        if i == 0: print('prob shape', prob.shape)

        # now calculate the test acc and AUC
        acc = np.argmax(prob[:, :x_test.shape[0]], axis=2)
        print('acc shape ', acc.shape)
        temp = acc==y_test
        acc = np.mean(acc==y_test, axis=1)
        #  # uncertainty = -np.sum(prob*np.log(prob), axis=1)
        #  # falsePositiveRate, truePositiveRate, _= roc_curve(label_ood, uncertainty)
        score = prob * np.log(prob)
        score = score.mean(axis=2)
        bestAcc = 0.87
        for j in range(score.shape[0]):
            epoch = 5*j
            falsePositiveRate, truePositiveRate, _= roc_curve(label_ood, -score[j])
            AUC = auc(falsePositiveRate.astype(np.float32), truePositiveRate.astype(np.float32))
            aucs.append(AUC)
            print('epoch {}, acc {:.4f}, auc {:.4f}'.format(epoch, acc[j], AUC))
            if i==0 and acc[j]>bestAcc:
                bestAcc = acc[j]
                dir = outputDir + '/ngsim_hist_confidence.npz'
                np.savez(dir, a=-score[j], b=x_test.shape[0], c=epoch)
                dir = outputDir + '/ngsim_roc.npz'
                np.savez(dir, a=falsePositiveRate, b=truePositiveRate, c=AUC)
        ACCs.append(acc)
        AUCs.append(aucs)

    AUCs = np.array(AUCs)
    ACCs = np.array(ACCs)
    print('AUCs, ACCs shape', AUCs.shape, ACCs.shape)
    dir = outputDir + '/mean_std_accs_aucs_net4.npz'
    np.savez(dir, a=np.mean(AUCs, axis=0), b=np.std(AUCs, axis=0),
             c=np.mean(ACCs, axis=0), d=np.std(ACCs, axis=0))
    # np.savez(outputDir+'/temp/score_de.npz', a=scores)
