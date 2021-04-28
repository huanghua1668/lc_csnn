import torch.utils.data
from torch.nn import functional as F
import numpy as np

from sklearn.metrics import roc_curve, auc

from models import nnz, active, step, eval_step, eval_combined
from models import csnn, Net2, Net3_learnable_r, csnn_learnable_r, csnn_learnable_r_3layers, MLP3, MLP4, pre_train

# ngsim data

dir = 'data/'
f = np.load(dir + "combined_dataset.npz")
# f = np.load(dir + "combined_dataset_trainUs80_testUs101.npz")
# f = np.load(dir + "combined_dataset_trainUs101_testUs80.npz")
outputDir='results/csnn/'
# outputDir='results/train_us80_test_us101/csnn/'
# outputDir='results/train_us101_test_us80/csnn/'

x_train = f['a']
y_train = f['b']
x_test = f['c']
y_test = f['d']
x_ood = f['e']
print('{} train samples, positive rate {:.3f}'.format(x_train.shape[0], np.mean(y_train)))
print('{} test samples, positive rate {:.3f}'.format(x_test.shape[0], np.mean(y_test)))

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
# hiddenUnits = 512
hiddenUnits = 64
learningRate = 0.0001
# learningRate = 0.0001
l2Penalty = 1.0e-3

alpha = 0.
# seeds = [0, 100057, 300089, 500069, 700079]
# runs = len(seeds)
runs = 10
r2 = 1.
maxAlpha = 1.
LAMBDAS = [0., 0.01, 0.02, 0.04, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
# LAMBDAS = [0., 0.01, 0.02, 0.04, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# LAMBDAS = [0.1]
MIU = 0.0
BIAS = True

trained = False
pretrained = False
learnable_r = True
seed = 0

np.random.seed(seed)
torch.manual_seed(seed)
for LAMBDA in LAMBDAS:
    print('lambda {:.2f}'.format(LAMBDA))
    for run in range(runs):
        dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batchSize, shuffle=True, drop_last=True)
        # when batchNorm is involved in model, sample size in batch has to be > 1, so drop last just in case
        dl_test = torch.utils.data.DataLoader(ds_test, batch_size=x_test.shape[0], shuffle=False)
        model = Net3_learnable_r(inputs, hiddenUnits, bias=BIAS)
        if learnable_r:
            model.set_lambda(LAMBDA)
            model.set_miu(MIU)
        optimizer = torch.optim.Adam(model.parameters(), lr=learningRate,
                                     weight_decay=l2Penalty)
        pre_train(model, optimizer, dl_train, dl_test, x_train, y_train, x_test, y_test, run, outputDir, maxEpoch=1)

    bestValidationAccs = []
    AUCs = []
    ACCs = []
    epochs = 1000
    ALPHAs = None
    for run in range(runs):
        dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batchSize, shuffle=True, drop_last=True)
        dl_test = torch.utils.data.DataLoader(ds_test, batch_size=x_test.shape[0], shuffle=False)
        dl_combined = torch.utils.data.DataLoader(ds_combined, batch_size=x_combined.shape[0], shuffle=False)
        PATH = outputDir + '/csnn_run{}_epoch{}.pth'.format(run, 0)
        l = torch.load(PATH)
        model = l['net']
        optimizer = torch.optim.Adam(model.parameters(), lr=learningRate,
                                     weight_decay=l2Penalty)

        losses = []
        accuracies = []
        losses_test = []
        accuracies_test = []
        alphas = []
        mmcs = []
        nzs = []
        aucs = []

        for epoch in range(epochs):
            alpha = maxAlpha * epoch/epochs
            for i, batch in enumerate(dl_train):
                step(model, optimizer, batch, alpha, r2, learnable_r)
            if learnable_r:
                accuracy, loss_ce, loss_r, loss_w = eval_step(model, x_train, y_train, alpha, r2, learnable_r)
                testacc, testloss_ce, testloss_r, testloss_w = eval_step(model, x_test, y_test, alpha, r2,
                                                                         learnable_r)
            else:
                accuracy, loss = eval_step(model, x_train, y_train, alpha, r2, learnable_r)
                testacc, testloss = eval_step(model, x_test, y_test, alpha, r2, learnable_r)

            if epoch % 5 == 0:
                if learnable_r:
                    losses.append(loss_ce+loss_r+loss_w)
                    losses_test.append(testloss_ce+testloss_r+testloss_w)
                else:
                    losses.append(loss)
                    losses_test.append(testloss)
                accuracies.append(accuracy)
                accuracies_test.append(testacc)
                if learnable_r:
                    nz, mmc = nnz(x_ood, model, alpha, model.r * model.r)
                else:
                    nz, mmc = nnz(x_ood, model, alpha, r2)
                alphas.append(alpha)
                mmcs.append(mmc)
                nzs.append(nz)
                uncertainties = eval_combined(model, dl_combined, alpha, r2, learnable_r)
                falsePositiveRate, truePositiveRate, _ = roc_curve(label_ood, -uncertainties)
                AUC = auc(falsePositiveRate.astype(np.float32), truePositiveRate.astype(np.float32))
                if LAMBDA==0.1 and run==0 and AUC>=0.99 and testacc>0.8665:
                    dir = outputDir + '/ngsim_hist_confidence_epoch{}.npz'.format(epoch)
                    np.savez(dir, a=uncertainties, b=x_test.shape[0], c=epoch)
                    dir = outputDir + '/ngsim_roc_epoch{}.npz'.format(epoch)
                    np.savez(dir, a=falsePositiveRate, b=truePositiveRate, c=AUC)
                    # plot_distribution(uncertainties, x_test.shape[0], outputDir, epoch)
                    # plot_save_roc(falsePositiveRate, truePositiveRate, AUC, outputDir + 'roc_epoch_{}.png'.format(epoch))
                aucs.append(AUC)
                if learnable_r:
                    rNorm = (torch.norm(model.r, p=2)).detach().numpy()
                else:
                    rNorm = np.sqrt(r2)
                print('epoch {}, alpha {:.2f}, r2 {:.1f}, nz {:.3f}, train {:.3f}, test {:.3f}, auroc {:.3f}, ||r||2 {:.3f}'
                  .format(epoch, alpha, r2, 1.-nz,accuracy,testacc, AUC, rNorm))
                if learnable_r:
                    print('loss: cross_entropy {:.4f}, penalty {:.4f}'.format(loss_ce, loss_r))

            # eliminate dead nodes
            # if (   (epoch<200 and (epoch + 1) % (epochs / 100) == 0)
            #    or (epoch>=200 and (epoch + 1) % (epochs/10) == 0)):
            #    #_, dmu = active(torch.tensor(x_train).float(), model, alpha, r2)
            #    PATH = outputDir+'/csnn_2_csnn_layers_run{}_epoch{}_r2{:.1f}_maxAlpha{:.1f}_affine_false.pth'.format(run, epoch+1, r2, maxAlpha)
            #    torch.save({'net':model, 'alpha':alpha, 'r2':r2}, PATH)
            #    # model.keepNodes(dmu > 0)

        # plot_save_loss(losses, losses_test, outputDir+'/loss_run{}.png'.format(run))
        # plot_save_loss(losses, losses_test, outputDir+'/loss_lambda{:.2f}_run{}.png'.format(LAMBDA, run))
        # plot_save_acc(accuracies, accuracies_test, outputDir+'/acc_run{}.png'.format(run))
        # plot_save_acc(accuracies, accuracies_test, outputDir+'/acc_lambda{:.2f}_run{}.png'.format(LAMBDA, run))
        # plot_save_acc_nzs_auroc(alphas, accuracies_test, nzs, aucs,
        #                       outputDir+'/acc_nzs_mmcs_lambda{:.2f}_run{}.png'.format(LAMBDA, run))

        AUCs.append(aucs)
        ACCs.append(accuracies_test)
        if ALPHAs is None: ALPHAs = alphas
    AUCs = np.array(AUCs)
    ACCs = np.array(ACCs)
    dir = outputDir + '/mean_std_accs_aucs_net4_lambda{:.2f}.npz'.format(LAMBDA)
    np.savez(dir, a=np.mean(AUCs, axis=0), b=np.std(AUCs, axis=0),
                   c=np.mean(ACCs, axis=0), d=np.std(ACCs, axis=0),
                   e=ALPHAs)
# plt.show()
