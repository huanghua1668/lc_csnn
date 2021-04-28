import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib._color_data as mcd
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import ImageGrid

def plot_loss_functions():
    def logistic(x):
        return -np.log(1./(1.+np.exp(-x)))

    def svm(x):
        y=1.-x
        y[y<0]=0.
        return y

    def lorenz(x):
        y=np.log(1.+(x-1.)*(x-1.))
        y[x>1.]=0.
        return y

    plt.figure()
    x=np.arange(-5,4)
    plt.plot(x,logistic(x), label='logistic loss')
    plt.plot(x,svm(x), label='hinge loss')
    plt.plot(x,lorenz(x), label='lorenz')
    plt.legend()
    plt.axis([-5, 3, -0.5, 6])

    plt.show()

def plot_feature_selections():
    validation_acc = [0.846, 0.846, 0.8484, 0.846, 0.8434, 0.8434, 0.8409, 0.8333, 0.8080]
    features = [10, 9, 8, 7, 6, 5, 4, 3, 2]
    plt.figure()
    plt.plot(features, validation_acc, '-o')
    plt.axis([10, 0, 0.8, 0.85])
    plt.ylabel('validation acc')
    plt.xlabel('features')
    plt.show()


def plot_save_loss(losses, losses_validate, dir):
    plt.figure()
    plt.plot(np.arange(len(losses)) + 1, losses, label='train')
    plt.plot(np.arange(len(losses)) + 1, losses_validate, label='validate')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.axis([0, 200, 0., 0.8])
    plt.legend()
    plt.savefig(dir)


def plot_save_acc(accuracies, accuracies_validate, dir):
    plt.figure()
    plt.plot(np.arange(len(accuracies)) + 1, accuracies, label='train')
    plt.plot(np.arange(len(accuracies)) + 1, accuracies_validate, label='validate')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.axis([0, 200, 0.6, 1.])
    plt.legend()
    plt.savefig(dir)


def plot_save_acc_average_std():
    # outputDir = '/home/hh/data/ngsim/combined_dataset/MLP/'
    # outputDir = '/home/hh/data/ngsim/train_us80_test_us101/'
    outputDir = '/home/hh/data/ngsim/train_us101_test_us80/'
    dir = outputDir + '/mean_std_accs_aucs_net4.npz'
    f = np.load(dir)
    avg = f['a']
    std = f['b']
    data = np.concatenate((avg, std))
    np.savetxt(outputDir+'acc_mean_std.csv', data)
    plt.figure()
    plt.plot(5*np.arange(avg.shape[0]), avg)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.axis([0, 500, 0.6, 0.95])
    plt.savefig(outputDir+'acc.png')

def plot_save_acc_nzs_auroc():
    outputDir = '../results/'
    f = np.load(outputDir+'/moons_acc_nzs_auroc.npz')
    alphas = f['a']
    acc = f['b']
    nzs = f['c']
    auroc = f['d']
    plt.figure()
    plt.plot(alphas, acc, lw=2, label='ACC')
    plt.plot(alphas, nzs, lw=2, label='NZ')
    plt.plot(alphas, auroc, lw=2, label='AUROC')
    plt.xlabel('epoch')
    plt.ylabel('ACC, NZ, AUROC')
    plt.axis([0, 200., 0., 1.])
    plt.legend()
    dir = outputDir+'moons_acc_nzs_auroc.png'
    plt.savefig(dir)


def plot_auc():
    auc = np.round(np.loadtxt('../auc_sigma0.4.dat'), 4)
    acc = np.round(np.loadtxt('../acc_sigma0.4.dat'), 4)
    output = []
    for i in range(11):
        temp = [acc[i,0], acc[i,1], auc[i,1]]
        print(temp)
        output.append(temp)
    output = np.array(output)
    np.savetxt('../acc_auc.csv', output, fmt='%1.4f')

    plt.figure()
    plt.plot(auc[:,0], auc[:,1], '-o')
    plt.axis([0, 1.0, 0.9, 1.])
    plt.ylabel('average AUC')
    plt.xlabel('$\lambda $')

    plt.figure()
    plt.plot(acc[:,0], acc[:,1], '-o')
    plt.axis([0, 1.0, 0.8, 0.85])
    plt.ylabel('average best validation accuracy')
    plt.xlabel('$\lambda $')
    plt.show()


def plot_distribution(plotMoons = False, deep_ensemble = False):
    # def plot_distribution(score, nValidation, outputDir='/home/hh/data/', eps=0):
    # outputDir = '../results/'
    outputDir = '../results/deep_ensemble/'

    if plotMoons:
        dir = outputDir + '/moons_hist_confidence.npz'
    else:
        # dir = outputDir + '/ngsim_hist_confidence_epoch240.npz'
        dir = outputDir + '/ngsim_hist_confidence.npz'
    f = np.load(dir)
    score = f['a']
    nValidation = f['b']
    eps = f['c']
    scoreIn = score[:nValidation]
    if plotMoons:
        scoreOut = score[nValidation:][::10]
    else:
        scoreOut = score[nValidation:][::100]
    plt.figure()
    if deep_ensemble:
        plt.hist([scoreIn, scoreOut], bins=50, range=(0., .35), color=['blue','orange'], label=['in-dis','OOD'])
    else:
        plt.hist([scoreIn, scoreOut], bins=50, range=(0.5, 1.), color=['blue','orange'], label=['in-dis','OOD'])
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.ylim(0,50)
    plt.legend()
    if plotMoons:
        dir = outputDir + 'moons_score_distribution.png'
    else:
        dir = outputDir + 'ngsim_score_distribution.png'
    plt.savefig(dir)


def plot_zDiff(score, z, nValidation, flag, outputDir = '/home/hh/data/', eps = 0):
    from collections import Counter
    from matplotlib.ticker import MaxNLocator

    scoreIn = score[:nValidation]
    # scoreOut = score[nValidation:][::100]
    scoreOut = score[nValidation:][::10]
    plt.figure()
    plt.hist([scoreIn, scoreOut], bins=10, range=(0.5, .6), color=['blue','orange'], label=['in-dis','OOD'])
    plt.xlabel('score')
    plt.ylabel('Counts')
    plt.ylim(0,10)
    plt.legend()
    dir = outputDir + 'zDiff0_distribution_duq_eps{}.png'.format(eps)
    plt.savefig(dir)

    plt.figure()
    flag0 = np.ones(z.shape[0])
    flag0[nValidation:] = 0
    flag0 = flag0.astype(np.bool)
    scoreIn = z[np.logical_and(flag0, flag)]
    scoreOut = z[np.logical_and(~flag0, flag)][::10]
    counts = np.zeros((2,3))
    scoreIn = Counter(scoreIn)
    scoreIn = sorted(scoreIn.items())
    for p in scoreIn:
        counts[0][p[0]] += p[1]
    scoreOut = Counter(scoreOut)
    scoreOut = sorted(scoreOut.items())
    for p in scoreOut:
        counts[1][p[0]] += p[1]
    plt.plot([0,1,2], counts[0], 'o-', label='in-dis')
    plt.plot([0,1,2], counts[1], '*-', label='OOD')
    plt.xlabel('non-zero logits')
    plt.ylabel('Counts')
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylim(0,50)
    plt.legend()
    dir = outputDir + 'zDiff1_distribution_duq_eps{}.png'.format(eps)
    plt.savefig(dir)

def plot_func():
    epochs = 1000
    delta =200
    x=np.arange(epochs)
    y=np.exp((x-epochs)/delta)
    plt.figure()
    plt.plot(x, y)
    plt.show()

def plot_save_roc(plotMoons = False):
    # def plot_save_roc(falsePositiveRate, truePositiveRate, AUC, dir):
    outputDir = '../results/'
    outputDir = '../results/deep_ensemble/'
    if plotMoons:
        dir = outputDir + '/moons_roc.npz'
    else:
        # dir = outputDir + '/ngsim_roc_epoch240.npz'
        dir = outputDir + '/ngsim_roc.npz'
    f = np.load(dir)
    falsePositiveRate = f['a']
    truePositiveRate = f['b']
    AUC = f['c']
    plt.figure()
    plt.plot(falsePositiveRate, truePositiveRate, color='darkorange',
             lw=2, label='ROC curve (AUROC = %0.4f)' % AUC)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    if plotMoons:
        dir = outputDir + '/moons_roc.png'
    else:
        dir = outputDir + '/ngsim_roc.png'
    plt.savefig(dir)

def plot_auc_acc_mlp():
    outputDir = '/home/hh/data/ngsim/combined_dataset/deep_ensemble/'
    # outputDir='/home/hh/data/train_us80_test_us101/deep_ensemble/'
    # outputDir = '/home/hh/data/train_us101_test_us80/deep_ensemble/'
    dir = outputDir + '/mean_std_accs_aucs_net4.npz'
    f = np.load(dir)
    avg_acc = f['a'].reshape(1,-1)
    std_acc = f['b'].reshape(1,-1)
    avg_auc = f['c'].reshape(1,-1)
    std_auc = f['d'].reshape(1,-1)
    epochs = np.arange(0, 200, 5).reshape(1,-1)
    data = epochs.copy()
    data = np.concatenate((data, avg_acc))
    data = np.concatenate((data, std_acc))
    data = np.concatenate((data, avg_auc))
    data = np.concatenate((data, std_auc))
    plt.figure()
    plt.plot(epochs[0], avg_acc[0], color='darkorange', lw=2, label='accuracy')
    plt.plot(epochs[0], avg_auc[0], color='blue', lw=2, label='AUROC')
    # plt.xlim([0.0, 1.0])
    plt.xlim([0.0, 1.])
    plt.ylim([0.0, 1.])
    plt.xlabel('$\\alpha$')
    plt.ylabel('accuracy, AUROC')
    plt.legend(loc="lower right")
    dir = outputDir + 'acc_auc.png'
    plt.savefig(dir)
    data = data.transpose()
    print(data.shape)
    np.savetxt(outputDir+'acc_auc.csv', data, fmt='%.4f')

def plot_acc_mlp_feautre_selection():
    outputDir = '/home/hh/data/ngsim/combined_dataset/mlp/feature_selection/'
    dir = outputDir + '/mean_std_accs_aucs_net4_10.npz'
    f = np.load(dir)
    avg_acc = f['a']
    std_acc = f['b']
    epochs = np.arange(0, 200, 5)
    plt.figure()
    plt.plot(epochs, avg_acc, lw=2, label='accuracy')
    plt.xlim([0.0, 1.])
    plt.ylim([0.0, 1.])
    plt.xlabel('$epoch$')
    plt.ylabel('accuracy')
    plt.legend(loc="lower right")
    dir = outputDir + 'acc_10_features.png'
    plt.savefig(dir)
    data = np.vstack((epochs, avg_acc))
    data = np.vstack((data, std_acc))
    data = data.T
    print(data.shape)
    np.savetxt(outputDir+'acc_auc.csv', data, fmt='%.4f')

def plot_feautre_selection_results():
    outputDir = '../results/'
    accs = [0.876, 0.88, 0.881, 0.88, 0.877, 0.875, 0.871, 0.862, 0.836]
    features = [10, 9, 8, 7, 6, 5, 4, 3, 2]
    plt.figure()
    plt.plot(features, accs, '-o', color='darkorange', lw=2, markersize=9, mec = 'k', mfc = 'k')
    plt.xlabel('number of features')
    plt.ylim([0.83, .89])
    plt.ylabel('accuracy')
    plt.gca().invert_xaxis()
    # plt.legend(loc="lower right")
    dir = outputDir + 'feature_selection.png'
    plt.savefig(dir)

def plot_auc_acc_csnn():

    LAMBDAS = [0., 0.01, 0.02, 0.04, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    #            1.6, 1.7, 1.8, 1.9, 2.]
    # LAMBDAS = [0., 0.01, 0.02, 0.04, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # LAMBDAS = [0.1]
    # outputDir = '/home/hh/data/ngsim/combined_dataset/'
    # outputDir='/home/hh/data/ngsim/train_us80_test_us101/'
    # outputDir = '/home/hh/data/ngsim/train_us101_test_us80/'
    outputDir = '../results/csnn/'
    for LAMBDA in LAMBDAS:
        dir = outputDir + '/mean_std_accs_aucs_net4_lambda{:.2f}.npz'.format(LAMBDA)
        f = np.load(dir)
        avg_auc = f['a'].reshape(1,-1)
        std_auc = f['b'].reshape(1,-1)
        avg_acc = f['c'].reshape(1,-1)
        std_acc = f['d'].reshape(1,-1)
        ALPHAs = f['e'].reshape(1,-1)
        # print(ACCs)
        # print(AUCs)
        # if LAMBDA == 0.:
        if LAMBDA == 0.1:
            data = ALPHAs.copy()
        data = np.concatenate((data, avg_acc))
        data = np.concatenate((data, std_acc))
        data = np.concatenate((data, avg_auc))
        data = np.concatenate((data, std_auc))

        plt.figure()
        plt.plot(ALPHAs[0], avg_acc[0], color='darkorange', lw=2, label='accuracy')
        plt.plot(ALPHAs[0], avg_auc[0], color='blue', lw=2, label='AUROC')
        # plt.xlim([0.0, 1.0])
        plt.xlim([0.0, 1.])
        plt.ylim([0.0, 1.])
        plt.xlabel('$\\alpha$')
        plt.ylabel('accuracy, AUROC')
        plt.legend(loc="lower right")
        dir = outputDir + 'ngsim_acc_auc_lambda{:.2f}.png'.format(LAMBDA)
        plt.savefig(dir)
        print('lambda {:.2f}'.format(LAMBDA), ', max acc {:.4f}'.format(np.amax(avg_acc)))
    data = data.transpose()
    print(data.shape)
    # np.savetxt(outputDir+'acc_auc.csv', data, fmt='%.4f')

def plot_auc_acc_duq():

    lambdas = np.linspace(0., 1., 11)
    length_scales = np.linspace(0.1, 1., 10)
    outputDir = '/home/hh/data/ngsim/combined_dataset/duq/'
    # outputDir = '/home/hh/data/train_us101_test_us80/duq/temp/'
    # outputDir = '/home/hh/data/train_us80_test_us101/duq/'
    aucs = None
    accs = None
    max_acc = 0.
    for k in range(lambdas.shape[0]):
        for j in range(length_scales.shape[0]):
            dir = outputDir + 'mean_std_accs_aucs_lambda{:.3f}_sigma{:.3f}.npz'.format(lambdas[k], length_scales[j])
            f = np.load(dir)
            avg_auc = f['a'].reshape(1,-1)
            std_auc = f['b'].reshape(1,-1)
            avg_acc = f['c'].reshape(1,-1)
            std_acc = f['d'].reshape(1,-1)
            epochs = f['e'].reshape(1,-1)
            if aucs is None:
                aucs = epochs.copy()
                accs = epochs.copy()
            accs = np.concatenate((accs, avg_acc))
            accs = np.concatenate((accs, std_acc))
            aucs = np.concatenate((aucs, avg_auc))
            aucs = np.concatenate((aucs, std_auc))
            plt.figure()
            plt.plot(epochs[0], avg_acc[0], color='darkorange', lw=2, label='accuracy')
            plt.plot(epochs[0], avg_auc[0], color='blue', lw=2, label='AUROC')
            # plt.xlim([0.0, 1.0])
            plt.xlim([0, 400])
            plt.ylim([0.0, 1.])
            plt.xlabel('epoch')
            plt.ylabel('accuracy, AUROC')
            plt.legend(loc="lower right")
            dir = outputDir + 'acc_auc_lambda{:.3f}_sigma{:.3f}.png'.format(lambdas[k], length_scales[j])
            plt.savefig(dir)
            maxInd = np.argmax(avg_acc)
            print('lambda{:.3f}_sigma{:.3f}'.format(lambdas[k], length_scales[j]))
            print('acc {:.4f}, {:.4f}'.format(avg_acc[0, maxInd], std_acc[0, maxInd]))
            print('auc {:.4f}, {:.4f}'.format(avg_auc[0, maxInd], std_auc[0, maxInd]))
            if avg_acc[0, maxInd]>max_acc:
                max_acc = avg_acc[0, maxInd]
                max_std_acc = std_acc[0, maxInd]
                max_auc = avg_auc[0, maxInd]
                max_std_auc = std_auc[0, maxInd]

    print(aucs.shape, accs.shape)
    aucs = aucs.transpose()
    accs = accs.transpose()
    np.savetxt(outputDir+'acc.csv', accs, fmt='%.4f')
    np.savetxt(outputDir+'auc.csv', aucs, fmt='%.4f')


def plot_auc_acc_csnn_multiple_r2():
    aucs = []
    accs = []
    r2s = [0.1, 0.5, 1.0, 2.0]
    for r2 in r2s:
        dir = '/home/hh/data/mean_std_accs_aucs_csnn_2_layers_r2{:.1f}.npz'.format(r2)
        f = np.load(dir)
        AUCs = f['a']
        ACCs = f['c']
        ALPHAs = f['e']
        aucs.append(AUCs)
        accs.append(ACCs)

    plt.figure()
    for i in range(len(r2s)):
        plt.plot(ALPHAs, accs[i], lw=2, label='$r^2=${:.1f}'.format(r2s[i]))
        plt.xlim([0.0, 1.0])
        plt.ylim([0.5, 1.])
        plt.xlabel('$\\alpha$')
        plt.ylabel('ACC')
        plt.legend()
        dir = '/home/hh/data/acc_r2_impact.png'
        plt.savefig(dir)

    plt.figure()
    for i in range(len(r2s)):
        plt.plot(ALPHAs, aucs[i], lw=2, label='$r^2=${:.1f}'.format(r2s[i]))
        plt.xlim([0.0, 1.0])
        plt.ylim([0., 1.])
        plt.xlabel('$\\alpha$')
        plt.ylabel('AUC')
        plt.legend()
        dir = '/home/hh/data/auc_r2_impact.png'
        plt.savefig(dir)
    plt.show()

def plot_layer_effect():
    aucs = []
    accs = []
    outputDir = '/home/hh/data/csnn/'
    labels = ['Net2', 'Net3', 'Net4']
    alphas = None
    for net in labels:
        dir = outputDir + '/mean_std_accs_aucs_'+ net.lower()+'.npz'
        f = np.load(dir)
        AUCs = f['a']
        ACCs = f['c']
        ALPHAs = f['e']
        aucs.append(AUCs)
        accs.append(ACCs)
        if alphas is None: alphas = ALPHAs

    plt.figure()
    for i in range(len(labels)):
        plt.plot(alphas, accs[i], lw=2, label= labels[i])
        plt.xlim([0.0, 1.0])
        plt.ylim([0.5, 1.])
        plt.xlabel('$\\alpha$')
        plt.ylabel('ACC')
        plt.legend()
    dir = outputDir + 'acc_layers_impact.png'
    # plt.show()
    plt.savefig(dir)

    plt.figure()
    for i in range(len(labels)):
        plt.plot(alphas, aucs[i], lw=2, label=labels[i])
        plt.xlim([0.0, 1.0])
        plt.ylim([0., 1.])
        plt.xlabel('$\\alpha$')
        plt.ylabel('AUC')
        plt.legend()
    dir = outputDir + 'auc_layers_impact.png'
    # plt.show()
    plt.savefig(dir)

    dir = outputDir+'acc_auc_layers_impact.csv'
    aucs = np.array(aucs)
    accs = np.array(accs)
    alphas = alphas.reshape(1, -1)
    print(alphas.shape, aucs.shape, accs.shape)
    data = np.concatenate((alphas, aucs))
    data = np.concatenate((data, accs))
    data = data.transpose()
    rows = [4*i for i in range(10)]
    rows.append(39)
    np.savetxt(dir, data[rows], fmt='%.3e')
    print(data.shape)


def plot_pretrain():
    # dir0 = '/home/hh/data/train_us80_validate_us101/pre_train_acc_loss_'
    dir0 = '/home/hh/data/train_us101_validate_us80/pre_train_acc_loss_'
    nets = ['mlp3', 'mlp4', 'net3', 'net4']
    accs = []
    accs_validate = []
    losses = []
    losses_validate = []
    for net in nets:
        dir = dir0+net+'.npz'
        f = np.load(dir)
        accs.append(f['a'])
        accs_validate.append(f['e'])
        losses.append(f['c'])
        losses_validate.append(f['g'])
    epochs = np.arange(accs[0].shape[0])
    plt.figure()
    for i in range(len(nets)):
        plt.plot(epochs, accs[i], lw=2, label='train_'+nets[i])
        plt.plot(epochs, accs_validate[i], lw=2, label='validate_'+nets[i])
    plt.axis([0, 200, 0.5, 1.])
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    dir = dir0+'pre_train_acc.png'
    # plt.show()
    plt.legend()
    plt.savefig(dir)
    print('best validate acc ', np.max(accs_validate, axis=1))

    plt.figure()
    for i in range(len(nets)):
        plt.plot(epochs, losses[i], lw=2, label='train_'+nets[i])
        plt.plot(epochs, losses_validate[i], lw=2, label='validate_'+nets[i])
    plt.axis([0, 200, 0., 0.8])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    dir = dir0+'pre_train_loss.png'
    # plt.show()
    plt.legend()
    plt.savefig(dir)


def plot_train():
    # dir0 = '/home/hh/data/train_us80_validate_us101/mean_std_accs_aucs_'
    dir0 = '/home/hh/data/train_us101_validate_us80/mean_std_accs_aucs_'
    nets = ['net3', 'net4']
    aucs =[]
    accs = []
    accs_validate = []
    losses = []
    losses_validate = []
    alphas =  None
    for net in nets:
        dir = dir0+net+'.npz'
        f = np.load(dir)
        aucs.append(f['a'])
        accs.append(f['c'])
        accs_validate.append(f['e'])
        losses.append(f['g'])
        losses_validate.append(f['i'])
        if alphas is None:
            alphas = f['k']
    epochs = np.arange(accs[0].shape[0])
    plt.figure()
    for i in range(len(nets)):
        plt.plot(alphas, accs[i], lw=2, label='train_'+nets[i])
        plt.plot(alphas, accs_validate[i], lw=2, label='validate_'+nets[i])
    plt.axis([0, 1., 0.5, 1.])
    plt.xlabel('$\\alpha$')
    plt.ylabel('accuracy')
    dir = dir0+'train_acc.png'
    # plt.show()
    plt.legend()
    plt.savefig(dir)
    print('best validate acc ', np.max(accs_validate, axis=1))

    plt.figure()
    for i in range(len(nets)):
        plt.plot(alphas, losses[i], lw=2, label='train_'+nets[i])
        plt.plot(alphas, losses_validate[i], lw=2, label='validate_'+nets[i])
    plt.axis([0, 1., 0., 0.8])
    plt.xlabel('$\\alpha$')
    plt.ylabel('loss')
    dir = dir0+'train_loss.png'
    # plt.show()
    plt.legend()
    plt.savefig(dir)


    plt.figure()
    for i in range(len(nets)):
        plt.plot(alphas, accs_validate[i], lw=2, label='acc_'+nets[i])
        plt.plot(alphas, aucs[i], lw=2, label='auc_' + nets[i])
    for i in range(aucs[0].shape[0]):
        print(accs_validate[0][i], aucs[0][i])
    print('for net 2')
    for i in range(aucs[0].shape[0]):
        print(accs_validate[1][i], aucs[1][i])
    plt.axis([0, 1., 0., 1.])
    plt.xlabel('$\\alpha$')
    plt.ylabel('accuracy, auc')
    dir = dir0 + 'train_acc_auc.png'
    # plt.show()
    plt.legend()
    plt.savefig(dir)


def plot_min_distance_within_dataset():
    dir = '../data/min_dis_within_us80.npz'
    f = np.load(dir)
    dis_us80 = f['a']
    frequency_us80 = f['b']

    dir = '../data/min_dis_within_us101.npz'
    f = np.load(dir)
    dis_us101 = f['a']
    frequency_us101 = f['b']

    plt.plot(dis_us80, frequency_us80, lw=2, label='I-80')
    plt.plot(dis_us101, frequency_us101, lw=2, label='US-101')
    plt.axis([0, 3.0, 0., 1.1])
    plt.ylabel('percentage')
    plt.xlabel('minimum distance to other in-distribution samples')
    plt.legend(loc="lower right")
    dir = '../results/dist_to_in_distribution.png'
    plt.savefig(dir)
    #plt.show()


def plot_auc_score_functions():
    outputDir = '/home/hh/data/score_function/'
    dir = outputDir + '/mean_std_accs_aucs.npz'
    scores = ['logit', 'softmax', 'energy', 'logit+log_softmax']
    f = np.load(dir)
    aucs = f['a']
    alphas = f['e']
    plt.figure()
    for i in range(len(scores)):
        plt.plot(alphas, aucs[:, i], lw=2, label='auc_' + scores[i])
    plt.axis([0, 1., 0., 1.])
    plt.xlabel('$\\alpha$')
    plt.ylabel('auc')
    dir = outputDir + 'aucs.png'
    plt.legend()
    plt.show()
    plt.savefig(dir)

def plot_learnable_r():
    outputDir = '/home/hh/data/moons/radius_penalty_impact/lambda_0.01/temp'
    # outputDir = '/home/hh/data/two_gaussian/radius_penalty_impact/lambda_0.00/temp'
    dir = outputDir + '/mean_std_accs_aucs_net4.npz'
    f = np.load(dir)
    rs = f['l']
    print(rs.shape)
    epochs = (np.arange(rs.shape[0])+1)*5
    plt.figure()
    plt.plot(epochs, rs[:,0], label='$||r||_\infty$')
    plt.plot(epochs, rs[:,1]/8., label='$||r||_2/\sqrt{n_{hidden}}$')
    plt.axis([0, 500, 0., 1.8])
    plt.ylabel('r norm')
    plt.xlabel('epochs')
    plt.legend()
    plt.show()
    # dir = outputDir + '/learnable_r.png'
    # plt.savefig(dir)


def plot_circles(layer, x_train, y_train, alpha, r, epoch, dir0, bias=False):
    figure, ax = plt.subplots()
    mask = y_train.astype(np.bool)
    w = layer.weight.data.numpy()
    # w is of shape (64,2)
    center = w/alpha
    radius2 = r*r + np.sum(w*w, axis=1)*(1/alpha/alpha-1)
    if bias:
        b = layer.bias.data.numpy()
        # b is of shape (64,)
        # radius2 -= (1.-b/alpha)*(1.-b/alpha)
        radius2 += b*b*(1/alpha/alpha-1)
        # radius2[radius2<0.] = 0.
    radius = np.sqrt(radius2)
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    N = w.shape[0]
    col = 1. / N * np.arange(0, N)
    for i in range(w.shape[0]):
        # circle = plt.Circle(center[i], radius[i], fill=False)
        # circle = plt.Circle(center[i], radius[i], color=str(col[i]))
        circle = plt.Circle(center[i], radius[i], color=cm.jet(col[i]), fill=False)
        ax.add_artist(circle)
        ax.axis('equal')
        # ax.autoscale(False)

    plt.scatter(x_train[mask, 0], x_train[mask, 1], zorder=1)
    plt.scatter(x_train[~mask, 0], x_train[~mask, 1], zorder=1)
    dir = dir0 + '/confidence_circle_epoch_{}.png'.format(epoch)
    # mean = [0.50048237, 0.24869538]
    # std = [0.8702328,  0.50586896]
    # x_range = [-2.5, 3.5]
    # y_range = [-3., 3.]
    ax.set_xlim([-3.,3.])
    ax.set_ylim([-3.,3.])
    # plt.axis([-3, 3., -3, 3])
    plt.savefig(dir)

def plot_two_gaussian(x_train):
    figure, ax = plt.subplots()
    mask = x_train[:, -1] == 0
    plt.scatter(x_train[mask, 0], x_train[mask, 1], zorder=1)
    plt.scatter(x_train[~mask, 0], x_train[~mask, 1], zorder=1)
    ax.set_xlim([-8.,8.])
    ax.set_ylim([-8.,8.])
    # plt.show()
    # plt.axis([-3, 3., -3, 3])
    dir = '/home/hh/data/two_gaussian/'
    plt.savefig(dir+'two_gaussian.png')

def plot_radius_penalty_impact_on_acc():
    dir0 = '/home/hh/data/two_gaussian/radius_penalty_impact/'
    lambdas = [0.0, 0.02, 0.08, 0.16, 0.32, 0.64, 1.28]
    accs = []
    accs_valid = []
    for l in lambdas:
        dir = dir0+'lambda_{:.2f}'.format(l)
        dir = dir + '/mean_std_accs_aucs_net4.npz'
        data = np.load(dir)
        acc = np.mean(data['c'][-10:])
        acc_valid = np.mean(data['e'][-10:])
        accs.append(acc)
        accs_valid.append(acc_valid)
    figure, ax = plt.subplots()
    plt.plot(lambdas, accs, label='train')
    plt.plot(lambdas, accs_valid, label='test')
    plt.legend()
    plt.ylabel('Accuracy')
    plt.xlabel('$\lambda$')
    # plt.show()
    plt.savefig(dir0+'radius_penalty_on_accuracy.png')

def plot_confidence_map_alpha_impact():
    # Figure 4.1: Confidence map for α = 0 (left) and α = 1 (right)
    dir0 = '../results/'
    dir = dir0 + 'moons_confidence_alpha0.npz'
    data = np.load(dir)
    x_lin = data['a']
    y_lin = data['b']
    z0 = data['c']
    X_vis = data['d']
    mask = data['e']
    dir = dir0 + 'moons_confidence_alpha1.npz'
    data = np.load(dir)
    z1 = data['c']

    z = [z0, z1]
    axs = []
    fig = plt.figure()
    l = np.linspace(0.5, 1., 21)
    titles = ['$\\alpha=0$', '$\\alpha=1$']
    fig, axs = plt.subplots(1, 2)
    fig = plt.figure()
    grid = ImageGrid(fig, 111, nrows_ncols=(1,2), axes_pad=0.15,
                      share_all=True, cbar_location="right",
                      cbar_mode="single", cbar_size="7%",
                      cbar_pad=0.15)
    for i in range(len(z)):
        axs = grid[i]
        img = axs.contourf(x_lin, y_lin, z[i], cmap=plt.get_cmap('inferno'), levels=l)  # , extend='both')
        axs.scatter(X_vis[mask, 0][::5], X_vis[mask, 1][::5], s=6, c='r')
        axs.scatter(X_vis[~mask, 0][::5], X_vis[~mask, 1][::5], s=6)
        axs.set_title(titles[i])
        axs.set(xlim=(-2.4, 2.4), ylim=(-2.4, 2.4))
        axs.set_aspect('equal')
    axs.cax.colorbar(img)
    axs.cax.toggle_label(True)
    dir = dir0 + 'confidence_alpha_impact.png'
    plt.savefig(dir)

def plot_score_csnn_de():
    dir0 = '/home/hh/data/ngsim/uncertainty_separation/'
    data = np.load(dir0+'confidence_csnn.npz')
    score_csnn = data['a']
    data = np.load(dir0+'score_de.npz')
    score_de = data['a']
    fig = plt.figure()
    plt.scatter(score_csnn, -score_de)
    plt.xlabel('confidence_csnn')
    plt.ylabel('entropy_de')
    # plt.show()
    dir = dir0 + 'confidence_csnn_de.png'
    plt.savefig(dir)


def plot_radius_penalty_effect():
    import sklearn.datasets
    import torch
    import sys
    sys.path.append('../')
    from models import csnn_learnable_r

    # lambdas = [0.00, 0.01, 0.02, 0.04, 0.16, 0.64]
    lambdas = [0.00, 0.02, 0.04, 0.64]
    outputDir = '../results/radius_penalty_impact/'

    # load models
    models = []
    for l in lambdas:
        model = csnn_learnable_r(2, 64, bias=False)
        PATH = outputDir+'lambda_{:.2f}/csnn_run0_epoch500.pth'.format(l)
        model.load_state_dict(torch.load(PATH))
        models.append(model)
    print('load models successfully')

    # load rs
    rs = []
    for l in lambdas:
        f = np.load(outputDir+'lambda_{:.2f}/mean_std_accs_aucs_net4.npz'.format(l))
        rs.append(f['l'])
    print('load radius successfully')

    # load confidences
    confidences = []
    x_lin = None
    y_lin = None
    for l in lambdas:
        f = np.load(outputDir+'lambda_{:.2f}/moons_confidence_alpha1.npz'.format(l))
        confidences.append(f['c'])
        if x_lin is None:
            x_lin = f['a']
            y_lin = f['b']
    print('load confidences successfully')

    # Moons
    noise = 0.1
    # sklearn has no random seed, it depends on numpy to get random numbers
    np.random.seed(0)
    x_train, y_train = sklearn.datasets.make_moons(n_samples=1500, noise=noise)
    x_train0 = x_train
    mask = y_train.astype(np.bool)
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    print('mean, std', mean, std)
    x_train = (x_train-mean)/std/np.sqrt(2)

    fig, axs = plt.subplots(len(lambdas), 3, figsize=(10, 10.6), dpi=300)
    alpha = 1
    for i in range(len(lambdas)):
        # support circles
        w = models[i].fc1.weight.data.numpy()
        r = models[i].r.detach().numpy()
        center = w/alpha
        radius2 = r*r + np.sum(w*w, axis=1)*(1/alpha/alpha-1)
        radius = np.sqrt(radius2)
        N = w.shape[0]
        col = 1. / N * np.arange(0, N)
        for j in range(w.shape[0]):
            circle = plt.Circle(center[j], radius[j], color=cm.jet(col[j]), fill=False)
            axs[i, 0].add_artist(circle)

        axs[i, 0].scatter(x_train[mask, 0], x_train[mask, 1],  s=1, c='r')
        axs[i, 0].scatter(x_train[~mask, 0], x_train[~mask, 1], s=1)
        axs[i, 0].set(xlim=(-2.4, 2.4), ylim=(-2.4, 2.4))
        axs[i, 0].set_aspect('equal')
        if i != len(lambdas)-1: axs[i, 0].set_xticklabels([])

        # confidence
        level = np.linspace(0.5, 1., 21)
        contour = axs[i, 1].contourf(x_lin, y_lin, confidences[i], cmap=plt.get_cmap('inferno'), levels=level)  # , extend='both')
        fig.colorbar(contour, ax = axs[i, 1], format='%.2f')
        axs[i, 1].scatter(x_train[mask, 0], x_train[mask, 1], s=1, c='r')
        axs[i, 1].scatter(x_train[~mask, 0], x_train[~mask, 1], s=1)
        axs[i, 1].set(xlim=(-2.4, 2.4), ylim=(-2.4, 2.4))
        axs[i, 1].set_aspect('equal')
        axs[i, 1].set_yticklabels([])
        if i != len(lambdas)-1: axs[i, 1].set_xticklabels([])

        # radius
        epochs = np.arange(101)*5
        axs[i, 2].plot(epochs, rs[i][:,0], lw=2, label='$||r||_\infty$')
        axs[i, 2].plot(epochs, rs[i][:,1]/8., lw=2, label='$||r||_2/\sqrt{n_{hidden}}$')
        axs[i, 2].fill_between(epochs, rs[i][:,1]/8.-rs[i][:,2], rs[i][:,1]/8.+rs[i][:,2], facecolor='orange', alpha=0.2)
        axs[i, 2].axis([0, 500, 0., 1.8])
        axs[i, 2].set_ylabel('r norm')
        if i == len(lambdas)-1: axs[i, 2].set_xlabel('epochs')
        axs[i, 2].legend()
        axs[i, 2].yaxis.tick_right()
        axs[i, 2].yaxis.set_label_position("right")
        if i != len(lambdas)-1: axs[i, 2].set_xticklabels([])

    plt.savefig(outputDir+'evolvement.png', dpi=300)


def plot_evolvement():
    import sklearn.datasets
    import torch
    import sys
    sys.path.append('../')
    from models import csnn_learnable_r

    epochs = [0, 5, 10, 50, 100, 500]
    outputDir = '../results/evolvement/'

    # load models
    models = []
    for epoch in epochs:
        model = csnn_learnable_r(2, 64, bias=False)
        PATH = outputDir + 'csnn_epoch{}.pth'.format(epoch)
        model.load_state_dict(torch.load(PATH))
        models.append(model)
    print('load models successfully')

    # load confidences
    confidences = []
    x_lin = None
    y_lin = None
    for epoch in epochs:
        dir = outputDir + 'moons_confidence_alpha1_epoch{}.npz'.format(epoch)
        f = np.load(dir)
        confidences.append(f['c'])
        if x_lin is None:
            x_lin = f['a']
            y_lin = f['b']
    print('load confidences successfully')

    # Moons
    noise = 0.1
    # sklearn has no random seed, it depends on numpy to get random numbers
    np.random.seed(0)
    x_train, y_train = sklearn.datasets.make_moons(n_samples=1500, noise=noise)
    x_train0 = x_train
    mask = y_train.astype(np.bool)
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    print('mean, std', mean, std)
    x_train = (x_train-mean)/std/np.sqrt(2)

    # support circles
    fig, axs = plt.subplots(2, 3, figsize=(10, 6.67), dpi=300)
    alpha = 1
    for i in range(len(epochs)):
        row = i//3
        col = i%3
        w = models[i].fc1.weight.data.numpy()
        r = models[i].r.detach().numpy()
        center = w/alpha
        radius2 = r*r + np.sum(w*w, axis=1)*(1/alpha/alpha-1)
        radius = np.sqrt(radius2)
        # cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        N = w.shape[0]
        color = 1. / N * np.arange(0, N)
        for j in range(w.shape[0]):
            circle = plt.Circle(center[j], radius[j], color=cm.jet(color[j]), fill=False)
            axs[row, col].add_artist(circle)
            # axs[0].axis('equal')

        axs[row, col].scatter(x_train[mask, 0], x_train[mask, 1],  s=1, c='r')
        axs[row, col].scatter(x_train[~mask, 0], x_train[~mask, 1], s=1)
        axs[row, col].set(xlim=(-2.4, 2.4), ylim=(-2.4, 2.4))
        axs[row, col].set_aspect('equal')
        if row == 0: axs[row, col].set_xticklabels([])
        if col != 0: axs[row, col].set_yticklabels([])
        axs[row, col].set_title('epoch={}'.format(epochs[i]))
        plt.savefig(outputDir + 'evolvement_circles.png', dpi=300)

    # confidence
    fig, axs = plt.subplots(2, 3, figsize=(10, 5), dpi=300)
    for i in range(len(epochs)):
        row = i // 3
        col = i % 3
        level = np.linspace(0.5, 1., 21)
        contour = axs[row, col].contourf(x_lin, y_lin, confidences[i], cmap=plt.get_cmap('inferno'), levels=level)  # , extend='both')
        fig.colorbar(contour, ax = axs[row, col], format='%.2f')
        axs[row, col].scatter(x_train[mask, 0], x_train[mask, 1], s=1, c='r')
        axs[row, col].scatter(x_train[~mask, 0], x_train[~mask, 1], s=1)
        axs[row, col].set(xlim=(-2.4, 2.4), ylim=(-2.4, 2.4))
        axs[row, col].set_aspect('equal')
        if row == 0: axs[row, col].set_xticklabels([])
        if col != 0: axs[row, col].set_yticklabels([])
        axs[row, col].set_title('epoch={}'.format(epochs[i]))
        plt.savefig(outputDir + 'evolvement_conf.png', dpi=300)


def plot_moons_scatter():
    outputDir = '../results/'
    f = np.load(outputDir + 'moons_train_test_ood.npz' )
    x_train = f['a']
    y_train = f['b']
    xOOD = f['e']

    plt.figure()
    mask = y_train.astype(np.bool)
    plt.scatter(x_train[mask, 0], x_train[mask, 1], s=6)
    plt.scatter(x_train[~mask, 0], x_train[~mask, 1], s=6)
    plt.scatter(xOOD[:, 0], xOOD[:, 1], s=6)
    axs = plt.gca()
    axs.set(xlim=(-2.4, 2.4), ylim=(-2.4, 2.4))
    axs.set_aspect('equal')
    dir = outputDir + '/moons_inDis_ood_samples.png'
    plt.savefig(dir)
    # plt.show()

def plot_trajectory():
    import numpy as np
    import matplotlib.pyplot as plt

    data = np.genfromtxt('../data/lane_changes_trajectories.csv', delimiter=',')
    start = 0
    end = np.searchsorted(data[:, 0], 1) - 1
    plt.plot(data[start:end + 1, 3], data[start:end + 1, 2])
    for i in range(1, int(data[-1, 0]) + 1):
        start = end + 1
        if i == int(data[-1, 0]):
            end = data.shape[0] - 1
        else:
            end = np.searchsorted(data[:, 0], i + 1) - 1
        # if abs(data[end,2]-data[start,2])<1.85:
        #    plt.plot(data[start:end+1, 3]-data[start,3],data[start:end+1,2])
        plt.plot(data[start:end + 1, 3], data[start:end + 1, 2])
        if i == 200: break
    plt.ylabel('$\Delta y \enspace [m]$')
    plt.xlabel('$\Delta x \enspace [m]$')
    plt.axis([-60, 70, -4., 4.])
    dir =  '../results/trajectories.png'
    plt.savefig(dir)
    # plt.show()


def visualize_sample_labeled_by_dx():
    f = np.load('../data/samples_relabeled_by_decrease_in_dx.npz')
    records = f['a']

    samples0=records[records[:,-1]==0]
    samples1=records[records[:,-1]==1]
    plt.figure(figsize=(10, 5), dpi=300)
    plt.scatter(samples0[:,1], samples0[:,0], color="black", marker='x',
                label='adv', s=3)
    plt.scatter(samples1[:,1], samples1[:,0], color="blue", marker='o',
                label='coop', s=3)
    plt.ylabel('$\Delta x \enspace [m]$', fontsize=15)
    plt.xlabel('$\Delta v \enspace [m/s^2]$', fontsize=15)
    plt.axis([-15, 15, -50, 100])
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=15)
    dir = '../results/samples_relabeled_by_decrease_in_dx.png'
    plt.savefig(dir)
    # plt.show()


def plot_accelerations():
    data = np.genfromtxt('../data/lane_changes_trajectories.csv', delimiter=',')
    start = 0
    end = np.searchsorted(data[:, 0], 1) - 1
    for i in range(0, 3):
        start = np.searchsorted(data[:, 0], i)
        if i == int(data[-1, 0]):
            end = data.shape[0] - 1
        else:
            end = np.searchsorted(data[:, 0], i + 1) - 1
        start0 = start
        end0 = end
        while (start0 < end0 and data[start0, -4] == 0.): start0 += 1
        if start0 == end0: continue
        while (start0 < end0 and data[end0, -4] == 0.): end0 -= 1
        if (data[end0, 3] - data[end0, -3] < 5.):
            print(i, data[end0, 3] - data[end0, -3])
            continue

        plt.plot(data[start0:end0 + 1, 1],
                 data[start0:end0 + 1, -2])
        plt.ylabel('$\\alpha \enspace [m/s^2]$')
        plt.xlabel('t $\enspace [s]$')
        plt.axis([-2, 5, -4, 4])

    dir = '../results/accelerations.png'
    plt.savefig(dir)
    # plt.show()


def plot_ngsim_scatter():
    def helper(f):
        '''
            input: [merge_front/after(0/1), u0, du0, du1, du2, dx0, dx1, dx2, dy0, dy1, dy2, y]
        '''

        data0 = f['a']
        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        data = data0[:, 1:-1]
        # print(data[:5])
        merge_after = data0[:, 0].astype(int)
        label = data0[:, -1]
        data00 = data[np.logical_and(label == 0, merge_after == 0)]
        data01 = data[np.logical_and(label == 0, merge_after == 1)]
        data1 = data[label == 1]
        ax.scatter(data00[:, 1], data00[:, 4], c='black', marker='x', linewidths=0.5, s=4, label='adv (merge infront)')
        ax.scatter(data01[:, 1], data01[:, 4], c='red', marker='x', linewidths=0.5, s=4, label='adv (merge after)')
        ax.scatter(data1[:, 1], data1[:, 4], c='blue', marker='o', s=4, label='coop')
        # print(data00[:10, [1,4]])

        plt.ylabel('$\Delta x \enspace [m]$', fontsize=15)
        plt.xlabel('$\Delta v \enspace [m/s^2]$', fontsize=15)
        plt.axis([-15, 15, -50, 100])
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(fontsize=12)
        ax.legend(prop={'size': 15})
        # outputDir = dir + 'us80.png'
        # plt.savefig(outputDir)

    f=np.load('../data/us80.npz')
    helper(f)
    dir = '../results/us80.png'
    plt.savefig(dir)
    f=np.load('../data/us101.npz')
    helper(f)
    dir = '../results/us101.png'
    plt.savefig(dir)

def plot_moons_de():
    outputDir = '../results/'
    f = np.load(outputDir + "moons_confidence_map_deep_ensemble.npz")
    x_lin = f['a']
    y_lin = f['b']
    z = f['c']
    x_train = f['d']
    y_train = f['e']
    plt.figure()
    l = np.linspace(0, 1., 21)
    # plt.contourf(x_lin, y_lin, z, cmap=cmaps.cividis)
    # cntr = plt.contourf(x_lin, y_lin, z, cmap=plt.get_cmap('inferno'), levels=l, extend='both')
    cntr = plt.contourf(x_lin, y_lin, z, cmap=plt.get_cmap('inferno'), levels=l)
    plt.colorbar()
    # plt.contourf(x_lin, y_lin, z)
    # plt.contourf(x_lin, y_lin, z)
    mask = y_train==1
    plt.scatter(x_train[mask,0][::2], x_train[mask,1][::2], s=6, c='r')
    plt.scatter(x_train[~mask,0][::2], x_train[~mask,1][::2], s=6)
    plt.xlim(-2.4, 2.4)
    plt.ylim(-2.4, 2.4)
    dir = outputDir + 'moons_confidence_de.png'
    plt.savefig(dir)

if __name__ == "__main__":
    # plot_auc_acc_mlp()
    # plot_auc_acc_csnn()
    # plot_auc_acc_duq()
    # plot_min_distance_within_dataset()
    # plot_pretrain()
    # plot_train()
    # plot_auc_score_functions()
    # plot_layer_effect()
    # plot_learnable_r()
    # plot_radius_penalty_impact_on_acc()
    # plot_confidence_map_alpha_impact()
    # plot_save_acc_average_std()
    # plot_save_acc_nzs_auroc()
    # plot_distribution(plotMoons=False)
    # plot_save_roc(plotMoons=False)
    # plot_score_csnn_de()
    # plot_acc_mlp_feautre_selection()
    # plot_feautre_selection_results()
    # plot_radius_penalty_effect()
    # plot_evolvement()
    # plot_moons_scatter()
    # plot_trajectory()
    # visualize_sample_labeled_by_dx()
    # plot_accelerations()
    # plot_ngsim_scatter()
    plot_moons_de()
