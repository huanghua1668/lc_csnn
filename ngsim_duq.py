import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
import numpy as np

from sklearn.metrics import roc_curve, auc


class Model_bilinear(nn.Module):
    def __init__(self, inputs, features, num_embeddings, sigma, gamma=0.99, embedding_size=64, nInit=20):

        # features are number of hidden units...
        super().__init__()

        self.gamma = gamma
        self.sigma = sigma

        self.fc1 = nn.Linear(inputs, features)
        self.W = nn.Parameter(torch.normal(torch.zeros(embedding_size, num_embeddings, features), 1))
        self.register_buffer('N', torch.ones(num_embeddings) * nInit)
        self.register_buffer('m', torch.normal(torch.zeros(embedding_size, num_embeddings), 1))
        ###
        # If you have parameters in your model, which should be saved and restored in the state_dict,
        # but not trained by the optimizer, you should register them as buffers.
        # Buffers won’t be returned in model.parameters(), so that the optimizer won’t have a change to update them.
        ###

        self.m = self.m * self.N.unsqueeze(0)

    def embed(self, x):
        batchNorm = False
        if batchNorm:
            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.bn2(self.fc2(x)))
            x = self.bn3(self.fc3(x))
        else:
            x = F.relu(self.fc1(x))
        # i is batch, m is embedding_size, n is num_embeddings (classes)
        x = torch.einsum('ij,mnj->imn', x, self.W)

        return x

    def bilinear(self, z):
        # z is W_c.f(x)
        # embeddings is e_c
        embeddings = self.m / self.N.unsqueeze(0)
        # embeddings size (embedding_size, num_embeddings)

        diff = z - embeddings.unsqueeze(0)
        y_pred = (- diff ** 2).mean(1).div(2 * self.sigma ** 2).exp()

        return y_pred

    def forward(self, x):
        z = self.embed(x)
        y_pred = self.bilinear(z)

        return z, y_pred

    def update_embeddings(self, x, y):
        z = self.embed(x)

        # normalizing value per class, assumes y is one_hot encoded
        self.N = torch.max(self.gamma * self.N + (1 - self.gamma) * y.sum(0), torch.ones_like(self.N))

        # compute sum of embeddings on class by class basis
        features_sum = torch.einsum('ijk,ik->jk', z, y)

        self.m = self.gamma * self.m + (1 - self.gamma) * features_sum


def calc_gradient_penalty(x, y_pred):
    gradients = torch.autograd.grad(
        outputs=y_pred,
        inputs=x,
        grad_outputs=torch.ones_like(y_pred),
        create_graph=True,
        # retain_graph=True,
    )[0]
    # grad_outputs is the vector [[1],[1]] for matrix vector product
    # x (batch_size, 2), y_pred (batch_size, 2), gradients (batch_size, 2)
    # gradients = [d(y1+y2)/dx1, d(y1+y2)/dx2]
    gradients = gradients.flatten(start_dim=1)
    # no need to flatten here? as start_dim=1 for shape(batch_size, 2) will do nothing

    # L2 norm
    grad_norm = gradients.norm(2, dim=1)

    # Two sided penalty
    gradient_penalty = ((grad_norm - 1.) ** 2).mean()
    # gradient_penalty = ((grad_norm** 2. - 1) ** 2).mean()
    # does not match eq.7 in paper, in which (grad_norm**2-1)**2

    # One sided penalty - down
    #     gradient_penalty = F.relu(grad_norm - 1).mean()

    return gradient_penalty


def step(model, optimizer, batch, l_gradient_penalty):
    model.train()
    optimizer.zero_grad()

    x, y = batch
    x.requires_grad_(True)

    z, y_pred = model(x)

    loss1 = F.binary_cross_entropy(y_pred, y)
    loss2 = l_gradient_penalty * calc_gradient_penalty(x, y_pred)
    # dp1, dp2 = cal_grad(x, y_pred)

    loss = loss1 + loss2

    loss.backward()
    optimizer.step()

    with torch.no_grad():
        model.update_embeddings(x, y)

    return loss.item(),x.detach().numpy(), y.detach().numpy(), y_pred.detach().numpy()


def eval_step(model, x, y, l_gradient_penalty):
    model.eval()

    # x, y = batch
    x = torch.tensor(x, dtype=torch.float)
    x.requires_grad_(True)

    z, y_pred = model(x)
    y0 = np.copy(y)
    y = F.one_hot(torch.from_numpy(y)).float()
    loss1 = F.binary_cross_entropy(y_pred, y)
    loss2 = l_gradient_penalty * calc_gradient_penalty(x, y_pred)
    loss = loss1 + loss2
    y_pred = np.argmax(y_pred.detach().numpy(), axis=1)
    accuracy = np.mean(y0 == y_pred)
    return accuracy, loss1.detach().numpy(), loss2.detach().numpy()


def eval_combined(model, dl_combined):
    model.eval()
    for i, batch in enumerate(dl_combined):
        x = batch[0]
        with torch.no_grad():
            z, y_pred = model(x)
    kernel_distance, pred = y_pred.max(1)
    return kernel_distance.numpy()


if __name__ == "__main__":
    dir = 'data/'
    f = np.load(dir + "combined_dataset.npz")
    # f = np.load(dir + "combined_dataset_trainUs80_testUs101.npz")
    # f = np.load(dir + "combined_dataset_trainUs101_testUs80.npz")
    outputDir = 'results/duq/'
    # outputDir='results/train_us80_test_us101/duq/'
    # outputDir = 'results/train_us101_test_us80/duq/'
    x_train = f['a']
    y_train = f['b']
    x_validate = f['c']
    y_validate = f['d']
    x_ood = f['e']
    print('{} train samples, positive rate {:.3f}'.format(x_train.shape[0], np.mean(y_train)))
    print('{} validate samples, positive rate {:.3f}'.format(x_validate.shape[0], np.mean(y_validate)))

    x_combined = np.concatenate((x_validate, x_ood))
    label_ood = np.zeros(x_combined.shape[0])
    label_ood[x_validate.shape[0]:] = 1

    ds_train = torch.utils.data.TensorDataset(torch.from_numpy(x_train).float(),
                                              F.one_hot(torch.from_numpy(y_train)).float())

    ds_test = torch.utils.data.TensorDataset(torch.from_numpy(x_validate).float(),
                                             F.one_hot(torch.from_numpy(y_validate)).float())

    ds_combined = torch.utils.data.TensorDataset(torch.from_numpy(x_combined).float())

    # network hyper param
    inputs = 4
    batchSize = 64
    epochs = 401
    hiddenUnits = 64
    learningRate = 0.0005
    l2Penalty = 1.0e-3
    num_classes = 2

    # duq param
    lambdas = np.linspace(0., 1., 11)
    # lambdas = np.array([.3])
    length_scales = np.linspace(0.1, 1., 10)
    # length_scales = np.linspace(0.3, 1., 8)
    # length_scales = np.array([0.4])
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    # runs = len(seeds)
    runs = 10

    for k in range(lambdas.shape[0]):
        for j in range(length_scales.shape[0]):
            AUCs = []
            ACCs = []
            print('lambda {:.3f} sigma {:.3f}'.format(lambdas[k], length_scales[j]))
            for run in range(runs):
                model = Model_bilinear(inputs, hiddenUnits, num_classes, length_scales[j])
                optimizer = torch.optim.Adam(model.parameters(), lr=learningRate,
                                             weight_decay=l2Penalty)
                dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batchSize, shuffle=True, drop_last=False)
                dl_test = torch.utils.data.DataLoader(ds_test, batch_size=x_validate.shape[0], shuffle=False)
                dl_combined = torch.utils.data.DataLoader(ds_combined, batch_size=x_combined.shape[0], shuffle=False)
                losses = []
                accuracies = []
                losses_validate = []
                accuracies_validate = []
                aucs = []
                for epoch in range(epochs):
                    for i, batch in enumerate(dl_train):
                        loss, x, y, y_pred = step(model, optimizer, batch, lambdas[k])
                    if epoch % 5 == 0:
                        accuracy, bce_loss, gp_loss = eval_step(model, x_train, y_train, lambdas[k])
                        losses.append(bce_loss + gp_loss)
                        accuracies.append(accuracy)
                        accuracy, bce_loss, gp_loss = eval_step(model, x_validate, y_validate, lambdas[k])
                        losses_validate.append(bce_loss + gp_loss)
                        accuracies_validate.append(accuracy)
                        uncertainties = eval_combined(model, dl_combined)
                        falsePositiveRate, truePositiveRate, _= roc_curve(label_ood, -uncertainties)
                        AUC = auc(falsePositiveRate.astype(np.float32), truePositiveRate.astype(np.float32))
                        aucs.append(AUC)
                        print('train: epoch', epoch, ', bce loss', bce_loss, 'gp loss', gp_loss, 'accuracy', accuracy,
                              'auc', AUC)

                # plot_save_loss(losses, losses_validate, outputDir + '/loss_run{}.png'.format(run))
                # plot_save_acc(accuracies, accuracies_validate, outputDir + '/acc_run{}.png'.format(run))
                AUCs.append(aucs)
                ACCs.append(accuracies_validate)
            AUCs = np.array(AUCs)
            ACCs = np.array(ACCs)
            dir = outputDir + 'mean_std_accs_aucs_lambda{:.3f}_sigma{:.3f}.npz'.format(lambdas[k], length_scales[j])
            np.savez(dir, a=np.mean(AUCs, axis=0), b=np.std(AUCs, axis=0),
                     c=np.mean(ACCs, axis=0), d=np.std(ACCs, axis=0),
                     e=np.arange(0, epochs, 5))


