import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

primeNumber = 3
seqLen = 10

def transform_both_dataset():
    dir = '/home/hh/ngsim/I-80-Emeryville-CA/i-80-vehicle-trajectory-data/vehicle-trajectory-data/'
    f = np.load(dir + 'us80.npz')
    us80 = f['a']

    dir = '/home/hh/ngsim/US-101-LosAngeles-CA/us-101-vehicle-trajectory-data/vehicle-trajectory-data/'
    f = np.load(dir + 'us101.npz')
    us101 = f['a']

    mask = np.array([1, 2, 4, 5]) # already been chosen to delete
    us80_x = us80[:, 1:-1]
    us80_x = us80_x[:, mask]
    us80_y = us80[:, -1]
    us80_y = us80_y.astype(int)
    us101_x = us101[:, 1:-1]
    us101_x = us101_x[:, mask]
    us101_y = us101[:, -1]
    us101_y = us101_y.astype(int)

    dir = '/home/hh/data/ngsim/'
    # train us80, validate us101
    sc_X = StandardScaler()
    us80 = sc_X.fit_transform(us80_x)
    np.savez(dir + "us80_train.npz", a=us80, b=us80_y)
    us101 = sc_X.transform(us101_x)
    np.savez(dir + "us101_validate.npz", a=us101, b=us101_y)
    # train us101, validate us80
    sc_X = StandardScaler()
    us101 = sc_X.fit_transform(us101_x)
    np.savez(dir + "us101_train.npz", a=us101, b=us101_y)
    us80 = sc_X.transform(us80_x)
    np.savez(dir + "us80_validate.npz", a=us80, b=us80_y)


def load_data_both_dataset(trainUS80 = True):
    transformed = False
    if not transformed:
        transform_both_dataset()
    dir = '/home/hh/data/ngsim/'
    if trainUS80:
        f = np.load(dir + "us80_train.npz")
        us80_x = f['a']
        us80_y = f['b']
        f = np.load(dir + "us101_validate.npz")
        us101_x = f['a']
        us101_y = f['b']
        return us80_x, us80_y, us101_x, us101_y
    else:
        f = np.load(dir + "us101_train.npz")
        us101_x = f['a']
        us101_y = f['b']
        f = np.load(dir + "us80_validate.npz")
        us80_x = f['a']
        us80_y = f['b']
        return us101_x, us101_y, us80_x, us80_y

    # return (x_train, y_train, x_validate, y_validate, x_ood)


def preprocess_both_dataset():
    '''assemble all the lane changes and window abortions in both dataset, and shuffle them'''

    primeNumber = 3
    datas = []
    I80 = True
    # I80 = False

    # for i-80
    if I80:
        dir = '/home/hh/ngsim/I-80-Emeryville-CA/i-80-vehicle-trajectory-data/vehicle-trajectory-data/'
        datas.append(np.genfromtxt(dir+'0400pm-0415pm/samples_snapshots.csv', delimiter=','))
        datas.append(np.genfromtxt(dir+'0500pm-0515pm/samples_snapshots.csv', delimiter=','))
        datas.append(np.genfromtxt(dir+'0515pm-0530pm/samples_snapshots.csv', delimiter=','))
        datas.append(np.genfromtxt(dir+'0400pm-0415pm/samples_merge_after_snapshots.csv', delimiter=','))
        datas.append(np.genfromtxt(dir+'0500pm-0515pm/samples_merge_after_snapshots.csv', delimiter=','))
        datas.append(np.genfromtxt(dir+'0515pm-0530pm/samples_merge_after_snapshots.csv', delimiter=','))
    # for i-101
    else:
        dir = '/home/hh/ngsim/US-101-LosAngeles-CA/us-101-vehicle-trajectory-data/vehicle-trajectory-data/'
        datas.append(np.genfromtxt(dir + '0750am-0805am/samples_snapshots.csv', delimiter=','))
        datas.append(np.genfromtxt(dir + '0805am-0820am/samples_snapshots.csv', delimiter=','))
        datas.append(np.genfromtxt(dir + '0820am-0835am/samples_snapshots.csv', delimiter=','))
        datas.append(np.genfromtxt(dir + '0750am-0805am/samples_merge_after_snapshots.csv', delimiter=','))
        datas.append(np.genfromtxt(dir + '0805am-0820am/samples_merge_after_snapshots.csv', delimiter=','))
        datas.append(np.genfromtxt(dir + '0820am-0835am/samples_merge_after_snapshots.csv', delimiter=','))

    for i in range(3):
        # 0 for merge in front
        temp = np.zeros_like(datas[i])
        temp[:, 1:] = datas[i][:, 1:]
        datas[i] = temp
    for i in range(3, 6):
        # 1 for merge after
        temp = np.ones_like(datas[i])
        temp[:, 1:] = datas[i][:, 1:]
        datas[i] = temp

    data = np.vstack((datas[0], datas[1]))
    for i in range(2, 6):
        data = np.vstack((data, datas[i]))
    # data = data[:, 1:]  # delete index of lane changes
    ys = data[:, -1]
    print(ys.shape[0], ' samples, and ', np.mean(ys) * 100, '% positives')

    np.random.seed(primeNumber)
    np.random.shuffle(data)

    nCoop = data[:, -1].sum()
    nAfter = data[:, 0].sum()
    nTotal = data.shape[0]
    print('coop samples', nCoop)
    print('merge after samples', nAfter)
    print('adv samples of merge in front', nTotal - nAfter - nCoop)
    if I80:
        dir = '/home/hh/ngsim/I-80-Emeryville-CA/i-80-vehicle-trajectory-data/vehicle-trajectory-data/'
        np.savez(dir + "us80.npz", a=data)
        f = np.load(dir+'us80.npz')
    else:
        dir = '/home/hh/ngsim/US-101-LosAngeles-CA/us-101-vehicle-trajectory-data/vehicle-trajectory-data/'
        np.savez(dir + "us101.npz", a=data)
        f = np.load(dir+'us101.npz')


def inspect_abnormal():
    '''check whether the abnormal samples make sense'''
    vehicleLength = 5.
    detectionRange = 100
    laneWidth = 3.7

    # for i-80
    dir0 = '/home/hh/ngsim/I-80-Emeryville-CA/i-80-vehicle-trajectory-data/vehicle-trajectory-data/'
    # dir = dir0 + '0400pm-0415pm/'
    # dir = dir0 + '0500pm-0515pm/'
    dir = dir0 + '0515pm-0530pm/'


    data = np.genfromtxt(dir+'lane_changes.csv', delimiter=',')
    # output = open(dir+'samples_snapshots.csv', 'w')

    # writer = csv.writer(output)
    count = 0
    minTimeBeforeLC = 1.5  # from decision to cross lane divider
    minTimeAfterLC = 1.5  # from cross lane divider to end of lane change
    observationLength = 5
    cooperates = 0

    fig, ax = plt.subplots()
    ax.set(xlim=(-15, 15), ylim=(-50., 100.))
    for i in range(0, int(data[-1, 0]) + 1):
        # for i in range(0, 5):
        start = np.searchsorted(data[:, 0], i)
        if i == int(data[-1, 0]):
            end = data.shape[0] - 1
        else:
            end = np.searchsorted(data[:, 0], i + 1) - 1

        start0 = start
        end0 = end
        while (start0 < end0 and data[start0, -5] == 0.):
            # lag vehicle not show up as x_lag=0 (lateral position)
            start0 += 1
        if start0 == end0: continue
        if data[start0, 1] > -minTimeBeforeLC: continue
        # find lag at time shorter than 2 seconds before cross lane division
        while (start0 < end0 and data[end0, -5] == 0.): end0 -= 1
        if data[end0, 1] < minTimeAfterLC: continue
        if data[end0, 3] - data[end0, 15] < vehicleLength: continue
        # print(count, 'after trim ', i, start0, end0, end0 - start0, data[start0, 1], data[end0, 1])

        # handle missing vehicle values
        for j in range(start0, start0 + observationLength):
            if data[j, 11] == 0:  # no corresponding preceding obstacle for target lane
                data[j, 12] = data[j, 4]
                data[j, 11] = data[j, 3] + detectionRange
                data[j, 10] = 0.5 * laneWidth
                if data[j, 14] < 0.:
                    data[j, 10] *= -1.
                # print('no leading vehilce at ', j)
            if data[j, 7] == 0:  # no corresponding obstacle for leading in old lane
                data[j, 8] = data[j, 4]
                data[j, 7] = data[j, 3] + detectionRange
                data[j, 6] = 0.5 * laneWidth
                if data[j, 2] < 0.:
                    data[j, 6] *= -1.
                # print('no leading vehilce at original lane at ', j)

        j = start0 + observationLength -1
        dx0 = data[j, 3] - data[j, 15]
        dx1 = data[j, 3] - data[j, 11]
        dx2 = data[j, 3] - data[j, 7]
        dy0 = data[j, 2] - data[j, 14]
        dy1 = data[j, 2] - data[j, 10]
        dy2 = data[j, 2] - data[j, 6]
        u0 = np.mean(data[start0:start0 + observationLength, 4])
        du0 = np.mean(data[start0:start0 + observationLength, 4] - data[start0:start0 + observationLength, 16])
        du1 = np.mean(data[start0:start0 + observationLength, 4] - data[start0:start0 + observationLength, 12])
        du2 = np.mean(data[start0:start0 + observationLength, 4] - data[start0:start0 + observationLength, 8])
        y = data[start0, -1]
        cooperates += y

        sample = [count, u0, du0, du1, du2, dx0, dx1, dx2, dy0, dy1, dy2, y]
        ###
        if y == 0 and du0 > 3. and  dx0 >= 14.:
            print('abnormal adv at i {}, dv {:.2f}, dx {:.2f}'.format(i, du0, dx0 ))
            plt.scatter(du0, dx0, color='red', marker='o')
        if y == 1 and du0 < -2. and  dx0 <= 10.:
            print('abnormal coop at i {}, dv {:.2f}, dx {:.2f}'.format(i, du0, dx0 ))
            plt.scatter(du0, dx0, color='blue', marker='o')

        # writer.writerow(np.around(sample, decimals=3))
        count += 1


def visualize_ood_sample(x, x_validate, xGenerated):
    xIndistribution = np.concatenate((x, x_validate))
    fig, ax = plt.subplots()
    ax.scatter(xIndistribution[:,0], xIndistribution[:,2], c='blue', marker='o', label='In-distribution')
    ax.scatter(xGenerated[::10,0], xGenerated[::10,2], c='orange', marker='x', label='OOD')
    ax.set(xlim=(-30.2, 21.3), ylim=(-50., 100.))
    plt.legend()
    plt.show()


def get_threshold(x, percentage, dir):
    from collections import Counter
    from numpy import linalg as LA

    print('In total, ', x.shape[0], ' samples')
    minDis = []
    for i in range(x.shape[0]):
        mask = np.ones(x.shape[0], dtype=np.bool)
        mask[i] = False
        diff = x - x[i]
        diff = LA.norm(diff, ord=2, axis=1)
        diff = min(diff[mask])
        minDis.append(diff)
    minDis0 = minDis.copy()

    minDis = Counter(minDis)
    minDis = sorted(minDis.items())
    dis = []
    frequency = []
    count = 0.
    for p in minDis:
        dis.append(p[0])
        count += p[1]
        frequency.append(count)
    frequency = np.array(frequency)
    dis = np.array(dis)
    frequency /= count
    np.savez(dir, a=dis, b=frequency)
    # plt.plot(dis, frequency, '-o')
    # plt.axis([0, 3.0, 0., 1.1])
    # plt.ylabel('percentage')
    # plt.xlabel('minimum distance to in-distribution samples')
    # plt.show()
    for i in range(dis.shape[0]-1, 0, -1):
        if frequency[i]>percentage and frequency[i-1]<=percentage:
            minDis = dis[i-1]
            break
    print('min dis ', minDis, 'for percentage ', percentage)
    oodLabel = minDis0>=minDis
    return minDis, oodLabel


def extract_ood(xInDistribution, xGenerated, minDis):
    # minDis = 1.4967  # for percentage 0.99
    from numpy import linalg as LA
    # minDis = get_threshold(xInDistribution, percentage)
    # minDis = 1.2846  # for percentage 0.98
    mask = np.zeros(xGenerated.shape[0], dtype=np.bool)
    for i in range(xGenerated.shape[0]):
        diff = xInDistribution - xGenerated[i]
        diff = LA.norm(diff, ord=2, axis=1)
        diff = min(diff)
        if diff >= minDis:
            mask[i] = True
        if i % 10000 == 0: print('done for ', i+1, ', distance ', diff )
    xOOD = xGenerated[mask]
    print('from ', xGenerated.shape[0], 'samples, extracted ', xOOD.shape[0], 'OOD samples with threshold ', minDis)
    # np.savez("/home/hh/data/ood_sample.npz", a=xOOD)
    # np.savez("/home/hh/data/ood_sample_cleaned.npz", a=xOOD)
    return mask, xOOD


def generate_ood(x):
    '''input x, output generated OOD;
       should not be normalized '''
    print('dv0 range ', min(x[:,0]), max(x[:,0]))
    print('dv1 range ', min(x[:,1]), max(x[:,0]))
    print('dx0 range ', min(x[:,2]), max(x[:,2]))
    print('dx1 range ', min(x[:,3]), max(x[:,3]))
    dv0_min = 1.5 * min(x[:,0])
    dv0_max = 1.5 * max(x[:,0])
    dv1_min = 1.5 * min(x[:,1])
    dv1_max = 1.5 * max(x[:,1])
    detectionRange = 100.
    pointsInEachDim = 20
    uniformGrid = False
    if uniformGrid:
        dv0 = np.linspace(dv0_min, dv0_max + 0.5, pointsInEachDim)
        dv1 = np.linspace(dv1_min, dv1_max + 0.5, pointsInEachDim)
        dx0 = np.linspace(-detectionRange/2., detectionRange + 0.5, pointsInEachDim)
        dx1 = np.linspace(-detectionRange, detectionRange/2., pointsInEachDim)
        v0, v1, x0, x1 = np.meshgrid(dv0, dv1, dx0, dx1)
        xGenerated = np.column_stack([v0.flatten(), v1.flatten(), x0.flatten(), x1.flatten()])
    else:
        l = [dv0_min, dv1_min, -detectionRange/2., -detectionRange]
        h = [dv0_max, dv1_max, detectionRange, detectionRange/2.]
        xGenerated = np.random.uniform(low=l, high=h, size = (pointsInEachDim**4, 4))
    return xGenerated


def prepare_validate_and_generate_ood():
    '''
    0 for lag vehicle in target lane
    1 for front vehicle in target lane
    2 for front vehicle in original lane
    x = [v_ego, dv0, dv1, dv2, dx0, dx1, dx2, dy0, dy1, dy2]
    us80.npz, us101.npz stores raw [merge after/before(1/0), v_ego, dv0, dv1, dv2, dx0, dx1, dx2, dy0, dy1, dy2, label(0/1)]
    '''

    percentage = 0.99

    featureMask = np.array([1, 2, 4, 5]) # already been chosen to delete
    np.random.seed(0)

    dir = '../'
    f = np.load(dir + "us80.npz")
    us80 = f['a']
    f = np.load(dir + "us101.npz")
    us101 = f['a']
    print('us80: ', us80.shape[0], ' samples,', np.sum(us80[:,-1]), 'coop', np.sum(us80[:,0]), 'merge after')
    print('us101: ', us101.shape[0], ' samples,', np.sum(us101[:,-1]), 'coop', np.sum(us101[:,0]), 'merge after')
    combined = np.concatenate((us80, us101))
    np.random.seed(primeNumber)
    np.random.shuffle(combined)
    x_combined = combined[:, 1:-1]
    y_combined = combined[:, -1]
    x_combined = x_combined[:, featureMask]
    x_combined0 = x_combined.copy()
    sc_X = StandardScaler()
    x_combined = sc_X.fit_transform(x_combined)
    # need to scale first before any calculating of distance

    # clean data, 99% dis to in-dis, 1% to OOD
    minDis, oodLabel = get_threshold(x_combined, percentage, dir)
    x_inDis = x_combined[~oodLabel]
    y_inDis = y_combined[~oodLabel]
    x_ood = x_combined[oodLabel]
    print('total sample ', x_combined.shape[0], ', in dis ', x_inDis.shape[0], ', ood ', x_ood.shape[0])

    # generate ood samples
    combined_ood = generate_ood(x_combined0[~oodLabel])
    combined_ood0 = combined_ood.copy()
    combined_ood = sc_X.transform(combined_ood)
    mask, xGenerated = extract_ood(x_inDis, combined_ood, minDis)

    # split to train and test
    sz = x_inDis.shape[0]
    trainRatio = 0.75
    x_train = x_inDis[:int(sz * trainRatio)]
    y_train = y_inDis[:int(sz * trainRatio)].astype(int)
    x_test = x_inDis[int(sz * trainRatio):]
    y_test = y_inDis[int(sz * trainRatio):].astype(int)
    print(y_train.shape[0], ' trainning samples, and ',
          np.mean(y_train) * 100, '% positives')
    print(y_test.shape[0], ' validate samples, and ',
          np.mean(y_test) * 100, '% positives')

    x_train0 = x_combined0[~oodLabel][:int(sz * trainRatio)]
    print('for training:')
    print('dv0 range ', min(x_train0[:,0]), max(x_train0[:,0]))
    print('dv1 range ', min(x_train0[:,1]), max(x_train0[:,0]))
    print('dx0 range ', min(x_train0[:,2]), max(x_train0[:,2]))
    print('dx1 range ', min(x_train0[:,3]), max(x_train0[:,3]))
    print('for ood:')
    print('dv0 range ', min(combined_ood0[mask][:,0]), max(combined_ood0[mask][:,0]))
    print('dv1 range ', min(combined_ood0[mask][:,1]), max(combined_ood0[mask][:,0]))
    print('dx0 range ', min(combined_ood0[mask][:,2]), max(combined_ood0[mask][:,2]))
    print('dx1 range ', min(combined_ood0[mask][:,3]), max(combined_ood0[mask][:,3]))
    # x_ood = sc_X.transform(xGenerated)
    # np.savez(dir + "combined_dataset.npz", a=x_train, b=y_train, c=x_test, d=y_test, e=x_ood)
    np.savez(dir + "combined_dataset.npz", a=x_train, b=y_train, c=x_test, d=y_test, e=xGenerated)


def prepare_validate_and_feature_selection():
    np.random.seed(primeNumber)

    dir = '../'
    f = np.load(dir + "us80.npz")
    us80 = f['a']
    f = np.load(dir + "us101.npz")
    us101 = f['a']
    print('us80: ', us80.shape[0], ' samples,', np.sum(us80[:,-1]), 'coop', np.sum(us80[:,0]), 'merge after')
    print('us101: ', us101.shape[0], ' samples,', np.sum(us101[:,-1]), 'coop', np.sum(us101[:,0]), 'merge after')
    combined = np.concatenate((us80, us101))
    np.random.shuffle(combined)
    x_combined = combined[:, 1:-1]
    y_combined = combined[:, -1]
    sc_X = StandardScaler()
    x_combined = sc_X.fit_transform(x_combined)

    # split to train and test
    sz = x_combined.shape[0]
    trainRatio = 0.75
    x_train = x_combined[:int(sz * trainRatio)]
    y_train = y_combined[:int(sz * trainRatio)].astype(int)
    x_test = x_combined[int(sz * trainRatio):]
    y_test = y_combined[int(sz * trainRatio):].astype(int)
    print(y_train.shape[0], ' trainning samples, and ',
          np.mean(y_train) * 100, '% positives')
    print(y_test.shape[0], ' validate samples, and ',
          np.mean(y_test) * 100, '% positives')
    np.savez(dir + "combined_dataset_before_feature_selection.npz", a=x_train, b=y_train, c=x_test, d=y_test)


def prepare_validate_and_generate_ood_trainUs80_testUs101():
    percentage = 0.99
    featureMask = np.array([1, 2, 4, 5]) # already been chosen to delete
    np.random.seed(0)

    dir = '/home/hh/data/ngsim/'
    f = np.load(dir + "us80.npz")
    us80 = f['a']
    f = np.load(dir + "us101.npz")
    us101 = f['a']
    print('us80: ', us80.shape[0], ' samples,', np.sum(us80[:,-1]), 'coop', np.sum(us80[:,0]), 'merge after')
    print('us101: ', us101.shape[0], ' samples,', np.sum(us101[:,-1]), 'coop', np.sum(us101[:,0]), 'merge after')
    combined = np.concatenate((us80, us101))
    dataUs80 = np.ones(combined.shape[0], dtype = np.bool)
    dataUs80[us80.shape[0]:] = False
    np.random.seed(primeNumber)
    # np.random.shuffle(combined)
    x_combined = combined[:, 1:-1]
    y_combined = combined[:, -1]
    x_combined = x_combined[:, featureMask]
    x_combined0 = x_combined.copy()
    sc_X = StandardScaler()
    x_combined = sc_X.fit_transform(x_combined)
    # need to scale first before any calculating of distance

    # clean data, 99% dis to in-dis, 1% to OOD
    minDis, oodLabel = get_threshold(x_combined, percentage, dir)
    x_inDis = x_combined[~oodLabel]
    y_inDis = y_combined[~oodLabel]
    x_ood = x_combined[oodLabel]
    print('total sample ', x_combined.shape[0], ', in dis ', x_inDis.shape[0], ', ood ', x_ood.shape[0])

    # generate ood samples
    combined_ood = generate_ood(x_combined0[~oodLabel])
    combined_ood0 = combined_ood.copy()
    combined_ood = sc_X.transform(combined_ood)
    mask, xGenerated = extract_ood(x_inDis, combined_ood, minDis)

    # split to train and test
    # for train us80, test us101
    print('train us80, test us101')
    x_train = x_combined[np.logical_and(~oodLabel, dataUs80)]
    y_train = y_combined[np.logical_and(~oodLabel, dataUs80)].astype(int)
    x_test = x_combined[np.logical_and(~oodLabel, ~dataUs80)]
    y_test = y_combined[np.logical_and(~oodLabel, ~dataUs80)].astype(int)
    print(y_train.shape[0], ' trainning samples, and ',
          np.mean(y_train) * 100, '% positives')
    print(y_test.shape[0], ' validate samples, and ',
          np.mean(y_test) * 100, '% positives')
    np.savez(dir + "combined_dataset_trainUs80_testUs101.npz", a=x_train, b=y_train, c=x_test, d=y_test, e=xGenerated)

    # for train us101, test us80
    print('train us101, test us80')
    x_train = x_combined[np.logical_and(~oodLabel, ~dataUs80)]
    y_train = y_combined[np.logical_and(~oodLabel, ~dataUs80)].astype(int)
    x_test = x_combined[np.logical_and(~oodLabel, dataUs80)]
    y_test = y_combined[np.logical_and(~oodLabel, dataUs80)].astype(int)
    print(y_train.shape[0], ' trainning samples, and ',
          np.mean(y_train) * 100, '% positives')
    print(y_test.shape[0], ' validate samples, and ',
          np.mean(y_test) * 100, '% positives')
    np.savez(dir + "combined_dataset_trainUs101_testUs80.npz", a=x_train, b=y_train, c=x_test, d=y_test, e=xGenerated)


def generate_two_gaussian():
    from plot_utils import plot_two_gaussian
    np.random.seed(0)
    variance = 2.
    n_train = 1500
    n_test = 1000
    mean0 = [-2., 0.]
    mean1 = [2., 0.]
    cov = [[variance, 0], [0, variance]]
    x0 = np.random.multivariate_normal(mean0, cov, n_train)
    x0 = np.hstack((x0, np.zeros((x0.shape[0],1))))
    x1 = np.random.multivariate_normal(mean1, cov, n_train)
    x1 = np.hstack((x1, np.ones((x1.shape[0],1))))
    x_train = np.concatenate((x0, x1))
    np.random.shuffle(x_train)
    plot_two_gaussian(x_train)

    x0 = np.random.multivariate_normal(mean0, cov, n_test)
    x0 = np.hstack((x0, np.zeros((x0.shape[0],1))))
    x1 = np.random.multivariate_normal(mean1, cov, n_test)
    x1 = np.hstack((x1, np.ones((x1.shape[0],1))))
    x_test = np.concatenate((x0, x1))
    np.random.shuffle(x_test)
    dir = '/home/hh/data/two_gaussian/'
    np.savez(dir+'two_gaussian_train_test.npz', a=x_train, b=x_test)


def extract_car_total_trajectories():
    def count(data):
        ans0 = 1
        ans = 1
        for i in range(1, data.shape[0]):
            if(data[i,0] == data[i-1,0]): continue
            ans0 += 1
            if(data[i, -8] != 2): continue
            ans += 1
            if(i%100000 == 0): print(i)
        print('total ', ans0, ', car ', ans)
        return ans0, ans

    ans = 0
    ans0 = 0
    # for I80
    dir0 = '/home/hh/ngsim/I-80-Emeryville-CA/i-80-vehicle-trajectory-data/vehicle-trajectory-data/'
    dir = dir0 + '0400pm-0415pm/'
    data = np.loadtxt(dir + 'trajectories-0400-0415.txt')
    n0, n = count(data)
    ans0 += n0
    ans += n
    dir = dir0 + '0500pm-0515pm/'
    data = np.loadtxt(dir + 'trajectories-0500-0515.txt')
    n0, n = count(data)
    ans0 += n0
    ans += n
    dir = dir0 + '0515pm-0530pm/'
    data = np.loadtxt(dir + 'trajectories-0515-0530.txt')
    n0, n = count(data)
    ans0 += n0
    ans += n
    print('car trajectories in us80', ans0, ans)

    # for I101
    ans0 = 0
    ans = 0
    dir0 = '/home/hh/ngsim/US-101-LosAngeles-CA/us-101-vehicle-trajectory-data/vehicle-trajectory-data/'
    dir = dir0 + '0750am-0805am/'
    data = np.loadtxt(dir + 'trajectories-0750am-0805am.txt')
    n0, n = count(data)
    ans0 += n0
    ans += n
    dir = dir0 + '0805am-0820am/'
    data = np.loadtxt(dir + 'trajectories-0805am-0820am.txt')
    n0, n = count(data)
    ans0 += n0
    ans += n
    dir = dir0 + '0820am-0835am/'
    data = np.loadtxt(dir + 'trajectories-0820am-0835am.txt')
    n0, n = count(data)
    ans0 += n0
    ans += n
    print('car trajectories in us101', ans0, ans)


if __name__ == "__main__":
    extract_car_total_trajectories()
    # preprocess_both_dataset()
    # inspect_abnormal()
    # generate_two_gaussian()
    # prepare_validate_and_generate_ood()
    # prepare_validate_and_generate_ood_trainUs80_testUs101()
    # prepare_validate_and_feature_selection()