import numpy as np
import csv
import matplotlib.pyplot as plt

detectionRange = 100
laneWidth = 3.7  # 3.7m to feet


def extract_sample():
    vehicleLength = 5.
    data = np.genfromtxt('lane_change_merge_after_4_obstacles.csv', delimiter=',')
    output = open('samples_merge_after.csv', 'w')
    writer = csv.writer(output)
    for i in range(0, int(data[-1, 0]) + 1):
        start = np.searchsorted(data[:, 0], i)
        if i == int(data[-1, 0]):
            end = data.shape[0] - 1
        else:
            end = np.searchsorted(data[:, 0], i + 1) - 1

        start0 = start
        end0 = end
        while (start0 < end0 and data[start0, 14] == 0.): start0 += 1
        if start0 == end0: continue
        if data[start0, 1] > -2.: continue
        # find lag at time shorter than 2 seconds before cross lane division
        while (start0 < end0 and data[end0, 14] == 0.): end0 -= 1
        if data[end0, 1] < 2.: continue
        # lag disappear at time shorter than 2 seconds after cross lane division

        # handle missing vehicle values
        if data[start0, 12] == 0:  # no corresponding obstacle for leading
            data[start0, 12] = data[start0, 4]
            data[start0, 11] = data[start0, 3] + detectionRange
            data[start0, 10] = 0.5 * laneWidth
            if data[start0, 14] < 0.:
                data[start0, 10] *= -1.
            print('no leading vehilce at ', i)
        if data[start0, 8] == 0:  # no corresponding obstacle for leading in old lane
            data[start0, 8] = data[start0, 4]
            data[start0, 7] = data[start0, 3] + detectionRange
            data[start0, 6] = 0.5 * laneWidth
            if data[start0, 2] < 0.:
                data[start0, 6] *= -1.
            print('no leading vehilce at original lane at ', i)

        du0 = data[start0, 4] - data[start0, 16]
        du1 = data[start0, 4] - data[start0, 12]
        du2 = data[start0, 4] - data[start0, 8]
        dx0 = data[start0, 3] - data[start0, 15]
        dx1 = data[start0, 3] - data[start0, 11]
        dx2 = data[start0, 3] - data[start0, 7]
        dy0 = data[start0, 2] - data[start0, 14]
        dy1 = data[start0, 2] - data[start0, 10]
        dy2 = data[start0, 2] - data[start0, 6]

        # if data[end0, 3]-data[end0,15] <vehicleLength: continue
        y = 0
        sample = [data[start0, 4], du0, du1, du2, dx0, dx1, dx2, dy0, dy1, dy2, y]

        if y == 1:
            plt.scatter(du0, dx0, color='blue', marker='o')
        else:
            plt.scatter(du0, dx0, color='red', marker='o')

        writer.writerow(np.around(sample, decimals=3))

    output.close()
    plt.ylabel('$\Delta x$')
    plt.xlabel('$\Delta v$')
    plt.axis([-10, 10, -30, 70])
    plt.show()


def extract_sample_snapshots():
    vehicleLength = 5.
    data = np.genfromtxt('lane_change_merge_after_4_obstacles.csv', delimiter=',')
    output = open('samples_merge_after_snapshots.csv', 'w')
    writer = csv.writer(output)
    count = 0
    for i in range(0, int(data[-1, 0]) + 1):
        start = np.searchsorted(data[:, 0], i)
        if i == int(data[-1, 0]):
            end = data.shape[0] - 1
        else:
            end = np.searchsorted(data[:, 0], i + 1) - 1

        start0 = start
        end0 = end
        while (start0 < end0 and data[start0, 14] == 0.): start0 += 1
        if start0 == end0: continue
        if data[start0, 1] > -2.: continue
        # find lag at time shorter than 2 seconds before cross lane division
        while (start0 < end0 and data[end0, 14] == 0.): end0 -= 1
        if data[end0, 1] < 2.: continue
        # lag disappear at time shorter than 2 seconds after cross lane division
        while data[end0, 1] > -5.:   end0 -= 1
        if end0 <= start0: continue

        print(count, 'after trim ', i, start0, end0, end0 - start0,
              data[start0, 1], data[end0, 1])

        for j in range(start0, end0 + 1):
            # handle missing vehicle values
            if data[j, 12] == 0:  # no corresponding obstacle for leading
                data[j, 12] = data[j, 4]
                data[j, 11] = data[j, 3] + detectionRange
                data[j, 10] = 0.5 * laneWidth
                if data[j, 14] < 0.:
                    data[j, 10] *= -1.
                print('no leading vehilce at ', i)
            if data[j, 8] == 0:  # no corresponding obstacle for leading in old lane
                data[j, 8] = data[j, 4]
                data[j, 7] = data[j, 3] + detectionRange
                data[j, 6] = 0.5 * laneWidth
                if data[j, 2] < 0.:
                    data[j, 6] *= -1.
                print('no leading vehilce at original lane at ', i)

            dt = data[j, 1]
            du0 = data[j, 4] - data[j, 16]
            du1 = data[j, 4] - data[j, 12]
            du2 = data[j, 4] - data[j, 8]
            dx0 = data[j, 3] - data[j, 15]
            dx1 = data[j, 3] - data[j, 11]
            dx2 = data[j, 3] - data[j, 7]
            dy0 = data[j, 2] - data[j, 14]
            dy1 = data[j, 2] - data[j, 10]
            dy2 = data[j, 2] - data[j, 6]

            # if data[end0, 3]-data[end0,15] <vehicleLength: continue
            y = 0
            sample = [count, dt, data[j, 4], du0, du1, du2, dx0, dx1, dx2, dy0, dy1, dy2, y]

            if y == 1:
                plt.scatter(du0, dx0, color='blue', marker='o')
            else:
                plt.scatter(du0, dx0, color='red', marker='o')

            writer.writerow(np.around(sample, decimals=3))
        count += 1

    output.close()
    plt.ylabel('$\Delta x$')
    plt.xlabel('$\Delta v$')
    plt.axis([-10, 10, -30, 70])
    plt.show()


def extract_sample_multiple_snapshot():
    vehicleLength = 5.
    data = np.genfromtxt('lane_changes.csv', delimiter=',')
    output = open('samples_multiple_snapshot.csv', 'w')
    writer = csv.writer(output)
    laneChanges = 0
    for i in range(0, int(data[-1, 0]) + 1):
        start = np.searchsorted(data[:, 0], i)
        if i == int(data[-1, 0]):
            end = data.shape[0] - 1
        else:
            end = np.searchsorted(data[:, 0], i + 1) - 1

        start0 = start
        end0 = end
        while (start0 < end0 and data[start0, -5] == 0.): start0 += 1
        if start0 == end0: continue
        if data[start0, 1] > -2.: continue
        # find lag at time shorter than 2 seconds before cross lane division
        while (start0 < end0 and data[end0, -5] == 0.): end0 -= 1
        if data[end0, 1] < 2.: continue
        # lag disappear at time shorter than 2 seconds after cross lane division
        print('before trim ', i, start0, end0, end0 - start0)
        while data[start0, 1] < -3.5: start0 += 1
        # while data[end0,1]  > 3.5:   end0-=1
        if data[end0, 3] - data[end0, 15] < vehicleLength: continue
        print('after trim ', i, start0, end0, end0 - start0)

        while data[start0, 1] <= -3.:
            du0 = data[start0, 4] - data[start0, 16]
            du1 = data[start0, 4] - data[start0, 12]
            du2 = data[start0, 4] - data[start0, 8]
            dx0 = data[start0, 3] - data[start0, 15]
            dx1 = data[start0, 3] - data[start0, 11]
            dx2 = data[start0, 3] - data[start0, 7]
            dy0 = data[start0, 2] - data[start0, 14]
            dy1 = data[start0, 2] - data[start0, 10]
            dy2 = data[start0, 2] - data[start0, 6]

            y = data[start0, -1]
            sample = [laneChanges, data[start0, 4], du0, du1, du2, dx0, dx1, dx2, dy0, dy1, dy2, y]

            if y == 1:
                plt.scatter(du0, dx0, color='blue', marker='o')
            else:
                plt.scatter(du0, dx0, color='red', marker='o')

            writer.writerow(np.around(sample, decimals=3))
            start0 += 1
        laneChanges += 1

    output.close()
    plt.ylabel('$\Delta x$')
    plt.xlabel('$\Delta v$')
    plt.axis([-15, 15, -50, 100])
    plt.show()


def extract_sample_modified_loss_function():
    '''
    extract snapshot from the lane changes for downstream machine learning
    input: lane changes, each lane change is a time sequence
    output: the extact snapshot for prediction
            [index, u0, du0, du1, du2, dx0, dx1, dx2, dy0, dy1, dy2, y]
    '''
    vehicleLength = 5.

    # for i-80
    # dir0 = '/home/hh/ngsim/I-80-Emeryville-CA/i-80-vehicle-trajectory-data/vehicle-trajectory-data/'
    # dir = dir0 + '0400pm-0415pm/'
    # dir = dir0 + '0500pm-0515pm/'
    # dir = dir0 + '0515pm-0530pm/'

    # for i-101
    dir0 = '/home/hh/ngsim/US-101-LosAngeles-CA/us-101-vehicle-trajectory-data/vehicle-trajectory-data/'
    # dir = dir0 + '0750am-0805am/'
    # dir = dir0 + '0805am-0820am/'
    dir = dir0 + '0820am-0835am/'

    data = np.genfromtxt(dir + 'lane_changes_merge_after.csv', delimiter=',')
    output = open(dir + 'samples_merge_after_snapshots.csv', 'w')
    writer = csv.writer(output)
    count = 0
    observeLength = 5
    allowedVelocityDiff = 4.
    for i in range(0, int(data[-1, 0]) + 1):
        start = np.searchsorted(data[:, 0], i)
        if i == int(data[-1, 0]):
            end = data.shape[0] - 1
        else:
            end = np.searchsorted(data[:, 0], i + 1) - 1

        start0 = start
        end0 = end
        while (start0 < end0 and data[start0, 14] == 0.): start0 += 1
        # preceding vehicles
        if start0 == end0: continue
        if data[start0, 1] > -2.: continue
        # find lag at time shorter than 2 seconds before cross lane division
        while (start0 < end0 and data[end0, 14] == 0.): end0 -= 1
        if end0 < start0 + observeLength: continue

        # sanity check to exclude 1 abnormal trajectory
        abnormal = False
        for j in range(start0 + 1, start0 + observeLength + 1):
            if abs(data[j, 16] - data[j - 1, 16]) > allowedVelocityDiff:
                # velocity of preceding vehicles
                print('abnormal huge acc/dec at trajectory ', i)
                abnormal = True
                break
        if abnormal:
            continue

        # print(count, 'after trim ', i, start0, end0, end0 - start0, data[start0, 1], data[end0, 1])

        # handle missing vehicle values
        # in extract lane changes, already guaranteed ego existed
        for j in range(start0, start0 + observeLength):
            if data[j, 12] == 0:  # no corresponding obstacle for leading
                data[j, 12] = data[j, 4]
                data[j, 11] = data[j, 3] + detectionRange
                data[j, 10] = 0.5 * laneWidth
                if data[j, 14] < 0.:
                    data[j, 10] *= -1.
                print('no leading vehilce at ', i)
            if data[j, 8] == 0:  # no corresponding obstacle for leading in old lane
                data[j, 8] = data[j, 4]
                data[j, 7] = data[j, 3] + detectionRange
                data[j, 6] = 0.5 * laneWidth
                if data[j, 2] < 0.:
                    data[j, 6] *= -1.
                print('no leading vehilce at original lane at ', i)
        j=start0 + observeLength- 1
        dx0 = data[j, 3] - data[j, 15]
        dx1 = data[j, 3] - data[j, 11]
        dx2 = data[j, 3] - data[j, 7]
        dy0 = data[j, 2] - data[j, 14]
        dy1 = data[j, 2] - data[j, 10]
        dy2 = data[j, 2] - data[j, 6]
        du0_0 = data[j, 4] - data[j, 16]
        du1_0 = data[j, 4] - data[j, 12]
        du2_0 = data[j, 4] - data[j, 8]
        du0 = np.mean(data[start0:start0+observeLength, 4] - data[start0:start0+observeLength, 16])
        du1 = np.mean(data[start0:start0+observeLength, 4] - data[start0:start0+observeLength, 12])
        du2 = np.mean(data[start0:start0+observeLength, 4] - data[start0:start0+observeLength, 8])
        # print(count, (du0_0-du0)/du0, (du1_0-du1)/du1, (du2_0-du2)/du2)
        # print(count, '(', du0_0,',', du0, '), (', du1_0, ',', du1, '), (', du2_0, ',', du2, ')')

        # if data[end0, 3]-data[end0,15] <vehicleLength: continue
        y = 0
        sample = [count, data[j, 4], du0, du1, du2, dx0, dx1, dx2, dy0, dy1, dy2, y]

        if y == 1:
            plt.scatter(du0, dx0, color='blue', marker='o')
        else:
            plt.scatter(du0, dx0, color='red', marker='o')

        writer.writerow(np.around(sample, decimals=3))
        # print('sample ', count, 'corresponding to trajectory ', i)
        count += 1

    output.close()
    plt.ylabel('$\Delta x$')
    plt.xlabel('$\Delta v$')
    plt.axis([-10, 10, -30, 70])
    plt.show()


# extract_sample()
# extract_sample_3seconds()
# extract_sample_multiple_snapshot()
# extract_sample_snapshots()
extract_sample_modified_loss_function()
