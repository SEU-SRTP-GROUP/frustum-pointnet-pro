''' Util functions for training and evaluation.

Author: Charles R. Qi
Date: September 2017
'''

import numpy as np
import copy

def get(martrix1, martrix2, X, num):
    h = np.min(martrix2)
    row = np.where(martrix2 == np.min(martrix2))[0][0]
    col = np.where(martrix2 == np.min(martrix2))[1][0]

    row2 = martrix1[row][-2]
    col2 = martrix1[-1][col]

    martrix1[row][-1] -= 1
    martrix1[col][-1] -= 1

    if np.sum(martrix1 == -2) == 2 and len(martrix1) != 3 and X[col] == X[row] and X[col] != 0:
        while np.sum(martrix1 == -2) == 2 and X[col] == X[row] and X[col] != 0:
            martrix1[row][col] = 99
            martrix2[row][col] = 99
            martrix1[col][row] = 99
            martrix2[col][row] = 99
            martrix1[row][-1] += 1
            martrix1[col][-1] += 1

            h = np.min(martrix2)
            # print("awdwadwa",martrix1,X,num)
            # print(h)
            row = np.where(martrix2 == np.min(martrix2))[0][0]
            col = np.where(martrix2 == np.min(martrix2))[1][0]

            row2 = martrix1[row][-2]
            col2 = martrix1[-1][col]

            martrix1[row][-1] -= 1
            martrix1[col][-1] -= 1

            # print(len(np.where(martrix1==-1)[0]))

    if X[col] != 0 and X[row] != 0:
        if X[col] < X[row]:
            a = X[col]

            for i in range(len(X)):
                if X[i] == a:
                    X[i] = X[row]
        else:
            b = X[row]
            for i in range(len(X)):
                if X[i] == b:
                    X[i] = X[col]
    else:
        if X[col] != 0 or X[row] != 0:
            if X[col] == 0:
                X[col] = X[row]
            else:
                X[row] = X[col]
        else:
            X[row] = num
            X[col] = num

    martrix1[row][col] = 99
    martrix2[row][col] = 99
    martrix1[col][row] = 99
    martrix2[col][row] = 99

    # print(martrix1,h,X,num)
    # print(martrix1[row][-1],martrix1[col][-1])
    # print(row,col)
    if martrix1[col][-1] == -2:
        martrix1 = np.delete(martrix1, col, axis=0)
        martrix1 = np.delete(martrix1, col, axis=1)
        martrix2 = np.delete(martrix2, col, axis=0)
        martrix2 = np.delete(martrix2, col, axis=1)
        X = np.delete(X, col, axis=0)

    if martrix1[row][-1] == -2:
        martrix1 = np.delete(martrix1, row, axis=0)
        martrix1 = np.delete(martrix1, row, axis=1)
        martrix2 = np.delete(martrix2, row, axis=0)
        martrix2 = np.delete(martrix2, row, axis=1)
        X = np.delete(X, row, axis=0)
    # print(martrix1,h)
    return h, martrix1, martrix2, row2, col2, X, num - 1


def sort(martrix):
    # 排成一串
    num_order = []
    num = martrix[0][0]
    num_order.append(num)
    martrix[0][0] = -1

    num = martrix[0][1]
    num_order.append(num)
    martrix[0][1] = -1

    while np.max(martrix) != -1:
        # print(martrix)
        row2 = np.where(martrix == num)[0][0]
        col2 = np.where(martrix == num)[1][0]
        martrix[row2][col2] = -1

        if col2 == 0:
            num = martrix[row2][1]
            num_order.append(num)
            martrix[row2][1] = -1
        else:
            num = martrix[row2][0]
            num_order.append(num)
            martrix[row2][0] = -1
        # print(num)

    num_order.pop()
    return num_order


def resort(martrix, num_order):
    martrix2 = copy.deepcopy(martrix)
    num_order = list(map(int, num_order))
    for i in range(np.size(num_order)):
        martrix[i] = martrix2[num_order[i]]
    return martrix


def get_martrix(pointclouds):
    '''
    input:
        pointclouds: shape (N,4)
    output
        pointclouds_pl: shape (N,4)
    '''
    N = np.size(pointclouds, 0)
    H = [[0 for i in range(N)] for j in range(N)]
    for m in range(N):
        point1 = pointclouds[m]
        for n in range(N):
            point2 = pointclouds[n]
            H[m][n] = np.sqrt(
                (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2)
        for l in range(N):
            H[l][l] = 99
    num_order = get_string(H)
    pointclouds = resort(pointclouds, num_order)
    print(pointclouds)
    return pointclouds


def get_string(np1):
    N = np.size(np1, 0)
    for i in range(N):
        np1[i].append(i)
    collist = [i for i in range(N + 1)]
    np1.append(collist)
    np1[N][N] = 99
    # print(np1)

    np2 = np.delete(np1, -1, axis=1)
    np2 = np.delete(np2, -1, axis=0)

    np1 = np.insert(np1, len(np1), 0, axis=1)
    # list
    new_data = []
    new_data1 = []
    X = [0 for i in range(np.size(np1, 1))]
    num = -3
    for i in range(N):
        h, np1, np2, row, col, X, num = get(np1, np2, X, num)
        new_data.append([row, col])
        new_data1.append(h)
        # print(np1)
    # print(new_data)
    num_order = sort(new_data)
    return num_order

def get_batch(dataset, idxs, start_idx, end_idx,
              num_point, num_channel,
              from_rgb_detection=False):
    ''' Prepare batch data for training/evaluation.
    batch size is determined by start_idx-end_idx

    Input:
        dataset: an instance of FrustumDataset class
        idxs: a list of data element indices
        start_idx: int scalar, start position in idxs
        end_idx: int scalar, end position in idxs
        num_point: int scalar
        num_channel: int scalar
        from_rgb_detection: bool
    Output:
        batched data and label
    '''
    if from_rgb_detection:
        return get_batch_from_rgb_detection(dataset, idxs, start_idx, end_idx,
            num_point, num_channel)

    bsize = end_idx-start_idx
    batch_data = np.zeros((bsize, num_point, num_channel))
    batch_label = np.zeros((bsize, num_point), dtype=np.int32)
    batch_center = np.zeros((bsize, 3))
    batch_heading_class = np.zeros((bsize,), dtype=np.int32)
    batch_heading_residual = np.zeros((bsize,))
    batch_size_class = np.zeros((bsize,), dtype=np.int32)
    batch_size_residual = np.zeros((bsize, 3))
    batch_rot_angle = np.zeros((bsize,))
    if dataset.one_hot:
        batch_one_hot_vec = np.zeros((bsize,3)) # for car,ped,cyc
    for i in range(bsize):
        if dataset.one_hot:
            ps,seg,center,hclass,hres,sclass,sres,rotangle,onehotvec = \
                dataset[idxs[i+start_idx]]
            batch_one_hot_vec[i] = onehotvec
        else:
            ps,seg,center,hclass,hres,sclass,sres,rotangle = \
                dataset[idxs[i+start_idx]]
        batch_data[i,...] = get_martrix(ps[:,0:num_channel])
        batch_label[i,:] = seg
        batch_center[i,:] = center
        batch_heading_class[i] = hclass
        batch_heading_residual[i] = hres
        batch_size_class[i] = sclass
        batch_size_residual[i] = sres
        batch_rot_angle[i] = rotangle
    if dataset.one_hot:
        return batch_data, batch_label, batch_center, \
            batch_heading_class, batch_heading_residual, \
            batch_size_class, batch_size_residual, \
            batch_rot_angle, batch_one_hot_vec
    else:
        return batch_data, batch_label, batch_center, \
            batch_heading_class, batch_heading_residual, \
            batch_size_class, batch_size_residual, batch_rot_angle

def get_batch_from_rgb_detection(dataset, idxs, start_idx, end_idx,
                                 num_point, num_channel):
    bsize = end_idx-start_idx
    batch_data = np.zeros((bsize, num_point, num_channel))
    batch_rot_angle = np.zeros((bsize,))
    batch_prob = np.zeros((bsize,))
    if dataset.one_hot:
        batch_one_hot_vec = np.zeros((bsize,3)) # for car,ped,cyc
    for i in range(bsize):
        if dataset.one_hot:
            ps,rotangle,prob,onehotvec = dataset[idxs[i+start_idx]]
            batch_one_hot_vec[i] = onehotvec
        else:
            ps,rotangle,prob = dataset[idxs[i+start_idx]]
        batch_data[i,...] = get_martrix(ps[:,0:num_channel])
        batch_rot_angle[i] = rotangle
        batch_prob[i] = prob
    if dataset.one_hot:
        return batch_data, batch_rot_angle, batch_prob, batch_one_hot_vec
    else:
        return batch_data, batch_rot_angle, batch_prob


