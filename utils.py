# coding:utf-8

import numpy as np 
import sys

from sklearn.cluster import KMeans

def kmeans_cluster(X, k):
    '''
        X: [N, d]
        idxs: list k * [N//k, ], origin index of samples in each cluster
        centers: [k, d], centers of each cluster
    '''
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    idxs = [[] for _ in range(k)]
    for i in range(len(X)):
        idxs[kmeans.labels_[i]].append(i)
    idxs=np.array(idxs)
    centers = kmeans.cluster_centers_
    return {'indexs':idxs, 'centers':centers}

def cov(S):
    return np.diag(np.cov(S.T))

def uniform_disturbance(matrix, seed=None):
    if seed:
        np.random.seed(seed)
    # d = np.abs(np.min(matrix[matrix>0]))
    d = 1e-10
    random_noise = np.random.uniform(-d, d, matrix.shape)
    condlist = [matrix!=0, matrix==0]
    choicelist = [matrix, random_noise]
    res = np.select(condlist, choicelist)
    return res

def truncate(matrix, maxvalue):
    return np.where(matrix>maxvalue, maxvalue, matrix)

def save_divide(a, b, bias=1e-10):
    if b==0:
        return a/bias
    else:
        return a/b

def get_b(seq, miu, U, simplify=True):
    d = miu.shape[-1]
    # U = uniform_disturbance(U) # avoid Singular
    # if not simplify:
    #     for i in range(U.shape[0]):
    #         for j in range(U.shape[1]):
    #             if np.linalg.det(U[i,j])==0:
    #                 U[i,j] = np.diag(np.diag(U[i,j]))

    # [N, M, T, d]
    delta = np.expand_dims(np.expand_dims(seq,0),0) - np.expand_dims(miu, 2)

    if simplify:
        # [N, M, d]
        prods = np.matmul(np.expand_dims(delta / np.expand_dims(U, 2), -2), np.expand_dims(delta, -1))
        prods = prods[:, :, :, 0, 0]
        
        log_b = -0.5 * d * Log(2*np.pi) - 0.5 * Log(np.abs(np.expand_dims(np.prod(U, -1), -1))) - 0.5 * prods 
        
    else:
        # [N, M, d, d]
        prods = np.matmul(np.matmul(delta, np.linalg.inv(U)), np.transpose(delta, (0,1,3,2)))
        # [N, M, d]
        prods = np.diagonal(prods, 0, 2, 3)
        log_b = -0.5 * d * Log(2*np.pi) - 0.5 * np.expand_dims(Log(np.abs(np.linalg.det(U))), -1) - 0.5 * prods
    
    log_b = truncate(log_b, 200)
    b = np.exp(log_b)

    return b

def get_B(b, C):
    return np.sum(b * np.expand_dims(C, -1), axis=1)

def l1_norm(x, y):
    return np.mean(np.abs(x - y))

def check_nan(mat):
    tmp = np.reshape(mat, (np.prod(mat.shape), ))
    for i in range(len(tmp)):
        if np.isnan(tmp[i]):
            return True 
    return False

def zscore(seq):
	# seq, [T,d]
	mius = np.mean(seq, axis=0)
	stds = np.std(seq, axis=0)
	return (seq - mius) / stds 

def filtering(seq, window=3):
    # seq, [T,d]
	d = window // 2
	res = []
	for i,j in zip([0] * d + range(len(seq) - d), range(d, len(seq)) + [len(seq)] * d):
		res.append(np.mean(seq[i:j+1, :], axis=0))
	return np.asarray(res)

def get_class_type(fs):
    res = []
    for f in fs:
        tmp = f.split('/')[-1].split('_')
        if tmp[0] == 'num':
            res.append(ord(tmp[1]) - ord('0'))
        elif tmp[0] == 'upper':
            res.append(ord(tmp[1]) - ord('A') + 10)
        elif tmp[0] == 'lower':
            res.append(ord(tmp[1]) - ord('a') + 36)
        else:
            raise ValueError('Invalid field {}.'.format(tmp[0]))
    return res

def Log(x):
    # bias = -1.7976931348623157E308
    # bias = - sys.float_info.max
    bias = -1e20
    log_x = np.log(np.where(x==0, 1.0, x))
    bias_x = np.ones_like(x) * bias
    
    res = np.select([x!=0, x==0], [log_x, bias_x])
    return res

def remove_padding(seq):
    for i in range(len(seq)-2, -1, -1):
        if np.sum(np.abs(seq[i]-seq[i+1]))> 1:
            pos = i 
            break 
    return seq[:pos+1]

def get_oneclass_data(data, labels, label):
    idxs = []
    for i in range(len(labels)):
        if labels[i] == label:
            idxs.append(i)
    return [data[i] for i in idxs]
