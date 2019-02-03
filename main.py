# coding:utf-8
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np 
from CHMM import CHMM
from utils import remove_padding, filtering, zscore, get_oneclass_data

def CharacterTrajectories():
    symbols = ['a','b','c','d','e','g','h','l','m','n','o','p','q','r','s','u','v','w','y','z']
    statecounts = [4, 3, 2, 4, 3, 4, 3, 2, 6, 4, 3, 3, 4, 3, 3, 4, 2, 4, 2, 3] 
    gaussiancounts = [2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2]

    CT = np.load('/home/scw4750/songbinxu/datasets/CharacterTrajectories/CharacterTrajectories.npz')
    x_train = CT['x_train'][:, :, :2]
    x_train = [remove_padding(seq) for seq in x_train]
    x_train = [filtering(seq, window=5) for seq in x_train]
    x_train = [zscore(seq) for seq in x_train]
    y_train = CT['y_train']

    '''train'''
    # for label in range(20):
    #     sub_data = get_oneclass_data(x_train, y_train, label)
    #     chmm = CHMM(sub_data, state_num=statecounts[label], gaussian_num=gaussiancounts[label], 
    #                 name='character_'+symbols[label], simplify=False)
    #     chmm.train(500)
    #     chmm.save_model('save/CharacterTrajectories/chmm_'+symbols[label]+'.npz')
    
    '''test'''
    # result = []
    # for label in range(20):
    #     print symbols[label]
    #     chmm = CHMM(x_train, state_num=statecounts[label], gaussian_num=gaussiancounts[label], 
    #                 name='character_'+symbols[label], mode='test', simplify=False)
    #     chmm.load_model('save/CharacterTrajectories/chmm_'+symbols[label]+'.npz')
    #     chmm.Viterbi_decode()
    #     result.append(np.max(chmm.delta, 1)[:, None])
    # pred = np.argmax(np.concatenate(result, 1), 1)
    # correct, total = np.sum(np.equal(y_train, pred).astype(int)), len(y_train)
    # print "train: correct=%d total=%d accuracy=%.4f" % (correct, total, correct/float(total))

    x_test = CT['x_test'][:, :, :2]
    x_test = [remove_padding(seq) for seq in x_test]
    x_test = [filtering(seq, window=5) for seq in x_test]
    x_test = [zscore(seq) for seq in x_test]
    y_test = CT['y_test']
    result = []
    for label in range(20):
        print symbols[label], 
        chmm = CHMM(x_test, state_num=statecounts[label], gaussian_num=gaussiancounts[label], 
                    name='character_'+symbols[label], mode='test', simplify=False)
        chmm.load_model('save/CharacterTrajectories/chmm_'+symbols[label]+'.npz')
        chmm.Viterbi_decode()
        result.append(np.max(chmm.delta, 1)[:, None])
    pred = np.argmax(np.concatenate(result, 1), 1)
    correct, total = np.sum(np.equal(y_test, pred).astype(int)), len(y_test)
    print "test: correct=%d total=%d accuracy=%.4f" % (correct, total, correct/float(total)) # 88.44%

    confuse_matrix = np.zeros((20, 20))
    for i in range(len(y_test)):
        confuse_matrix[y_test[i], pred[i]] += 1.0
    confuse_matrix = 1.0-(confuse_matrix - np.min(confuse_matrix))/(np.max(confuse_matrix)-np.min(confuse_matrix))
    
    ax = plt.subplot(111)
    plt.imshow(confuse_matrix, origin='lower', cmap='gray', interpolation='nearest')
    plt.xticks(range(20))
    plt.yticks(range(20))
    ax.set_xticklabels(symbols)
    ax.set_yticklabels(symbols)

    plt.savefig('save/CharacterTrajectories/confuse_matrix.png')
    plt.clf()

if __name__ == "__main__":
    CharacterTrajectories()