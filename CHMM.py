# coding:utf-8

import numpy as np 
from collections import Counter
from utils import *

import matplotlib 
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class CHMM(object):
    def __init__(self, dataset, state_num, gaussian_num, name, mode='train', simplify=False):
        print "state={}, gaussian={}".format(state_num, gaussian_num)
        self.dataset = dataset # list of [T, d]
        self.state_num = state_num # N
        self.gaussian_num = gaussian_num # M
        self.data_dim = self.dataset[0].shape[1]

        self.simplify = simplify
        self.name = name
        self.mode = mode
        
        self.gaussian_assignment_kmeans()
        self.initialize_params()
        self.set_thresholds()
        
    def set_thresholds(self, thresholds=None):
        default_thresholds = {
            'A': 1e-7,
            'miu': 1e-5,
            'cov': 1e-5
        }
        self.thresholds = thresholds or default_thresholds
    
    def gaussian_assignment_kmeans(self):
        '''use K-means to initialize gaussians'''
        if self.mode == 'test':
            return
        self.segments = [[] for _ in range(self.state_num)] # N * []
        self.kmeans_result = [[] for _ in range(self.state_num)] # N * []
        for seq in self.dataset:
            seq = np.array_split(seq, self.state_num) # N * [T//N, d]
            for i in range(self.state_num):
                self.segments[i].append(seq[i])
        self.segments = [np.concatenate(vec) for vec in self.segments] # N * [batchsize*(T//N), d]
        
        '''
                kmeans_result: N * dict('indexs', 'centers')
                kmeans_result[i]['indexs'] : M * [samplenum_of_Statei//M, ]
                kmeans_result[i]['centers'] : [M, d]
        '''
        for i in range(self.state_num):
            self.kmeans_result[i] = \
                kmeans_cluster(self.segments[i], self.gaussian_num)
    
    def initialize_params(self):
        '''initialize CHMM params'''
        # Pi
        self.P = np.ones((self.state_num, )) * (1.0 / self.state_num) # N
        # A
        self.A = np.zeros((self.state_num, self.state_num)) # N, N
        for i in range(self.state_num - 1):
            self.A[i,i] = 0.5
            self.A[i,i+1] = 0.5
        self.A[-1, -1] = 1.0

        # C
        self.C = np.zeros((self.state_num, self.gaussian_num)) # N, M
        # miu
        self.miu = np.zeros((self.state_num, self.gaussian_num, self.data_dim)) # N, M, d
        # U
        if self.simplify:
            self.cov = np.zeros((self.state_num, self.gaussian_num, self.data_dim)) # N, M, d
        else:
            self.cov = np.zeros((self.state_num, self.gaussian_num, self.data_dim, self.data_dim)) # N, M, d, d
            
        if self.mode == 'train':
            for j in range(self.state_num):
                for m in range(self.gaussian_num):
                    self.C[j][m] = len(self.kmeans_result[j]['indexs'][m]) / float(len(self.segments[j]))
            for j in range(self.state_num):
                for m in range(self.gaussian_num):
                    self.miu[j,m] = np.mean(self.segments[j][self.kmeans_result[j]['indexs'][m]], 0)
            for j in range(self.state_num):
                for m in range(self.gaussian_num):
                    if self.simplify:
                        self.cov[j,m] = np.diag(np.cov(self.segments[j][self.kmeans_result[j]['indexs'][m]].T))
                    else:
                        self.cov[j,m] = np.cov(self.segments[j][self.kmeans_result[j]['indexs'][m]].T)

        self.b = [np.zeros((self.state_num, self.gaussian_num, seq.shape[0])) for seq in self.dataset]
        self.B = [np.zeros((self.state_num, seq.shape[0])) for seq in self.dataset]
        self.alpha = [np.zeros((self.state_num, seq.shape[0])) for seq in self.dataset]
        self.beta = [np.zeros((self.state_num, seq.shape[0])) for seq in self.dataset]
        self.Gamma = [np.zeros((self.state_num, seq.shape[0])) for seq in self.dataset] # gamma_jt
        self.gamma = [np.zeros((self.state_num, self.gaussian_num, seq.shape[0])) for seq in self.dataset] # gamma_ijt
        self.ksai = [np.zeros((self.state_num, self.state_num, seq.shape[0])) for seq in self.dataset]
        
        self.delta = np.zeros((len(self.dataset), self.state_num))

        keys_1 = ['alpha', 'beta', 'gamma', 'Gamma', 'ksai', 'b', 'B']
        keys_2 = ['P', 'A', 'C', 'miu', 'cov']
        self.difference = {}
        for key in keys_1:
            self.difference[key] = np.zeros((len(self.dataset), ))
        for key in keys_2:
            self.difference[key] = 0.0

    def estimate(self):
        self.compute_B()
        self.update_alpha()
        self.update_beta()
        self.update_Gamma()
        self.update_gamma()
        self.update_ksai()
    
    def maximize(self):
        self.update_A()
        self.update_P()
        self.update_C()
        self.update_miu()
        self.update_cov()

    def compute_B(self):
        # compute bjmOt for each seq
        # compute BjOt for each seq
        if not self.simplify:
            for i in range(self.state_num):
                for j in range(self.gaussian_num):
                    if np.linalg.det(self.cov[i,j])==0:
                        print 'singular cov'
                        self.cov[i,j] = np.diag(np.diag(self.cov[i,j]))
        for idx in range(len(self.dataset)):
            old_b, old_B = self.b[idx], self.B[idx]
            self.b[idx] = get_b(self.dataset[idx], self.miu, self.cov, simplify=self.simplify) # [N, M, T]
            self.B[idx] = get_B(self.b[idx], self.C) # [N, T]
            self.difference['b'][idx], self.difference['B'][idx] = l1_norm(self.b[idx], old_b), l1_norm(self.B[idx], old_B)

    def update_alpha(self):
        # compute alpha for each sequence
        for idx in range(len(self.alpha)):
            # initialize
            alpha = np.zeros_like(self.alpha[idx]) # [N, T]
            alpha[:, 0] = self.P * self.B[idx][:, 0]
            alpha[:, 0] = save_divide(alpha[:, 0], np.sum(alpha[:, 0]))
            # iterations
            for t in range(alpha.shape[1] - 1):
                alpha[:, t+1] = np.sum(np.tile(alpha[:, t], (self.state_num, 1)).T * self.A, 0) * self.B[idx][:, t+1]
                alpha[:, t+1] = save_divide(alpha[:, t+1], np.sum(alpha[:, t+1]))
            
            self.difference['alpha'][idx] = l1_norm(alpha, self.alpha[idx])
            self.alpha[idx] = alpha.copy()

    def update_beta(self):
        # compute beta for each sequence
        for idx in range(len(self.beta)):
            # initialize
            beta = np.zeros_like(self.beta[idx])
            beta[-1, -1] = 1.0
            # iterations
            for t in range(beta.shape[1] - 2, -1, -1):
                beta[:, t] = np.sum(self.A * beta[:, t+1] * self.B[idx][:, t+1], 1)
                beta[:, t] = save_divide(beta[:, t], np.sum(beta[:, t]))

            self.difference['beta'][idx] = l1_norm(beta, self.beta[idx])
            self.beta[idx] = beta.copy()

    def update_Gamma(self):
        # compute gamma_jt for each seqeunce
        for idx in range(len(self.Gamma)):
            Gamma = self.alpha[idx] * self.beta[idx] # [N,T]
            sums = np.sum(Gamma, axis=0) # [T,]
            Gamma /= np.where(sums!=0, sums, 1e-15)
            self.difference['Gamma'][idx] = l1_norm(Gamma, self.Gamma[idx])
            self.Gamma[idx] = Gamma.copy()
    
    def update_gamma(self):
        # compute gamma_ijt for each seqeunce
        for idx in range(len(self.gamma)):
            gamma = np.expand_dims(self.C, -1) * self.b[idx] # [N,M,T]
            down = np.expand_dims(np.sum(gamma, 1), 1) # [N,1,T]
            gamma /= np.where(down!=0, down, 1e-15) # [N,M,T]
            gamma *= np.expand_dims(self.Gamma[idx], 1) # [N,M,T]
            self.difference['gamma'][idx] = l1_norm(gamma, self.gamma[idx])
            self.gamma[idx] = gamma.copy()

    def update_ksai(self):
        # compute ksai_ijt for each seqeunce
        for idx in range(len(self.ksai)):
            ksai = np.zeros_like(self.ksai[idx]) # [N,N,T]
            for i in range(self.state_num):
                for j in range(self.state_num):
                    ksai[i, j, :-1] = self.alpha[idx][i, :-1] * self.A[i,j] * self.B[idx][j, 1:] * self.beta[idx][j, 1:]
            sums = np.sum(np.sum(ksai, 0), 0) # [T,]
            ksai /= np.where(sums!=0, sums, 1e-15)
            self.difference['ksai'][idx] = l1_norm(ksai, self.ksai[idx])
            self.ksai[idx] = ksai.copy()

    def update_A(self):
        up = np.zeros_like(self.A)
        down = np.zeros_like(self.A)
        for idx in range(len(self.dataset)):
            up += np.sum(self.ksai[idx][:, :, :-1], -1)
            down += np.tile(np.sum(self.Gamma[idx][:, :-1], -1)[:,None], self.state_num)
        
        A = up / np.where(down!=0, down, 1e-15)
        self.difference['A'] = l1_norm(A, self.A)
        self.A = A.copy()

    def update_P(self):
        P = np.zeros_like(self.P)
        for idx in range(len(self.dataset)):
            P += self.Gamma[idx][:, 0]
        P /= float(len(self.dataset))
        self.difference['P'] = l1_norm(P, self.P)
        self.P = P.copy()
    
    def update_C(self):
        up = np.zeros_like(self.C) # [N,M]
        down = np.zeros_like(self.C)
        for idx in range(len(self.dataset)):
            tmp = np.sum(self.gamma[idx], -1) # [N,M]
            up += tmp
            down += np.tile(np.sum(tmp, -1)[:, None], self.gaussian_num) # [N,M]
        C = up / np.where(down!=0, down, 1e-15)
        self.difference["C"] = l1_norm(C, self.C)
        self.C = C.copy()
    
    def update_miu(self):
        up = np.zeros_like(self.miu) # [N,M,d]
        down = np.zeros_like(self.miu)
        for idx in range(len(self.dataset)):
            # sum([N,M,T,1] * [1,1,T,d], 2)->[N,M,d]
            up += np.sum(np.expand_dims(self.gamma[idx], -1) * np.expand_dims(np.expand_dims(self.dataset[idx], 0), 0), axis=2)
            # [N,M,1]
            down += np.expand_dims(np.sum(self.gamma[idx], -1), -1)
        miu = up / np.where(down!=0, down, 1e-15)
        self.difference['miu'] = l1_norm(miu, self.miu)
        self.miu = miu.copy()
    
    def update_cov(self):
        up = np.zeros_like(self.cov) # [N,M,d,d]
        down = np.zeros_like(self.cov) # [N,M,d,d]
        for idx in range(len(self.dataset)):
            # [1,1,T,d] - [N,M,1,d] -> [N,M,T,d]
            difference = np.expand_dims(np.expand_dims(self.dataset[idx], 0), 0) - np.expand_dims(self.miu, 2)
            if self.simplify:
                # sum([N,M,T,1] * [N,M,T,d], 2) -> [N,M,d]
                up += np.sum(np.expand_dims(self.gamma[idx], -1) * np.square(difference), 2)
                # [N,M,1]
                down += np.expand_dims(np.sum(self.gamma[idx], -1), -1)
            else:
                # sum([N,M,T,1,1] * ([N,M,T,d,1] dot [N,M,T,1,d]), 2) -> [N,M,d,d]
                up += np.sum(np.expand_dims(np.expand_dims(self.gamma[idx], -1), -1) * np.matmul(np.expand_dims(difference,-1), np.expand_dims(difference,-2)), 2)
                # [N,M,1,1]
                down += np.expand_dims(np.expand_dims(np.sum(self.gamma[idx], -1), -1), -1)
        cov = up / np.where(down!=0, down, 1e-15)
        self.difference['cov'] = l1_norm(cov, self.cov)
        self.cov = cov.copy()
    
    def judge_convergence(self):
        if self.difference['A'] < self.thresholds['A'] \
            or self.difference['cov'] < self.thresholds['cov'] \
            or self.difference['miu'] < self.thresholds['miu']:
            return True 
        else:
            return False
    
    def load_model(self, model_path):
        model = np.load(model_path)
        for key in model.keys():
            self.__dict__[key] = model[key]
        
    def save_model(self, model_path):
        print 'saved to {}'.format(model_path)
        to_save={
            'P':self.P, # [N,]
            'A':self.A, # [N, N]
            'C':self.C, # [N, M]
            'miu':self.miu, # [N, M, d]
            'cov':self.cov # [N, M, d]
        }
        np.savez(model_path, **to_save)
    
    def Viterbi_decode(self):
        # don't forget to compute B first
        self.compute_B()
        for idx in range(len(self.dataset)):
            delta = np.zeros_like(self.delta[idx])
            # P = np.where(self.P!=0, self.P, 1e-15) # i.e. [0.99,0.01,0.00,0.00]
            # B = np.where(self.B[idx][:, 0]!=0, self.B[idx][:, 0], 1e-10)
            delta = Log(self.P) + Log(self.B[idx][:, 0])
            for t in range(1, self.B[idx].shape[1]):
                # A = np.where(self.A!=0, self.A, 1e-15)
                # B = np.where(self.B[idx][:, t]!=0, self.B[idx][:, t], 1e-15)
                delta = np.max(np.tile(delta[:, None], self.state_num) + Log(self.A), 0) + Log(self.B[idx][:, t])
            self.delta[idx] = delta.copy()

    def train(self, iterations=10):
        for idx in range(iterations):
            self.estimate()
            self.maximize()
            
            if idx%10==0:
                print self.name+" iter [%3d/%3d] A=%.7f miu=%.5f cov=%.5f" % (idx, iterations, self.difference['A'], self.difference['miu'], self.difference['cov'])

            if self.judge_convergence():
                break

