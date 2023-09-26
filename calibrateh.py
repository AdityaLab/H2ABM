import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import scipy.sparse as sp
import networkx as nx
import numpy as np
import warnings
import datetime
import argparse
import random
import pickle
import math
import csv
import sys
from tqdm import tqdm
from glob import glob
from model import lhsu, checkbound, simulate_hyper_weekly
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--graphfile', type=str)
    parser.add_argument('--casefile', type=str)
    parser.add_argument('--outputfile', type=str)
    parser.add_argument('--Iter', type=int, default=20)
    parser.add_argument('--num_ens', type=int, default=300)

    args = parser.parse_args()

    Iter = args.Iter
    num_ens = args.num_ens

    with open(args.graphfile, 'rb') as f:
        Hs, H_ts, P, H, L = pickle.load(f)

    N = P + H + L

    t = 0

    infp_low, infp_high = 0, 0.02
    mu_low, mu_high = 0, 10
    sigma_low, sigma_high = 0, 10

    alpha_low, alpha_high = 0, 10
    beta_low, beta_high = 0, 2e-3
    delta_low, delta_high = 0, 0.1

    tau_p2p_low, tau_p2p_high = 0, 0.02
    tau_p2h_low, tau_p2h_high = 0, 0.02
    tau_p2l_low, tau_p2l_high = 0, 0.02
    tau_h2p_low, tau_h2p_high = 0, 0.02
    tau_h2h_low, tau_h2h_high = 0, 0.002
    tau_h2l_low, tau_h2l_high = 0, 0.02
    tau_l2p_low, tau_l2p_high = 0, 0.02
    tau_l2h_low, tau_l2h_high = 0, 0.01

    # Hypergraph G_threshold
    g_threshold_low, g_threshold_high = 0.25, 0.75

    parameter_low = np.array([infp_low,mu_low,sigma_low,
                              alpha_low, beta_low, delta_low,
                              tau_p2p_low,tau_p2h_low, tau_p2l_low,tau_h2p_low,tau_h2h_low,tau_h2l_low,tau_l2p_low,tau_l2h_low, 
                              g_threshold_low])

    parameter_high = np.array([infp_high,mu_high,sigma_high,
                               alpha_high, beta_high, delta_high,
                               tau_p2p_high,tau_p2h_high, tau_p2l_high,tau_h2p_high,tau_h2h_high,tau_h2l_high,tau_l2p_high,tau_l2h_high,
                               g_threshold_high])
        
    obs_truth = []

    with open (args.casefile,'r') as DataFile:
        DataFile.readline()
        while (True):
            Sentence = DataFile.readline()
            if not Sentence:
                break
            else:
                Sentence = Sentence.split(',')
                obs_truth.append(float(Sentence[1]))

    obs_truth = obs_truth[:len(Hs)+1]
    Result_whole = []

    obs_var = 1+np.power((0.1*np.array(obs_truth)),2)

    num_times = len(obs_truth)
        
    num_var = len(parameter_high)
    Theta = np.zeros((num_var,Iter+1))

    sig = np.zeros(Iter)
    alp = 0.9
    SIG = np.power((parameter_high - parameter_low),2)/4/len(obs_truth)

    obsprior = np.zeros((num_times,num_ens,Iter))
    obspost = np.zeros((num_times,num_ens,Iter))
    xinitialrec = np.zeros((num_var,num_ens,Iter))
    xpostrec = np.zeros((num_var,num_ens,num_times,Iter))

    for n in tqdm(range(Iter)):
        xprior = np.zeros((num_var,num_ens,num_times))
        xpost = np.zeros((num_var,num_ens,num_times))
        sig[n] = math.pow(alp,n)
        So = np.zeros((num_var,num_ens))
        Sigma = np.diag(sig[n]*sig[n]*SIG)

        if (n == 0):
            x0 = lhsu(parameter_low,parameter_high,num_ens)
            So = x0
            Theta[:,0] = np.mean(x0,axis=1)
        else:
            So = np.random.multivariate_normal(Theta[:,n],Sigma,num_ens).T

        So = checkbound(So,parameter_low,parameter_high)
        xinitialrec[:,:,n] = So

        states_all = np.zeros((num_ens,P))
        loads_all = np.zeros((num_ens,P+H+L))

        tau_dict = {'P':{'P': np.mean(So[6]), 'H': np.mean(So[7]), 'L': np.mean(So[8])}, \
            'H':{'P': np.mean(So[9]), 'H': np.mean(So[10]), 'L': np.mean(So[11])}, \
            'L':{'P': np.mean(So[12]), 'H': np.mean(So[13]), 'L': 0.0}}

        init_loads = []
        seeds = []

        for p in range(P):
            if np.random.rand() < np.mean(So[0]):
                seeds.append(p)
                init_loads.append(np.random.normal(np.mean(So[1]), np.mean(So[2])))
            else:
                init_loads.append(0.01*np.random.normal(np.mean(So[1]), np.mean(So[2])))

        init_loads = np.array(init_loads)
                
        loads_all[:,:P] = init_loads
        states_all[:,seeds] = 1

        for t in range(len(obs_truth)):
            print ('Iteration:',n,'Week:',t)
    
            obs = []

            for counter in range(num_ens):
                states_all[counter], loads_all[counter], cases = simulate_hyper_weekly(Hs, H_ts, P, H, L, tau_dict, loads_all[counter], states_all[counter], So[:,counter], 7*t)
                obs.append(np.sum(cases))

            xprior[:,:,t] = So
            obsprior[t,:,n] = obs

            prior_var = np.var(obs)
            post_var = prior_var*obs_var[t]/(prior_var+obs_var[t])

            if (prior_var == 0):
                post_var = 0
                prior_var = 0.001

            prior_mean = np.mean(obs)
            post_mean = post_var*(prior_mean/prior_var + obs_truth[t]/obs_var[t])    
            EAKF_alpha = math.pow((obs_var[t]/(obs_var[t]+prior_var)),0.5)
            dy = post_mean + EAKF_alpha * (obs-prior_mean) - obs
            rr = np.zeros((num_var))
            for j in range(num_var):
                A = np.cov(So[j,:],obs)
                rr[j] = (A[0,1])/prior_var
            dx = np.dot(rr.reshape(1,-1).T,dy.reshape(1,-1))

            So = So + dx
            So = checkbound(So,parameter_low,parameter_high)
            xpost[:,:,t] = So
            obspost[t,:,n] = obs+dy

        xpostrec[:,:,:,n] = xpost
        temp = np.squeeze(np.mean(xpost,1))
        Theta[:,n+1] = np.mean(temp,1)

        Result_whole = np.copy(xpost[:,:,-1])

    Result = Theta[:,Iter]

    with open(args.outputfile, 'wb') as f:
        pickle.dump(Result, f)
