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
from model import simulate_hyper_weekly
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--graphfile', type=str)
    parser.add_argument('--parameterfile', type=str)
    parser.add_argument('--casefile', type=str)
    parser.add_argument('--outputfile', type=str)
    parser.add_argument('--num_ens', type=int, default=100)

    args = parser.parse_args()

    num_ens = args.num_ens

    with open(args.graphfile, 'rb') as f:
        Hs, H_ts, P, H, L = pickle.load(f)
        
    N = P + H + L

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

    with open(args.parameterfile, 'rb') as pkl:
        parameter = pickle.load(pkl)

    infp, mu, sigma,\
    alpha, beta, delta,\
    tau_p2p,tau_p2h,tau_p2l,tau_h2p,tau_h2h,tau_h2l,tau_l2p,tau_l2h,\
    g_threshold = list(parameter)

    states_all = np.zeros((num_ens,P))
    loads_all = np.zeros((num_ens,P+H+L))

    tau_dict = {'P':{'P': tau_p2p, 'H': tau_p2h, 'L': tau_p2l}, \
        'H':{'P': tau_h2p, 'H': tau_h2h, 'L': tau_h2l}, \
        'L':{'P': tau_l2p, 'H': tau_l2h, 'L': 0}}

    init_loads = []
    seeds = []

    for p in range(P):
        if np.random.rand() < infp:
            seeds.append(p)
            init_loads.append(np.random.normal(mu, sigma))
        else:
            init_loads.append(0.01*np.random.normal(mu, sigma))

    init_loads = np.array(init_loads)
            
    loads_all[:,:P] = init_loads
    states_all[:,seeds] = 1

    Result = []

    for t in range(len(obs_truth)):
        print ('Week:',t+1)

        obs = []

        for counter in range(num_ens):
            states_all[counter], loads_all[counter], cases = simulate_hyper_weekly(Hs, H_ts, P, H, L, tau_dict, loads_all[counter], states_all[counter], parameter, 7*t)
            obs.append(np.sum(cases))
        
        Result.append(obs)

    with open(args.outputfile, 'wb') as f:
        pickle.dump((Result,obs_truth), f)
