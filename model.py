import networkx as nx
from scipy import sparse
import numpy as np
from math import cos, sin, pi
from glob import glob
import numpy as np
import scipy
from numpy.linalg import inv
from scipy.stats import multivariate_normal
from tqdm import tqdm
import scipy.sparse as sp
import pickle
import random

def lhsu(xmin,xmax,nsample):
    nvar = len(xmin)
    ran = np.random.rand(nsample,nvar)
    s = np.zeros((nsample,nvar))
    for j in range(nvar):
        idx = list(range(nsample))
        random.shuffle(idx)
        idx = np.array(idx)
        P = (idx-ran[:,j])/nsample;
        s[:,j] = xmin[j] + np.dot(P,xmax[j]-xmin[j])
    s = s.T
    return s

def checkbound(x,xmin,xmax):
    n = x.shape[0]
    for i in range(n):
        x[i,:][x[i,:] < xmin[i]] = xmin[i]
        x[i,:][x[i,:] > xmax[i]] = xmax[i]
    return x

def fill_in_taus(A, P, H, L, taus):
    
    tau_p2p, tau_p2h, tau_p2l, tau_h2p, tau_h2h, tau_h2l, tau_l2p, tau_l2h = taus

    R = A.tocoo()

    R.setdiag(0)
    R.sum_duplicates()
    R.eliminate_zeros()
    for idx, i, j in zip(range(R.nnz), R.row, R.col):
        if i < P and j < P:
            R.data[idx] *= tau_p2p
        elif i < P and j < P+H:
            R.data[idx] *= tau_h2p
        elif i < P and j < P+H+L:
            R.data[idx] *= tau_l2p
        elif i < P+H and j < P:
            R.data[idx] *= tau_p2h
        elif i < P+H and j < P+H:
            R.data[idx] *= tau_h2h
        elif i < P+H and j < P+H+L:
            R.data[idx] *= tau_l2h
        elif i < P+H+L and j < P:
            R.data[idx] *= tau_p2l
        elif i < P+H+L and j < P+H:
            R.data[idx] *= tau_h2l
        else:
            R.data[idx] *= 0

    return R.tocsr()

def simulate_subroutine(H_now, H_t_now, P, H, L, taus, loads, states, alpha, g_threshold):
    N = P + H + L
    ## TODO
    ## 1) Aggregate patient, HCW and Location loads in parallel using the various taus 
    ###### REMEMBER TO USE THE RIGHT TAUS, we have three for each node type
    ## 2) Sum up over vertices
    ## 3) Elementwise product with H with calculated loads
    ## 4) Apply g over each hyperedge and sum up over hyperedges (Use tanh and g_threshold)
    
    #print(f"new_loads shape is {new_loads.shape}")
    #print(f"states shape is {states.shape}")
    
    #print(f"P is {P}")
    #print(f"H shape is {H.get_shape()}")
    
    out_pp = H_now[:, :P] * taus['P']['P']
    out_ph = H_now[:, P:P+H] * taus['P']['H']
    out_pl = H_now[:, P+H:] * taus['P']['L']
    patient_out = np.squeeze(np.asarray(sparse.hstack([out_pp, out_ph, out_pl]).sum(axis=1)))
    
    out_hp = H_now[:, :P] * taus['H']['P']
    out_hh = H_now[:, P:P+H] * taus['H']['H']
    out_hl = H_now[:, P+H:] * taus['H']['L']
    hcw_out = np.squeeze(np.asarray(sparse.hstack([out_hp, out_hh, out_hl]).sum(axis=1)))
    
    out_lp = H_now[:, :P] * taus['L']['P']
    out_lh = H_now[:, P:P+H] * taus['L']['L']
    out_ll = H_now[:, P+H:] * taus['L']['L']
    location_out = np.squeeze(np.asarray(sparse.hstack([out_lp, out_lh, out_ll]).sum(axis=1)))

    patient_outs = np.squeeze(np.asarray(H_t_now[:P, :].multiply(patient_out).sum(axis=1)))
    hcw_outs = np.squeeze(np.asarray(H_t_now[P:P+H, :].multiply(hcw_out).sum(axis=1)))
    location_outs = np.squeeze(np.asarray(H_t_now[P+H:, :].multiply(location_out).sum(axis=1)))

    outs = np.concatenate((patient_outs, hcw_outs, location_outs))

    new_loads = loads * (1-outs)
    
    h_l = H_now.multiply(loads).tocsr()
    #print(f"h_l shape is {h_l.shape}")
    #print(f"N is {N}")
    # big_matrix = sparse.csc_matrix((H.get_shape()[0],N), dtype='float32')
    # Patients Receiving
    l_pp = h_l[:, :P] * taus['P']['P']
    l_ph = h_l[:, P:P+H] * taus['P']['H']
    l_pl = h_l[:, P+H:] * taus['P']['L']
    patient_sum = np.squeeze(np.asarray(np.tanh(sparse.hstack([l_pp, l_ph, l_pl]).sum(axis=1))*g_threshold))
    #patient_sum = np.squeeze(np.asarray(sparse.hstack([l_pp, l_ph, l_pl]).sum(axis=1)))
    #print(patient_sum)
    # HCW Receiving
    l_hp = h_l[:, :P] * taus['H']['P']
    l_hh = h_l[:, P:P+H] * taus['H']['H']
    l_hl = h_l[:, P+H:] * taus['H']['L']
    hcw_sum = np.squeeze(np.asarray(np.tanh(sparse.hstack([l_hp, l_hh, l_hl]).sum(axis=1))*g_threshold))
    #hcw_sum = np.squeeze(np.asarray(sparse.hstack([l_hp, l_hh, l_hl]).sum(axis=1)))
    # Location Receiving
    l_lp = h_l[:, :P] * taus['L']['P']
    l_lh = h_l[:, P:P+H] * taus['L']['H']
    l_ll = h_l[:, P+H:] * taus['L']['L']
    location_sum = np.squeeze(np.asarray(np.tanh(sparse.hstack([l_lp, l_lh, l_ll]).sum(axis=1))*g_threshold))
    #location_sum = np.squeeze(np.asarray(sparse.hstack([l_lp, l_lh, l_ll]).sum(axis=1)))
    # TODO: For further optimization, don't need to use variables
    # Sum up loads for each node type into the big matrix
    #print(f"h_l shape is {h_l.shape}")
    #print(f"patient_sum shape is {patient_sum.shape}\n{patient_sum}")
    #print(f"big_matrix shape is {big_matrix.shape}")
    #print(f"h_l.T*patient_sum shape is {h_l.T.multiply(patient_sum).T.shape}")
    ### TODO ###
    ### SERIOUSLY RETHINK THIS IMPLEMENTATION, converting a bit toooo much ####
    ### USE LINE PROFILER ###
    patient_loads = np.squeeze(np.asarray(H_t_now[:P, :].multiply(patient_sum).sum(axis=1)))
    hcw_loads = np.squeeze(np.asarray(H_t_now[P:P+H, :].multiply(hcw_sum).sum(axis=1)))
    location_loads = np.squeeze(np.asarray(H_t_now[P+H:, :].multiply(location_sum).sum(axis=1)))
    ## WILL NEED TO CHANGE THIS TO ACCOUNT FOR LOCATIONS IN MODEL WITH MORE LOCATIONS
    #location_loads = np.array([0.0]*(L+1))
    #print(f"patient_loads shape is {patient_loads.shape}\n{patient_loads}")
    #print(f"hcw_loads shape is {hcw_loads.shape}\n{hcw_loads}")
    #print(f"location_loads shape is {location_loads.shape}\n{location_loads}")
    #print(f"H_t.shape is {H_t.shape}")

    new_loads += np.concatenate((patient_loads, hcw_loads, location_loads))
    
    return new_loads

def simulate_hyper_weekly(Hs, H_ts, P, H, L, taus, loads, states, parameters, t):
    
    # Hs: List of Hypergraphs. Format is (H,V) where H is the hypergraph and V is the list of vertices/nodes
    # H_ts = Hs.T

    alpha, beta, delta, g_threshold = parameters[3], parameters[4], parameters[5], parameters[-1]

    cases = []

    for time in range(7):
        rands = np.random.rand(P)

        s2i = np.where((rands < beta*loads[:P]) & (states == 0))[0]
        i2s = np.where((rands < delta) & (states == 1))[0]
            
        if s2i.size != 0:
            states[s2i] = 1
        if i2s.size != 0:
            states[i2s] = 0

        new_loads = simulate_subroutine(Hs[time], H_ts[time], P, H, L, taus, loads, states, alpha, g_threshold)

        loads = new_loads.copy()
        loads[:P] += alpha*states

        cases.append(len(np.where(states == 1)[0]))

    return states, loads, cases
