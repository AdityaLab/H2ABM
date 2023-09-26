# python3 evaluate.py --casefile 'data/synthetic.csv' --hypergraph_result 'output/resulth.pkl'  --graph_result 'output/resultg.pkl' --outputfile 'output/result.csv'

import numpy as np
import argparse
import pickle
import math

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--casefile', type=str)
    parser.add_argument('--hypergraph_result', type=str)
    parser.add_argument('--graph_result', type=str)
    parser.add_argument('--outputfile', type=str)

    args = parser.parse_args()

    with open(args.hypergraph_result,'rb') as f:
        res_h, _ = pickle.load(f)
        
    with open(args.graph_result,'rb') as f:
        res_g, _ = pickle.load(f)

    obs = []

    with open (args.casefile,'r') as DataFile:
        DataFile.readline()
        while (True):
            Sentence = DataFile.readline()
            if not Sentence:
                break
            else:
                Sentence = Sentence.split(',')
                obs.append(float(Sentence[1]))

    res_h = np.mean(res_h, axis=1)
    res_g = np.mean(res_g, axis=1)

    with open(args.outputfile,'w') as DataFile:
        DataFile.write('Model,NRMSE,ND,Pearson correlation\n')
        DataFile.write('Hypergraph-HeterSIS,{:.4f},{:.4f},{:.4f}'.format(np.sqrt(np.mean(np.power(obs-res_h,2)))/np.mean(obs),np.sum(np.abs(obs-res_h))/np.sum(obs),np.corrcoef(obs,res_h)[0,1])+'\n')
        DataFile.write('Graph-HeterSIS,{:.4f},{:.4f},{:.4f}'.format(np.sqrt(np.mean(np.power(obs-res_g,2)))/np.mean(obs),np.sum(np.abs(obs-res_g))/np.sum(obs),np.corrcoef(obs,res_g)[0,1])+'\n')

