python3 calibrateh.py --graphfile 'data/synthetic.pkl' --casefile 'data/synthetic.csv' --outputfile 'output/parameterh.pkl' --Iter 10 --num_ens 100
python3 simulateh.py --graphfile 'data/synthetic.pkl' --parameterfile 'output/parameterh.pkl' --casefile 'data/synthetic.csv' --outputfile 'output/resulth.pkl' --num_ens 100
python3 evaluate.py --casefile 'data/synthetic.csv' --hypergraph_result 'output/resulth.pkl'  --graph_result 'output/resultg.pkl' --outputfile 'output/result.csv'
