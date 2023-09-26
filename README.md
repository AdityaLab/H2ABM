# H2ABM: Heterogeneous Agent-based Model on Hypergraphs to Capture Group Interactions

## Setup

First install Anaconda. The dependencies are listed in `environment.yml` file. 

Then run the following commands:

```bash
conda env create --prefix ./envs/hhabm --file environment.yml
source activate ./envs/hhabm
```

## Directory structure

```
-data
       - synthetic.pkl -> save synthetic hypergraphs as pkl file
       - synthetic.csv -> save synthetic number of cases as csv file
- model.py -> implementation of Hypergraph-HeterSIS model
- calibrateh.py -> calibrate the Hypergraph-HeterSIS model to the synthetic number of cases
- simulateh.py -> run simulations for the Hypergraph-HeterSIS model based on the calibrated parameters
- evaluate.py -> Calculate the NRMSE, ND, and Pearson correlation for Hypergraph-HeterSIS model results
- outputs -> save results
```

## Dataset

The dataset is at `data` folder. It contains the hypergraph file (synthetic.pkl) and the number of cases (synthetic.csv) used for Hypergraph-HeterSIS model calibration. 

## Demo

We provde a demo code to calibrate the Hypergraph-HeterSIS model and calculate the metrics we used in main article
Run:

```
chmod 777 run.sh
./run.sh
```
This will save the results (NRMSE, ND, and Pearson correlation value) in `output/result.csv`
