# [SDM 2024] H2ABM: Heterogeneous Agent-based Model on Hypergraphs to Capture Group Interactions (Best Poster Award)

## Publication

Implementation of the paper "H2ABM: Heterogeneous Agent-based Model on Hypergraphs to Capture Group Interactions."

Authors: Vivek Anand*, Jiaming Cui*, Jack Heavey, Anil Vullikanti, B. Aditya Prakash

*Equal contribution

Venue: SDM 2024

Link to the paper: https://sites.cc.gatech.edu/~jcui75/papers/h2abm-sdm24.pdf

We also release an anonymized version of the UVA dataset at: https://github.com/AdityaLab/UVA-Hypergraph/

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

## Cite our work
If you find our work useful, please cite our work:
- Vivek Anand*, Jiaming Cui*, Jack Heavey, Anil Vullikanti, B. Aditya Prakash. H2ABM: Heterogeneous Agent-based Model on Hypergraphs to Capture Group Interactions. SDM 2024.
```
@inproceedings{anand2024h2abm,
  title={},
  author={Anand, Vivek and Cui, Jiaming and Heavey, Jack and Vullikanti, Anil and Prakash, B Aditya},
  booktitle={Proceedings of the 2024 SIAM International Conference on Data Mining (SDM)},
  pages={280--288},
  year={2024},
  organization={SIAM}
}
```
