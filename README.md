# Clustering Geospatial Data - 

A python implementation of clustering algorithm with maximum cluster size constraint. The two aspects that are important here are:
1. The cluster size (or the deviation from the desired cluster size)
2. The quality of the clusters (i.e. how similar are points within a cluster). 

```
├── README.md         <- The top-level README for developers using this project.
│
├── requirements.txt  <- Required Python libraries
│
├── .gitignore        <- Contains files that must not go to git remote.
│
└── src               <- Source code for use in this project.
│    ├── __init__.py   <- Makes src a Python module
│    │
│    └── api.py       <- Handles the requisition, makes calculation and call optimization.
│    
├── data              <- Scripts to download or generate data
│
├── references        <- Data dictionaries, manuals, and all other explanatory materials.
│
└── tests             <- Scripts to test models 

```

## Run API

### Run locally

Go to ***src/*** and run:
```
$ python api.py --f "name_file.csv"
```
