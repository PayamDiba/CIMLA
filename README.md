## CIMLA (Counterfactual Inference by Machine Learning and Attribution Models)
Saurabh Sinha’s Lab, University of Illinois at Urbana-Champaign [Sinha Lab](https://www.sinhalab.net/sinha-s-home)

## Description
CIMLA is a counterfactual inference method for identifying differential causal associations between two groups from observational data.

## Getting Started
CIMLA can be installed from PyPI, but we recommend to first create a conda environment:

```conda create --name CIMLA_env python=3.8.12```

Activate this environment and install CIMLA:

```pip install CIMLA```

## Run CIMLA
CIMLA can be used in the enviornment created above. A YAML config file (see below) is required to run CIMLA:

```cimla --config config_test.yaml```

## CIMLA Config File
The config file consists of four sections described below

### data:

* path_g1:





