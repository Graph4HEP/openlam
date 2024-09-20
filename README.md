# A crystal structure optimization tool built on top of ASE

## The origin README can be found [here](https://github.com/deepmodeling/openlam)

## Installation

After cloning this project, install `lam-crystal-philately` with common dependencies (including requirements for workflows) by
```
pip install .
```
To install additional dependencies for DP
```
pip install ".[dp]"
```
or mace
```
pip install ".[mace]"
```

Download the latest DP model for structure optimization by
```
wget https://bohrium-api.dp.tech/ds-dl/lam-crystal-model-01oe-v3.zip
unzip lam-crystal-model-01oe-v3.zip
```

Note: To change the logic of the optimization, change the code in `lam_optimize` and do `pip install .` again to active the changes.


## Structure Optimization

### Commandline tool

To optimize structures using DP model
```
lam-opt relax -i examples/data -m <path-to-DP-model>
```
or using mace
```
lam-opt relax -i examples/data -t mace
```

