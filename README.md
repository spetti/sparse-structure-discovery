# sparse-structure-discovery

This repo contains the code associated with the paper (citation coming soon)

Data: (link coming soon)

Installation: A list of the libraries required to execute the notebooks is in requirements.txt
To install the libraries, run (ideally in a virtual environment)
pip install -r requirements.txt

Structure of repository

SSD: Sparse Structure Discovery

1. Tutorial Jupyter notebook (SSD/Example.ipynb)
2. Python script that generates factorizer objects for the data analyzed in the paper (SSD/factorizer_examples.py)
3. Notebooks generating the figures presented in the paper (SSD/{bbq, genotoxin, hub_synthetic, independent_synthetic, kinsler}_figures.ipynb)
4. Utils folder containing code that executes our ssd method (SSD/utils/ssd.py) and stores solutions across a range of regularization values (SSD/utils/factorizer.py)

QTL: Multi-phenotype QTL mapping.

This folder contains joint_qtl_mapping.py, which executes our joint QTL mapping pipeline. Code in this folder recreates the QTL mapping on the BBQ dataset. See README in folder for further description. 


