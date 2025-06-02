# DiDScanKDD2025
Source code and replication files for Difference-in-Differences Subset Scan, at KDD 2025.

# Files 

## Figures and Tables
Each file in this folder runs simulations to produce data as in a table or figure from the paper.

### figure_3.py
Simulations showing rejection power of DiD-Scan across effect size, affected region size and number of affected outcome variables (of three measured outcomes). 

### figure_4.py 
Simulations comparing DiD-Scan with GATEs and HTE-BH, with respect to overlap (accuracy of selected subset) as shown in the paper. 

### p2p_lending_analysis.py
Replicates tables 2 and 3 from the paper, showing the discovered subsets in the peer to peer lending study, as well as the results from the contribution/adjacent subset analysis. 

## Functions 

Algorithms and functions used in the Figures and Tables files. 

### bh.py 

Replication of HTE-BH from "False Discovery Rate Controlled Heterogeneous Treatment Effect Detection for Online Controlled Experiments" by Yuxiang Xie, Nanyu Chen, and Xiaolin Shi. 

### did_ss_corr.py

Algorithm and assistive functions for running DiD-Scan with correlated outcomes across time. The main function is "cor_search", which references other functions in did_ss_corr.py in its operation. 

### diff_in_diff.py 

Given a dataset, control-treatment assignment and pre and post time-periods, calculates the treatment effect for a simple difference in differences analysis. Used in some post-hoc analysis. 

### generate_wando_data.py

Simulations in the paper use data simulated from the estimated distribution from Wang and Overby (2021). This file generates the data. 

### genericml.py

Implementation of Chernozhukov et al.'s Generic ML estimators, especially GATEs, used in the comparison simulations. 

### make_data.py

Used to generate simulated data, and also defines a custom version of xarrays used in some of the algorithms. 

### rand_test.py

Used for permutation testing for assessing significance of the discovered subset. 