import numpy as np
import xarray as xr
import pandas as pd
#from numpy import random as rd
from functions.diff_in_diff import diff_in_diff as did
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


def quant_splitter(df, cov_names, suffix = None, keep_old_vars = True, nbins = None):

    if not suffix:
        suffix = "_binned"

    if not nbins: 
        nbins = [3 for i in cov_names]
    elif isinstance(nbins, int):
        nbins = [nbins for i in cov_names]

    for i in range(len(cov_names)): 
        cn = cov_names[i]
        nb = nbins[i]
        df[cn+suffix] = pd.qcut(df[cn], q=nb, labels = [str(i + 1) for i in np.arange(nb)])

    if not keep_old_vars: 
        df = df.drop(cov_names, axis = 1)
        # give the binned vars the old names: 
        df = df.rename({i + suffix: i for i in cov_names}, axis = 1)

    return df

def bh(ps, level):
    """ For now, using statsmodels built-in"""

    return 2 + 2



def hte_bh(data, search_vars = None, search_times = True, n_processes = 1, already_binned = True):

    # get ATE using diff in diff: 
    did_ate, did_se, did_p = did(data, model_type = 'twfe')

    # convert from xarray to pandas: 
    df = data.to_dataframe().reset_index()      

    # quantbin: 
    if already_binned:
        df = quant_splitter(df, search_vars, keep_old_vars=False)

    df = df.sort_values(by = ['unit', 'stream', 'period'])

    # get first differences for all individuals: 
    # TODO: change this so that it adapts to any selected reference periods
    premean = df[df.treat_time == 0].groupby(['unit', 'stream'])['outcome'].mean().reset_index().rename({"outcome":"premean"}, axis = 1)
    df = pd.merge(df, premean)
    df['dif'] = df['outcome'] - df['premean']

    # make M, the baseline values based on the covariates: 
    baseline = df[(df.group == "C")].groupby([i for i in np.append(search_vars, ["stream", "period"])], observed = False)["dif"].mean().reset_index().rename({"dif":"baseline"}, axis = 1)
    df = pd.merge(df, baseline)

    # FOR NOW: RATHER THAN USE FIRST DIFFS, ADD TREATGROUP REF MEAN TO BASELINE: 
    df['baseline'] = df['baseline'] + df['premean']

    search_dims = search_vars.copy()
    search_dims.extend(["stream"])
    if search_times: 
        search_dims.extend(["period"])

    # filter down to post-treatment: 
    dfpost = df[(df['treat_time']==1)].copy()
    dfpost['treat_group_binary'] = 0
    dfpost['treat_group_binary'] = dfpost['treat_group_binary'] + (dfpost['group'] != "C")

    # get tensor cells by search_dims: 
    tensor = dfpost.groupby(search_dims, observed = True).agg(
                outcome=('outcome', 'mean'),
                baseline=('baseline', 'mean'),
                outcome_sd=('outcome', 'std'),
                count=('outcome', 'count'),  # You can use any column for count
                p = ('treat_group_binary', 'mean') # probability of treatment conditional on subgroup
            ).reset_index()
    
    # drop rows with no treatment or no control: 
    tensor = tensor[(~np.isnan(tensor['outcome'])) & (~np.isnan(tensor['baseline']))]
    # drop rows with 0 or 1 p: 
    tensor = tensor[(~(tensor.p == 0)) & (~(tensor.p == 1))]

    # get group-unique id: 
    tensor['group_name'] = [str(i) for i in np.arange(len(tensor))]

    # add group-unique id to df: 
    dfpost = dfpost.merge(tensor[np.append(search_dims, ['group_name', 'p'])], how = "right")

    # make design matrix: 
    dmat = np.zeros((len(dfpost), len(dfpost.group_name.unique())))
    for j in np.arange(len(dfpost)):
        dmat[j, int(dfpost['group_name'][j])] = 1

    # get tform_outcome then get its deviation from ATE: 
    dfpost['tform_outcome'] = dfpost['dif'] * ((dfpost['group'] != "C") - dfpost['p']) / (dfpost['p']* (1 - dfpost['p']))
    dfpost['tform_outcome_dev'] = dfpost['tform_outcome'] - did_ate

    # run lm of the transformed outcome on the design matrix: (NOT DEVIATIONS BECAUSE THAT'S NOT REALLY ANALAGOUS TO WHAT DIDSCAN IS DOING)
    model = sm.OLS(dfpost['tform_outcome'], dmat, missing='drop')
    # print(model.fit().summary())
    
    p_values = model.fit().pvalues

    # Perform FDR correction (Benjamini-Hochberg method)
    tf, p_adj, _, _ = multipletests(p_values, alpha = 0.05, method='fdr_bh')
    tensor['inS'] = tf

    # join "tf" (reject or not reject) to df, so that we can assess whether an individual is in S or not: 
    dfpost = dfpost.merge(tensor[np.append(search_dims, ['inS'])], how = "left")

    # filter to just treated individuals: 
    treat_df = dfpost[dfpost['group'] != 'C']

    # assess overlap: 
    if len(treat_df[(treat_df['group'] == 'A') | (treat_df['inS'])]) > 0 :
        ol = len(treat_df[(treat_df['group'] == 'A') & (treat_df['inS'])]) / len(treat_df[(treat_df['group'] == 'A') | (treat_df['inS'])])
    else: 
        ol = 0

    return ol