import os

#os.chdir("C:\\Users\\wosta\\Documents\\Research Projects\\Difference in Differences Subset Scan\\did-subset-scan\\")

import numpy as np
import xarray as xr
import pandas as pd
#from numpy import random as rd
from numbers import Number
import functions.make_data as md
import sklearn.covariance as skcov
from itertools import product
import multiprocessing as mp
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import copy

import matplotlib.pyplot as plt

###################################################
## Scoring and Related Functions ##################
###################################################
    

def random_start(df, search_vars):
    """ Selects a random starting point in the search space """
    S = {}
    
    search_modes = search_vars + ["stream"]

    for v in search_modes:
        values = np.array([j for j in df[v].unique()])
        subset_size = np.random.randint(1, len(values) + 1)  # from 1 to n inclusive
        indices = np.random.choice(len(values), size=subset_size, replace=False)
        S[v] = [i for i in values[indices]]


    return S

def time_cov(df, just_post = True):
    """ Get covariance matrix for data across time periods, assuming all individuals have 
    the same time covariance matrix for each outcome variable. """

    #df = df.to_dataframe().reset_index()
    df = copy.deepcopy(df)
    df = df[df["group"] == "C"][['stream', 'unit', 'period', 'outcome', 'treat_time']]

    if just_post:
        df = df[df['treat_time'] == 1]

    # separate by stream: 
    dfs = {i:df[df['stream']==i] for i in df.stream.unique()}

    wide_time = {i:dfs[i].pivot(index=['unit', "stream"], columns='period', values='outcome') for i in dfs}
    covmats = {i:skcov.empirical_covariance(wide_time[i].to_numpy()) for i in wide_time}
    
    return covmats

def get_w_vector(windows, n_periods):
    """ Accepts 3-element tuple, returns array of binary vectors of the same length 
    as there are post-treatment times. 
    
    windows: 3-element tuple: (smallest window length, largest window length, step size)"""

    w0 = np.zeros(n_periods)

    wvecs = []

    if windows: 

        for i in range(windows[0], windows[1] + 1, windows[2]):
            for j in range(0, n_periods - i + 1):
                w = w0.copy()
                w[j:(i + j)] = 1
                wvecs.append(w)
    
    else: 
        wvecs.append(np.ones(n_periods))

    return(wvecs)

def quant_splitter(data, cov_names, suffix = None, keep_old_vars = True, nbins = None):

    if not suffix:
        suffix = "_binned"

    if not nbins: 
        nbins = [3 for i in cov_names]
    elif isinstance(nbins, int):
        nbins = [nbins for i in cov_names]

    data = copy.deepcopy(data)

    for i in range(len(cov_names)): 
        cn = cov_names[i]
        nb = nbins[i]
        data[cn+suffix] = pd.qcut(data[cn], q=nb, labels = [str(i) for i in np.arange(nb)])

    if not keep_old_vars: 
        data = data.drop(cov_names, axis = 1)
        # give the binned vars the old names: 
        data = data.rename({i + suffix: i for i in cov_names}, axis = 1)

    return data

def get_score(d): 
    scores = d[['priority', 'delta_hat']].copy()
    scores['score'] = 0.0
    
    for i in d['priority'].unique():

        p1 = d[d['priority'] <= i]['numer'].sum()
        p2 = d[d['priority'] <= i]['denom'].sum()

        d_hat = d.loc[d['priority']==i, 'delta_hat']

        # somehow, d_hat might occasionally be a list of length zero. This if statement handles this problem: (TODO: FIGURE OUT IF THIS SOLUTION is problematic)
        if np.isnan(d_hat.index.min()):
            scores.loc[scores['priority'] == i,'score'] = 0
        else: 
            d_hat = d_hat[d_hat.index.min()] # get just the number, and get just the first one if there is a draw (TODO: would it be better if we made it max?)
            scores.loc[scores['priority'] == i,'score'] = d_hat * p1 - d_hat * d_hat * p2 / 2

    return scores['score']

# get delta hats for each subset in the reduced set: 
def get_delta_hats(d):
    deltas = d[['priority']].copy()
    deltas['delta_hat'] = 0.0

    for i in d['priority'].unique():
        numer = d[d['priority'] <= i]['numer'].sum()
        denom = d[d['priority'] <= i]['denom'].sum()

        if denom == 0: 
            deltas.loc[deltas['priority'] == i,'delta_hat'] = 0
        else: 
            deltas.loc[deltas['priority'] == i,'delta_hat'] = numer / denom

    return deltas['delta_hat']


def edgesearch(df, modename, selected):
    """ Search one mode/edge of the tensor with ALTSS (one covariate, or the set of outcome variables) """

    # define a local, changeable version of selected: 
    sel_local = selected.copy()

    # filter out currently unselected values of other modes: 
    df_filt = df.copy()
    for mode, sel_vals in selected.items():
        if isinstance(sel_vals, str): # just in case we just have one value: 
            sel_vals = [sel_vals]
        if mode != modename:
            df_filt = df_filt[df_filt[mode].isin(sel_vals)]
    
    # sum across all other edges: 
    summed = df_filt.groupby(modename, observed = False)[['numer', 'denom']].sum().reset_index()

    # calculate priority scores
    summed['delta_max'] = 2 * summed["numer"] / summed["denom"]

    # rank: 
    summed['priority'] = summed['delta_max'].rank(ascending = False)
    summed = summed.sort_values(by = 'priority')
        
    sets = summed[['priority', modename, 'numer', 'denom']].copy()
    sets.loc[:,'delta_hat'] = get_delta_hats(summed)

    # get the score of each of the reduced subsets and select the greatest: 
    sets.loc[:,'score'] = get_score(sets)

    # define sel_local as all observations included in the locally optimizing set ####################

    # get lowest priority in S
    p_min = sets.loc[sets['score'] == sets['score'].max(), 'priority']
    p_min = p_min[p_min.index.min()] # get first in the set if there is a score draw 
    
    # get highest score: 
    hiscore = sets['score'].max()

    # get delta_hat associated with highest score: 
    delta_hat = sets.loc[sets['score'] == hiscore, 'delta_hat']
    delta_hat = delta_hat[delta_hat.index.min()] # get first in the set if there is a score draw 
    
    sel_local = sets.loc[sets['priority'] <= p_min, modename].unique()
    sel_local = [i for i in sel_local]

    return sel_local, hiscore, delta_hat

def covsearch(df, selected, niter = 5):

    # define a local, changeable version of selected: 
    sel_local = selected.copy()

    # get just the data for streams currently used 
    df_str = df[df["stream"].isin(selected['stream'])]

    for _ in range(niter):
        # iterate through covariate modes: 
        covs = selected # [i for i in selected if i != "stream"] (currently, search outcome as if it is another covariate, since we are not modeling correlations)
        for m in covs: 
            sel_local[m], score, delta_hat = edgesearch(df_str, m, sel_local)

    return sel_local, score, delta_hat

def search_given_w(df, search_vars, w, i_cov_mats = None, n_restarts = 10, n_iter = 5):

    # if cov_mats not provided, compute: 
    if i_cov_mats is None: 
        cov_mats = time_cov(df)
        i_cov_mats = {m:np.linalg.inv(cov_mats[m]) for m in cov_mats}


    # DEFINE FUNCTIONS CONDITIONAL ON W ###############################################
    def get_numer(g): 
        s = g.stream.unique()[0]
        dif2 = g.outcome - g.baseline
            
        return dif2.dot(i_cov_mats[s]).dot(w)

    def get_denom(g):
        s = g.stream.unique()[0]
        return w.dot(i_cov_mats[s]).dot(w)

    #  conditional on selected w, summarize each cell (outcome X cov profile) #############################################

    # numerator and denominator for summed elements of delta hat and delta max:
    nums = df.groupby(['unit', 'stream'])[['unit', 'stream', 'outcome', 'baseline']].apply(get_numer).reset_index()
    nums.columns = ['unit', 'stream', 'numer']
    dens = df.groupby(['unit', 'stream'])[['unit', 'stream', 'outcome', 'baseline']].apply(get_denom).reset_index()
    dens.columns = ['unit', 'stream', 'denom']

    # get dataframe with one-row per unit and join with numerator and denominator: 
    select = ["unit", "stream"]
    select.extend(search_vars)
    dfp_units = df[select].groupby(['stream', 'unit']).first().reset_index()
    dfp_units = dfp_units.merge(nums).merge(dens)

    results = {'score':[], 'selected':[], 'delta_hat':[]}
    
    for i in range(n_restarts):
        # make "selected" dictionary which defines the starting subset S 
        if i == 0: # if first iteration, make S all observations in the time window w:
            selected = {i:[j for j in df[i].unique()] for i in search_vars}
            selected["stream"] = [i for i in df.stream.unique()]
        else: # otherwise, random start: 
            selected = random_start(df, search_vars=search_vars)
        
        last_score = -99
        score = 0
        while score - last_score > 1e-4: # iterate until score converges
            last_score = score
            # conditional on the selected w and outcomes, search COVARIATES: 
            selected, score, delta_hat = covsearch(dfp_units, selected)

        results['score'].append(score)
        results['selected'].append(selected)
        results['delta_hat'].append(delta_hat)

    results = pd.DataFrame(results)
    results = results.loc[results.score == np.max(results.score)].iloc[0,] # get best subset from across all restarts.

    return w, results['selected'], results['score'], results['delta_hat']

def basic_base(data, model_vars, nbins, binned = False):
    # Simply gets the mean of control individuals in each cell, and matches the treatment group individuals with their cell mean. 

    if not binned: 
        data = quant_splitter(data, model_vars, keep_old_vars=False, nbins = nbins)

    # get control group for estimating baselines: 
    c_df = data.loc[(data['group']=="C"),]
    t_df = data.loc[(data['group']!="C"),]

    preds = c_df.groupby([i for i in np.append(model_vars, ["stream", "period"])], observed = False)["dif"].mean().reset_index().rename({"dif":"baseline"}, axis = 1)
    t_df = pd.merge(t_df, preds, on = model_vars + ["stream", "period"])
    return t_df.loc[:, ['stream', 'unit', 'period', 'baseline']] 

def rf_base(df, model_vars, nbins, binned): 
    """
    nbins and binned are used for nothing.
    """
    
    # get control group for estimating baselines: 
    c_df = df.loc[(df['group']=="C"),]

    # add stream and period to predictors:
    model_vars_plus = [i for i in np.append(model_vars, ["stream", "period"])]

    # get variables:
    label_encoder = LabelEncoder()
    x_cat = c_df[model_vars_plus].select_dtypes(include=['object']).apply(label_encoder.fit_transform).reset_index(drop=True)
    x_numeric = c_df[model_vars_plus].select_dtypes(exclude=['object']).values
    if (len(x_cat) > 0) & (len(x_numeric) > 0): 
        x = pd.concat([pd.DataFrame(x_numeric), x_cat], axis=1).values
    else: # there will always be two categorical variables because of stream and period
        x = x_cat.values

    y = c_df['dif'].values

    # initialize and run model: 
    regressor = RandomForestRegressor(n_estimators=30, oob_score=True)
    regressor.fit(x, y)

    # MSE:
    # np.mean((regressor.predict(x) - y)**2)

    # predictions for treatment group: 
    t_df = df.loc[(df['group']!="C"),]
    x_cat = t_df[model_vars_plus].select_dtypes(include=['object']).apply(label_encoder.fit_transform)
    x_numeric = t_df[model_vars_plus].select_dtypes(exclude=['object']).values
    if (len(x_cat) > 0) & (len(x_numeric) > 0): 
        x_treat = pd.concat([pd.DataFrame(x_numeric), pd.DataFrame(x_cat)], axis=1).values
    else: # there will always be two categorical variables because of stream and period
        x_treat = x_cat

    baselines = regressor.predict(x_treat)

    return baselines

def preprocess(data, baseline_func = basic_base, search_vars = None, model_vars = None, nbins = None, binned = False):
    # convert data to pandas dataframe: 
    df = data.to_dataframe().reset_index()
    df = df.sort_values(by = ['unit', 'stream', 'period'])

    if not model_vars: 
        model_vars = search_vars

    # get first differences for all individuals: 
    # TODO: change this so that it adapts to any selected reference periods
    premean = df[df.treat_time == 0].groupby(['unit', 'stream'])['outcome'].mean().reset_index().rename({"outcome":"premean"}, axis = 1)
    df = pd.merge(df, premean)
    df['dif'] = df['outcome'] - df['premean']

    # get covariance matrices: 
    cov_mats = time_cov(df)
    # cov_mats['0'] = np.eye(cov_mats['0'].shape[0]) # TODO: THIS IS A TEMPORARY TEST AND MUST BE REMOVED ASAP 

    # calculate inverse covariance matrices: 
    i_cov_mats = {m:np.linalg.inv(cov_mats[m]) for m in cov_mats}

    # make M, the baseline values based on the covariates: 
    baseline = baseline_func(df, model_vars, nbins = nbins, binned = binned)
    df = pd.merge(df, baseline, on = ['stream', 'unit', 'period']) # NOTE: merging here drops all control observations.

    # quantbin: 
    if not binned:
        df = quant_splitter(df, search_vars, keep_old_vars=False, nbins = nbins)

    # FOR NOW: RATHER THAN USE FIRST DIFFS, ADD TREATGROUP REF MEAN TO BASELINE: 
    df['baseline'] = df['baseline'] + df['premean']

    # just postXtreatment observations: 
    dfp = df[(df["treat_time"] == 1) & (df["group"] != "C")]

    return dfp, i_cov_mats


def get_score_given_set(data, S, search_vars = None, model_vars = None, baseline_func = basic_base, binned = False, return_n = False):
    
    # preprocess data: 
    if not search_vars: 
        search_vars = [i for i in S if i not in ['stream', 'period']]

    if not model_vars:
        model_vars = search_vars

    # infer w: 
    if 'period' in S:
        w = [i in S['period'] for i in data.attrs['treat.periods']] 
        w = np.array([int(1*i) for i in w])
    else: # if no period specified, assume all post-treatment periods are in 
        w = np.ones(len(data.attrs['treat.periods']))

    df, i_cov_mats = preprocess(data, baseline_func=baseline_func, search_vars=search_vars, model_vars=model_vars, binned=binned)

    # filter data according to S: 
    mask = pd.Series([True] * len(df))
    for key, value in S.items():
        mask = mask & df[key].isin(value).reset_index(drop = True)
    df = df[np.array(mask)]

    # get score and effect size ######################################3
    def get_numer(g): 
        s = g.stream.unique()[0]
        dif2 = g.outcome - g.baseline
            
        return dif2.dot(i_cov_mats[s]).dot(w)

    def get_denom(g):
        s = g.stream.unique()[0]
        return w.dot(i_cov_mats[s]).dot(w)

    #  conditional on selected w, summarize each cell (outcome X cov profile) #############################################

    # numerator and denominator for summed elements of delta hat and delta max:
    nums = df.groupby(['unit', 'stream'])[['unit', 'stream', 'outcome', 'baseline']].apply(get_numer).reset_index()
    nums.columns = ['unit', 'stream', 'numer']
    dens = df.groupby(['unit', 'stream'])[['unit', 'stream', 'outcome', 'baseline']].apply(get_denom).reset_index()
    dens.columns = ['unit', 'stream', 'denom']

    # get dataframe with one-row per unit and join with numerator and denominator: 
    select = ["unit", "stream"]
    select.extend(search_vars)
    dfp_units = df[select].groupby(['stream', 'unit']).first().reset_index()
    dfp_units = dfp_units.merge(nums).merge(dens)
    
    # calculate delta_hat: 
    delta_hat = np.divide(np.sum(dfp_units['numer']), np.sum(dfp_units['denom']))

    # calculate score: 
    score = delta_hat * np.sum(dfp_units['numer']) - delta_hat**2/2 * np.sum(dfp_units['denom'])

    if return_n: 
        return {'score': score, 'delta_hat': delta_hat, 'n': len(np.unique(df.unit))}
    else: 
        return {'score': score, 'delta_hat': delta_hat}



def cor_search(data, windowset = None, search_vars = None, model_vars = None, n_processes = 1, binned = False, nbins = None, baseline_func = basic_base, n_restarts = 10): # search_vars = ["variate" + str(i) for i in range(1,4)]

    # ARGUMENT DICTIONARY: 
    # windowset = (first, last, stepsize)
    # search_vars: what variables are used for search? 
    # model_vars: what variables are used for the model (outcome model as well as propensity model if we implement that). 
    #   If None, uses search_vars. 

    if not model_vars: 
        model_vars = search_vars.copy()

    # PREPROCESSING #############################################
    dfp, i_cov_mats = preprocess(data, baseline_func=baseline_func, search_vars=search_vars, model_vars=model_vars, binned=binned)

    # INITIALIZE #####################################################

    # initialize dataframe with one row per time period config, showing the best subset and its score
    scores_and_sets_by_w = {"w":[], "selected":[], "score":[], "delta_hat":[]}

    # iterate across windowset:
    if n_processes == 1: 
        for w in get_w_vector(windowset, len(dfp.period.unique())): # w = np.array([1,1,1,1,1])
            
            w, selected, score, delta_hat = search_given_w(dfp, search_vars, w, i_cov_mats=i_cov_mats, n_restarts=n_restarts)

            scores_and_sets_by_w['w'].append(w)
            scores_and_sets_by_w['selected'].append(selected)
            scores_and_sets_by_w['score'].append(score)
            scores_and_sets_by_w['delta_hat'].append(delta_hat)

    else: 
        ws = get_w_vector(windowset, len(dfp.period.unique()))
        parameters = list(product([dfp], [search_vars], ws, [i_cov_mats], [5]))

        with mp.Pool(processes=n_processes) as pool:
            results = pool.starmap(search_given_w, parameters)
        scores_and_sets_by_w = dict(zip(["w", "selected", "score", "delta_hat"], zip(*results)))

    # get highest scoring element in scores_and_sets: 
    top_w = np.array(scores_and_sets_by_w['w'][np.argmax(scores_and_sets_by_w['score'])])
    top_subset = scores_and_sets_by_w['selected'][np.argmax(scores_and_sets_by_w['score'])]
    top_score = np.max(scores_and_sets_by_w['score']) 
    top_delta = scores_and_sets_by_w['delta_hat'][np.argmax(scores_and_sets_by_w['score'])]

    # get name of selected time periods rather than just true and false: 
    all_post_treat_periods = [str(i) for i in np.sort([int(i) for i in dfp['period'].unique()])]
    top_w = [all_post_treat_periods[i] for i in range(len(all_post_treat_periods)) if top_w[i]==1]
    top_subset['period'] = top_w

    return({"score":top_score, "delta_hat":top_delta, "selected":top_subset})  


if __name__ == "__main__":
    
    ### get_counts_and_temporal_iterator inputs for testing ###
    # treatment = treatment_data; control = control_data
    # window = window; smallest_window_size= smallest_window_size
    # pre_time_cutoff = pre_time_cutoff; exclude_cutoff=exclude_cutoff
    # use_log=log_data
    sd = md.make_cov_matrix(n = 10)

    # Generate treatment and control data
    data, aff = md.make_dataset(1, 10000, 10, 1, -4, prop_affected=2/9, base_mean = 0.2, aff_mean = 0.4, base_sd = sd, distribution="multinorm", covariate = "3x3", noise_covariates=1)
    #data.vis_cov(cov_name = "variate1")
    search_vars = ["variate" + str(i) for i in range(1, 4)]

    cor_search(data, windowset = None, search_vars=search_vars) # windowset = (1, 5, 2)
    

    

    data = data.binvars(n_bins_per_dim = [3, 3, 3, 3, 3])
    data = data.first_diff()
    data = data.aggbinvars(varname = None, outcome = "outcome")

    data = {'Category1': ['A', 'A', 'B', 'B', 'A'],
        'Category2': ['X', 'Y', 'X', 'Y', 'X'],
        'Value': [10, 20, 30, 40, 50]}

    df = pd.DataFrame(data)

    # Group by multiple columns
    grouped = df.groupby(['Category1', 'Category2'])

    # Define function to apply to each group
    def my_function(group):
        # Access the index by name (Category1, Category2)
        category1_value = group.index.get_level_values('Category1')[0]  # Access value of 'Category1' in the group
        category2_value = group.index.get_level_values('Category2')[0]  # Access value of 'Category2' in the group
        print(f"Processing Group: Category1 = {category1_value}, Category2 = {category2_value}")
        # Do something with the group
        return group['Value'].sum()

    # Apply the function to the grouped DataFrame
    result = grouped.apply(my_function)
    print(result)
