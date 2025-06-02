import numpy as np
import pandas as pd
import functions.make_data as md
import functions.rand_test as rt
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import xarray as xr
import functions.diff_in_diff as did
import functions.did_ss_corr as ds

def did_search_n_split(agged_set, unagged_set, search_vars, splits = 20, test_side = 1, random_state = None):
    
    if random_state: 
        kf = KFold(n_splits=splits, shuffle=True, random_state=random_state)
    else: 
        kf = KFold(n_splits=splits, shuffle=True)
    
    # get just the outcome variables in agged_set
    unagged_set = unagged_set.sel({'stream':[i for i in agged_set.stream.to_numpy()]})

    # set aside array for estimated treatment effects:
    effects = []
    ses = []
    ps = []
    subsets = []

    for search_ix, te_ix in kf.split(unagged_set.unit.to_numpy()): # search_ix, te_ix = next(kf.split(unagged_set.unit.to_numpy()))

        # use ix to subset on units:
        search_set = unagged_set.sel({'unit':unagged_set["unit"][search_ix]})
        te_set = unagged_set.sel({'unit':unagged_set["unit"][te_ix]})

        result = ds.cor_search(data = search_set, search_vars=search_vars, binned = True)

        # extract score and discovered set: 
        score = result['score']
        subset = result['selected']

        # get just selected set from hold out data: 
        te_subset = md.subset_by_covs(xr_data=te_set, subset = subset)
        
        te, se, p = did.diff_in_diff(te_subset, model_type = 'twfe')

        effects.append(te)
        ses.append(se)
        ps.append(p)
        subsets.append(subset)

    # return {'effects':effects, 'ses':ses, 'ps':ps, 'subsets':pd.DataFrame(list(subsets))}
    return pd.DataFrame({'effects':effects, 'ses':ses, 'ps':ps}).join(pd.DataFrame(list(subsets)))        
    

if __name__ == "__main__":

    # LOAD DATA AND PREPROCESS #################################################################
    data = pd.read_csv("data/wando_cleaned.csv", index_col=0)
    x_names = ['employed_individuals', 'medianhouseholdincome', 'branch_count_percapita', 'internet_access_score']

    # log the outcomes of interest that are not already logged: 
    data.loc[data["stream"] == "bankruptcy_per_capita", "outcome"] = np.log(data[data["stream"] == "bankruptcy_per_capita"]["outcome"] + 1)
    data.loc[data["stream"] == "business_bankruptcy_percapita", "outcome"] = np.log(data[data["stream"] == "business_bankruptcy_percapita"]["outcome"] + 1)
    data.loc[data["stream"] == "nonbusiness_bankruptcy_percapita", "outcome"] = np.log(data[data["stream"] == "nonbusiness_bankruptcy_percapita"]["outcome"] + 1)
    data.loc[data["stream"] == "bankruptcy_percapita_lowliab", "outcome"] = np.log(data[data["stream"] == "bankruptcy_percapita_lowliab"]["outcome"] + 1)
    data.loc[data["stream"] == "bankruptcy_percapita_highliab", "outcome"] = np.log(data[data["stream"] == "bankruptcy_percapita_highliab"]["outcome"] + 1)
    data.loc[data["stream"] == "bankruptcy_percapita_highassets", "outcome"] = np.log(data[data["stream"] == "bankruptcy_percapita_highassets"]["outcome"] + 1)
    data.loc[data["stream"] == "bankruptcy_percapita_lowassets", "outcome"] = np.log(data[data["stream"] == "bankruptcy_percapita_lowassets"]["outcome"] + 1)

    # get an unaggregated version for diff in diff: 
    xr_data = md.df_to_dndset(data, diff=False, agg = False, searchvars = ['employed_individuals', 'medianhouseholdincome', 'internet_access_score', 'branch_count_percapita'])

    xr_data['internet_access_score'] = xr_data.internet_access_score.astype(str)





    # ANALYSIS ###################################################################################


    # overall bank ate
    S = {'stream': ['bankruptcy_per_capita']}
    subset = md.subset_by_covs(xr_data, subset = S) # get subset, which is all counties and all times, but just the overall bankruptcy outcome.
    did.diff_in_diff(subset, model_type = 'twfe')


    ### overall bankruptcy didscan (TABLE 1 ROW 1) ###
    ovrall_bank = xr_data.sel({'stream':'bankruptcy_per_capita'}).expand_dims('stream') 
    neworder = ["stream"]
    neworder.extend(list(ovrall_bank.dims)[:-1])
    neworder.extend([i for i in list(ovrall_bank.variables) if i not in list(ovrall_bank.dims)])
    ovrall_bank = ovrall_bank[neworder] # reorder dims so stream is first

    # get best subset and its score: 
    result = ds.cor_search(data = ovrall_bank, search_vars = x_names, binned = True)

    # estimate treatment effect at the best set: 
    te_output = did_search_n_split(ovrall_bank, xr_data, search_vars = x_names, splits = 5)
    te_output[['employed_individuals','medianhouseholdincome','branch_count_percapita','internet_access_score','effects']]

    # get null distribution using permutation test and get one-sided p value: 
    rtout_overall = rt.rand_test(result, ovrall_bank, n_samples =500, search_covs = x_names, n_processes = 7)
    np.mean(result['score'] < rtout_overall)
    # optionally save randomization distribution: 
    # pd.DataFrame({"null_scores" : rtout}).to_csv("nullscores.csv")



    ### bankruptcy by liability (TABLE 1 ROW 2) ### 
    liab_data = xr_data.sel({'stream':['bankruptcy_percapita_highliab', 'bankruptcy_percapita_lowliab']})

    # run didscan on liability outcomes:
    result = ds.cor_search(data = liab_data, search_vars = x_names, binned = True)

    # get null distribution using permutation test and get one-sided p value: 
    rtout_liab = rt.rand_test(result, liab_data, n_samples = 1000, search_covs = x_names, n_processes = 7, print_count = True)
    np.mean(result['score'] < rtout_liab)


    ### bankruptcy by asset (TABLE 2 ROW 3) ###
    asset_data = xr_data.sel({'stream':['bankruptcy_percapita_highassets', 'bankruptcy_percapita_lowassets']})

    # run didscan on liability outcomes:
    result = ds.cor_search(data = asset_data, search_vars = x_names, binned = True)

    # get null distribution using permutation test and get one-sided p value: 
    rtout_asset = rt.rand_test(result, liab_data, n_samples = 1000, search_covs = x_names, n_processes = 7, print_count = True)
    np.mean(result['score'] < rtout_asset)



    # TABLE 3: evaluate nearby subsets to the subset discovered for overall bankruptcy rate 
    S = {'employed_individuals': ['1', '2'], 'medianhouseholdincome': ['2'], 
                'internet_access_score': ['2'], 'branch_count_percapita': ['1']}
    ds.get_score_given_set(ovrall_bank, S, search_vars = x_names, binned = True)


    S = {'employed_individuals': ['1', '2'], 'medianhouseholdincome': ['2'], 
                'internet_access_score': ['1', '2'], 'branch_count_percapita': ['1','2']}
    ds.get_score_given_set(ovrall_bank, S, search_vars = x_names, binned = True)


    S = {'employed_individuals': ['1', '2'], 'medianhouseholdincome': ['1', '2'], 
                'internet_access_score': ['1', '2'], 'branch_count_percapita': ['2']}
    ds.get_score_given_set(ovrall_bank, S, search_vars = x_names, binned = True)


    S = {'employed_individuals': ['1', '2'], 'medianhouseholdincome': ['1', '2'], 
                'internet_access_score': ['1', '2'], 'branch_count_percapita': ['1']}
    ds.get_score_given_set(ovrall_bank, S, search_vars = x_names, binned = True)


