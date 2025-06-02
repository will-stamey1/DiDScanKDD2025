import sys
sys.path.append() # SPECIFY LOCATION OF THE FOLDER CONTAINING THE REPO
import numpy as np
import pandas as pd
import xarray as xr
import functions.make_data as md
from statsmodels import regression as reg
from itertools import product
from functions.generate_wando_data import wando_gen
import functions.did_ss_corr as ds
from concurrent.futures import ProcessPoolExecutor, as_completed


def overlap(selset, true_units): #true_region = [[0,2/3],[0,1/3],[0,1],[0,1]]):

    selset = np.unique(selset)
    true_units = np.unique(true_units)

    numer = len([i for i in selset if i in true_units])
    denom = len(selset) + len(true_units) - numer
    overlap = numer / denom

    return overlap

def figure_3_sim(i, n_iter):
    size = [0.2, 0.4, 0.6, 0.8, 1.0]
    size = [s / 10 for s in size]
    streams = [1,2,3]
    
    # add configurations where all cells are affected: 
    n_aff_covs = [0]
    reg_pct = [[3,3]]
    configs_1 = list(product(size, n_aff_covs, reg_pct, streams))
    
    # add configs for effect size range 1.2 to 2: 
    n_aff_covs = [1]
    reg_pct = [[1,1], [2,2]]
    configs_2 = list(product(size, n_aff_covs, reg_pct, streams))
    configs_1.extend(configs_2)

    n_aff_covs = [2]
    reg_pct = [[1,1], [1,2], [2,2]]
    configs_3 = list(product(size, n_aff_covs, reg_pct, streams))
    configs_1.extend(configs_3)

    configs = pd.DataFrame(configs_1, columns=["eff_size", "n_aff_covs", "reg_pct", "streams"]) 

    c = configs.iloc[i,:]

    results = {"score":[], "overlap":[], "pval":[], "rej_at_05":[], "eff_size":[], "n_aff_covs":[], "reg_pct":[], "streams":[]}

    for it in range(n_iter):

        results["eff_size"].append(c["eff_size"]) # get effect size
        results["n_aff_covs"].append(c["n_aff_covs"]) # get number of affected covs: [1, 2]
        results["reg_pct"].append(c["reg_pct"]) # get shape of affected subregion [1 is 1/3, 2 is 2/3]. If aff_covs == 1, the second value of reg_pct is ignored. 
        results["streams"].append(c["streams"]) # get number of affected streams 

        gml_vars = ["employed_individuals", "medianhouseholdincome", "branch_count_percapita", "internet_access_score"]
        searchvars = [w + "_bin" for w in gml_vars]

        df, true_set = wando_gen(c)

        xr_data = md.df_to_dndset(df, bin=False, agg=False, searchvars = searchvars)
        # Run didscan: 
        didss_res = ds.cor_search(xr_data, search_vars = searchvars, binned = True)

        if c.iloc[3] == 1:
            true_set["stream"] = ["1"] 
        elif c.iloc[3] == 2:
            true_set["stream"] = ["1", "2"]
        
        # get OVERLAP: 
        selected_obs = np.array(md.subset_by_covs(xr_data, didss_res['selected']).unit) # get selected subset
        true_aff_obs = np.array(md.subset_by_covs(xr_data, true_set).unit)
        did_ol = overlap(selected_obs, true_aff_obs) #, n_bins = 3, n_bins_true=3)
        
        results["score"].append(didss_res['score'])
        results["overlap"].append(did_ol)

        # use randomization test to check for significance: 
        null_data = pd.read_csv("data/null_dist.csv")
        results['pval'].append(np.mean(didss_res['score'] > null_data['didss_score']))
        results['rej_at_05'].append(results['pval'][it] < 0.05)

        print(it)
    res_df = pd.DataFrame(results)
    return res_df


def main():
    # FOR PARALLEL RUNS: 
    params = [(i, 1000) for i in range(0, 90)] # 1000 is the number of replications, 90 is the number of configurations which are handled in parallel.

    with ProcessPoolExecutor(max_workers = 6) as executor:
        futures = [executor.submit(figure_3_sim, ij, n_reps) for ij, n_reps in params]
        results = [f.result() for f in as_completed(futures)]

    # Concatenate all resulting DataFrames
    combined_df = pd.concat(results, ignore_index=True)
    combined_df.to_csv("outcome_search.csv")




if __name__ == "__main__":

    main()