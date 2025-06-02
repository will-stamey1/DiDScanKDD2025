import sys
sys.path.append() # SPECIFY LOCATION OF THE FOLDER CONTAINING THE REPO

import numpy as np
import pandas as pd
import functions.genericml as gml
import functions.make_data as md
from itertools import product
from functions.generate_wando_data import wando_gen
import functions.bh as bh
import functions.did_ss_corr as dscorr
import econml.grf as grf

# NOTE: The simulations below are nested in two for loops: the outer one iterates through the experiment 
# configurations (effect size, size of affected region), while the inner loop iterates through the 
# repititions of a given configuration. This is time-consuming. When we ran these simulations for the 
# paper, we split the outer loop iterations into separate jobs submitted to our university's computing 
# center. We recommend this or parallelization to anyone intent on replicating the full set of sims.

def overlap(selset, true_units): #true_region = [[0,2/3],[0,1/3],[0,1],[0,1]]):

    selset = np.unique(selset)
    true_units = np.unique(true_units)

    numer = len([i for i in selset if i in true_units])
    denom = len(selset) + len(true_units) - numer
    overlap = numer / denom

    return overlap
    

if __name__ == "__main__":

    i_per_config = 1000

    shape = [1/9, 2/9, 4/9]
    size = np.linspace(0, 1, 11)

    ol_results = {"genml_10":[], "genml_20":[], "genml_30":[], "didss":[], "bh":[], "eff_size":[], "n_aff_covs":[], "reg_pct":[], "streams":[]}

    # make the list of configurations
    size = np.arange(1,11) / 20
    size = [i for i in size]
    streams = [1,2]
    n_aff_covs = [1]
    reg_pct = [[1,1], [2,2]]
    configs_1 = list(product(size, n_aff_covs, reg_pct, streams))

    n_aff_covs = [2]
    reg_pct = [[1,1], [1,2], [2,1], [2,2]]
    configs_2 = list(product(size, n_aff_covs, reg_pct, streams))

    configs_1.extend(configs_2)
    configs = pd.DataFrame(configs_1, columns = ["eff_size", "n_aff_covs", "reg_pct", "streams"])

    # drop configs where only one stream is affected (we're just using the first stream, so this variation is redundant):
    configs = configs[configs.streams!=1]

    # for i in range(len(configs)):

    for j in range(len(configs)): # iterate through experiment configurations

        c = configs.iloc[j,:] # set configuration

        for t in range(i_per_config): # iterate through repetitions for configuration c

            # define the modeling and search variables: 
            gml_vars = ["employed_individuals", "medianhouseholdincome", "branch_count_percapita", "internet_access_score"]
            searchvars = [w + "_bin" for w in gml_vars]

            # generate new dataset from W and O's data distribution: 
            df, true_set = wando_gen(c)

            # drop second stream and all post-treatment periods but the first 
            df = df[df.stream=='1']
            df = df[df.period.isin(range(1,6))]
            xr_data = md.df_to_dndset(df, bin=False, agg=False, searchvars = searchvars) # convert to dndset

            # Run genml #############################################################################

            df_gml = xr_data.to_dataframe().reset_index()

            # get mean difference over time across post-periods for each unit
            meandifs = df_gml[df_gml['period'].isin(xr_data.attrs["treat.periods"])].groupby("unit", dropna = True)["dif"].mean().reset_index()
            df_gml = df_gml[df_gml.period == "5"].drop("dif", axis = 1).merge(meandifs) 
            df_gml['group'] = df_gml['group'].replace({"A": '1', "T": '1', "C": '0'}).astype(int) # rename group values to be suitable for cf learner

            # initialize gml model object:
            gl_mod = gml.genml()
            gl_mod.fit(df_gml, xnames = gml_vars, zname = 'group', splits = 20)

            # get 30% largest treatment effects:
            selset_30 = gl_mod.data.nlargest(int(np.round(gl_mod.data.shape[0] * 0.095, 0)), "mean_pred")['unit']

            # double check with vis: 
            #gl_mod.vis_effect("medianhouseholdincome")

            ol_results["genml_30"].append(overlap(selset_30, list(df_gml[df_gml.aff & (df_gml.group==1)]["unit"])))


            # get 20% largest treatment effects:
            selset_20 = gl_mod.data.nlargest(int(np.round(gl_mod.data.shape[0] * 0.2, 0)), "mean_pred")['unit']
            ol_results["genml_20"].append(overlap(selset_20, list(df_gml[df_gml.aff & (df_gml.group==1)]["unit"])))

            # get 10% largest treatment effects: 
            selset_10 = gl_mod.data.nlargest(int(np.round(gl_mod.data.shape[0] * 0.1, 0)), "mean_pred")['unit']
            ol_results["genml_10"].append(overlap(selset_10, list(df_gml[df_gml.aff & (df_gml.group==1)]["unit"])))

            # DID-Scan #######################################################################

            xr_data.attrs["binned"]=True
            #xr_data = xr_data.aggbinvars(vars2aggby=searchvars)
            didss_res = dscorr.cor_search(data = xr_data, search_vars=gml_vars, binned = False) # get results of DiD-Scan search
            didss_sel = didss_res['selected'] # get selected region

            # get observations in S and assess overlap: 
            sel_subset = df.copy().drop(gml_vars, axis = 1)
            sel_subset.period = [str(i) for i in sel_subset.period] # make period a string
            for column in sel_subset.columns: 
                if column in searchvars:
                    sel_subset = sel_subset.rename(columns = {column: column.replace("_bin", "", 1)})

            for key, allowed_values in didss_sel.items():
                sel_subset = sel_subset[sel_subset[key].isin(allowed_values)]
            sel_subset = list(sel_subset.loc[sel_subset.group=='T'].unit) # get treatment group units falling in the selected range

            did_ol = overlap(sel_subset, list(df[df.aff & (df.group=='T')]["unit"])) #, n_bins = 3, n_bins_true=3)
            ol_results["didss"].append(did_ol)


            # Run BH-HTE ###################################################################################
            
            df.loc[df.aff, 'group'] = 'A'
            xr_data = md.df_to_dndset(df, bin=False, agg=False, searchvars = searchvars) # get a fresh, unaggregated version of the xarray dataset
            bh_ol = bh.hte_bh(xr_data, search_vars = gml_vars, search_times=False)

            ol_results["bh"].append(bh_ol)

            # Run causal forest #############################################################################

            # convert back to dataframe:
            df = xr_data.to_dataframe().reset_index()

            # # get mean difference over time across post-periods for each unit
            meandifs = df[df['period'].isin(xr_data.attrs["treat.periods"])].groupby("unit", dropna = True)["dif"].mean().reset_index()
            df = df[df.period == "5"].drop("dif", axis = 1).merge(meandifs) 
            df['group'] = df['group'].replace({"A": '1', "T": '1', "C": '0'}).astype(int) # rename group values to be suitable for cf learner

            # define variables: 
            x = df[gml_vars]

            # # initialize causal forest model object:
            cfmod = grf.CausalForest(n_estimators=12, max_depth = 3)
            cfmod = cfmod.fit(X = x, T = df['ever_treat'], y = meandifs['dif'])

            # predictions and confidence intervals: 
            tgroup_x = df.loc[df['group'] == 1, gml_vars]
            ate_pred = cfmod.predict(tgroup_x)
            ci_pred = cfmod.predict_interval(tgroup_x)

            # check for each observation if their lower ci bound > 0: 
            treat_group = df.copy()
            treat_group = treat_group.loc[df.group == 1,:]
            treat_group['cf_selected'] = (ci_pred[0] > 0)
            # np.mean(ci_pred[0] > 0)

            # calculate overlap: 
            cf_ol = overlap(list(treat_group[treat_group.cf_selected]['unit']), list(df[df.aff & (df.group==1)]["unit"]))
            ol_results["cf"].append(cf_ol)


            # save experiment parameters:  
            ol_results["eff_size"].append(c.eff_size)
            ol_results["n_aff_covs"].append(c.n_aff_covs)
            ol_results["reg_pct"].append(c.reg_pct)
            ol_results["streams"].append(c.streams)

            print(str(t))

        oldf = pd.DataFrame(ol_results)

        oldf.to_csv("simul/sim_scripts/KDD_comparisons/sims_2-8-25/compare_" + str(j) + ".csv")
