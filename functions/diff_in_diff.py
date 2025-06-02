# This file contains functions for running difference in differences with XArrays

import numpy as np
import pandas as pd
from functions.make_data import make_dataset
from statsmodels import regression as reg

def apply_feffects(panel, axes=None):
    new_panel = panel.copy(deep=True) * 1.0
    if axes is None or 0 in axes:
        stream_feffects = panel.mean(dim = panel.dims[1:3])
        for stream in range(len(stream_feffects)):
            new_panel[stream, :, :] -= stream_feffects[stream]

    if axes is None or 1 in axes:
        unit_feffects = panel.mean(dim = [panel.dims[0], panel.dims[2]])
        for unit in range(len(unit_feffects)):
            new_panel[:, unit, :] -= unit_feffects[unit]

    if axes is None or 2 in axes:
        time_feffects = panel.mean(dim = panel.dims[0:2])
        for time in range(len(time_feffects)):
            new_panel[:, :, time] -= time_feffects[time]

    return new_panel

def get_ix(xr_dset, group, dim = None):
    """
    Takes xarray dataset and returns indices for the requested group.
    group is "A" for affected, "T" for treatment (including affected)
        and "C" for control.
    """

    if group == "A":
        ix = np.where(xr_dset["group"] == "A") # get affected group indices
    if group == "T":
        ix = np.where((xr_dset["group"] == "T") | (xr_dset["group"] == "A")) # get treatment group indices
    if group == "C":
        ix = np.where(xr_dset["group"] == "C") # get control group indices

    # make into named dictionary: 
    ix = {name: ix[dim] for dim, name in enumerate(xr_dset.dims)}

    # double check that all of the dims are present; one will disappear if it only takes one value: 
    if dim == None:
        ix = ([ix[d] for d in xr_dset.dims])
    else:
        ix = ix[dim]

    return ix



def diff_in_diff(panel_data, treatment_ixs = None, control_ixs = None, treatment_time = None, 
                 reference_time = None, model_type = "standard", exclude_cutoff = False):
    
    if treatment_ixs is None: 
        treatment_ixs = get_ix(panel_data, group = "T", dim = "unit")

    if control_ixs is None:
        control_ixs = get_ix(panel_data, group = "C", dim = "unit")
        #control_ixs = [i for i in range(panel_data.shape[1]) if i not in treatment_ixs]

    if treatment_time is None:
        ttimes_int = [int(i) for i in panel_data.attrs['treat.periods']]
        treatment_time = min(ttimes_int)

    if reference_time is None:
        reference_time = [int(i) for i in panel_data.attrs['all.periods'] if i not in panel_data.attrs['treat.periods']]

    if model_type == "standard":
        
        panel_data = panel_data.isel(unit = np.append(treatment_ixs, control_ixs))

        data = panel_data.to_dataframe()
        data = data.reset_index(level=list(panel_data.dims))
        meanstream = data.groupby(["unit", "period"]).mean("outcome")["outcome"]

        data = pd.merge(data.reset_index()[["unit", "group", "period"]].drop_duplicates(), meanstream.reset_index(level = ["unit", "period"]), on = ["unit", "period"])
        data["treatgroup"] = np.array([0 if i == "C" else 1 for i in data["group"]])
        data["post"] = data["period"].astype(int) >= treatment_time
        data["treatXpost"] = np.array([t * p for t, p in zip(data["treatgroup"], data["post"])])

        X = np.array(data[["treatgroup", "post", "treatXpost"]])
        y = np.array(data["outcome"])

        lm = reg.linear_model.OLS(endog = y, exog = X)
        results = lm.fit()
        
        te = results.params[2]
        p = results.pvalues[2]

        return te, p


    elif model_type == "twfe":

        panel_data = panel_data.isel(unit = np.append(treatment_ixs, control_ixs))

        data = panel_data.to_dataframe()
        data = data.reset_index(level=list(panel_data.dims))
        meanstream = data.groupby(["unit", "period"]).mean("outcome")["outcome"]

        data = pd.merge(data.reset_index()[["unit", "group", "period"]].drop_duplicates(), meanstream.reset_index(level = ["unit", "period"]), on = ["unit", "period"])
        data["treatgroup"] = np.array([0 if i == "C" else 1 for i in data["group"]])
        data["post"] = (data["period"].astype(int) >= treatment_time).astype(int)
        data["treatXpost"] = np.array([t * p for t, p in zip(data["treatgroup"], data["post"])])

        # get unit and period means for twfe: 
        unit_means = data.groupby("unit")[["outcome", "treatXpost"]].mean().reset_index().rename({"outcome":"yunitmean", "treatXpost":"Dunitmean"}, axis = 1)
        period_means = data.groupby("period")[["outcome", "treatXpost"]].mean().reset_index().rename({"outcome":"yperiodmean", "treatXpost":"Dperiodmean"}, axis = 1)
        grand_means = data[["outcome", "treatXpost"]].mean()
        data = data.merge(unit_means, on = ["unit"]).merge(period_means, on = ["period"])
        data["ygrand"] = grand_means["outcome"]
        data["dgrand"] = grand_means["treatXpost"]

        # demean to get fixed effects: 
        X = np.array([np.ones(data['treatXpost'].shape), data["treatXpost"] - data["Dunitmean"] - data["Dperiodmean"] + data["dgrand"]]).T
        y = np.array(data["outcome"] - data["yunitmean"] - data["yperiodmean"] + data["ygrand"])

        lm = reg.linear_model.OLS(endog = y, exog = X)
        results = lm.fit()

        groups = data['unit']        
        clustered_se = results.get_robustcov_results(cov_type = "cluster", groups = groups)

        te = clustered_se.params[1]
        se = clustered_se.bse[1]
        p = clustered_se.pvalues[1]

        return te, se, p


        print("Two way fixed effects not yet implemented.")

    else:
        # use the simple dnd estimator (post-treatment-mean - pre-treatment-mean) - (post-control-mean - pre-control-mean)
        treatment_group = panel_data[:, treatment_ixs, :]
        control_group = panel_data[:, control_ixs, :]

        # get arrays of times
        all_treat_periods = np.array([int(i) for i in treatment_group["period"].values])
        all_contr_periods = np.array([int(i) for i in control_group["period"].values])

        # get pre and post treatment indices (as strings to pass into xarray period dim indices)
        pre_treat_ixs = [str(p) for p in all_treat_periods if p < treatment_time]
        post_treat_ixs = [str(p) for p in all_treat_periods if (p > treatment_time & exclude_cutoff) or (p >= treatment_time & ~exclude_cutoff)]

        # use pre and post indices to get pre and post slices 
        treat_pre = treatment_group.sel(period = pre_treat_ixs)
        treat_post = treatment_group.sel(period = post_treat_ixs)

        pre_contr_ixs = [str(p) for p in all_contr_periods if p < treatment_time]
        post_contr_ixs = [str(p) for p in all_contr_periods if (p > treatment_time & exclude_cutoff) or (p >= treatment_time & ~exclude_cutoff)]

        #control_pre_data = control[:,:, pre_contr_ixs]
        #control_post_data = control[:,:, post_contr_ixs]
        contr_pre = control_group.sel(period = pre_contr_ixs)
        contr_post = control_group.sel(period = post_contr_ixs)

        te = (treat_post.mean() - treat_pre.mean()) - (contr_post.mean() - contr_pre.mean())

        return te
    
if __name__ == "__main__":

    df, aff = make_dataset(10, 500, 10, 1, -4, prop_affected=0.25, base_mean = "0.5*t", treat_mean = "2 + 0.5*t", aff_mean="3 + 0.5*t")

    treat_ix = get_ix(df, "T", dim = 1)
    contr_ix = get_ix(df, "C", dim = 1)
    tru_aff = get_ix(df, "A", dim = 1)

    diff_in_diff(df["outcome"], treat_ix, contr_ix)

    a = diff_in_diff(df)
    a[0]


    # Brute force debug: 

    df, aff = make_dataset(2, 20, 2, prop_affected=0.4, aff_mean = 1)

    treatment_ixs = np.arange(1)
    control_ixs = np.arange(10, 20)

    # Test on data for 3x3 with 3 noise variables:
    prop_i = 1
    aff_i = 1
    data, _ = make_dataset(3, n_units = 13500, n_times = 2, prop_affected = prop_i, treat_mean = 0,
                                treat_time = 1, aff_mean = aff_i, base_sd = 3, covariate = "3x3", noise_covariates= 3,
                                aff_streams = [True, True, False])

    data_1stream = data.sel({"stream":'1'}) # select just a subset of treatments 

    diff_in_diff(data_1stream)
  