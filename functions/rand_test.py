# Evaluate results of did_ss with a randomization test
import functions.make_data as md
import functions.did_ss_corr as ds # TODO: properly integrate with didsscorr
import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt
import xarray as xr
import multiprocessing as mp
from functools import partial

def sample_p(pars, distribution):
    if distribution=="binomial":
        a = pars[0] * pars[1]
        b = (1 - pars[0]) * pars[1]

        # shift a and b away from zero slightly if equal to 0: 
        a = a.where(a != 0, 0.001)
        b = b.where(b != 0, 0.001)

        return np.random.beta(a, b)

    if distribution == "gaussian":
        # TODO: i can't remember what's in pars for gaussian, double check this: 
        mu = pars[0]
        sig = pars[1]

        return np.random.normal(mu, sig)

def gen_sums(pars, coords, distribution):
    # generate a sums array from the parameters, the coordinates, and the distribution. 
    if distribution == "gaussian":
        means, sd_arr = pars
        # rand_means = this doesnt work yet, need to pull in sample sizes in pars. 
        sums = xr.DataArray(np.random.normal(means, sd_arr), coords = coords)
    elif distribution == "binomial":
        p, n = pars
        rand_p = sample_p(pars, distribution) # generate a random proportion from the beta distribution of proportions
        sums = xr.DataArray(np.random.binomial(n, rand_p), coords = coords)

    return sums


def one_sim(pars, post_pan, distribution, test_side, n_search_iter, cov_search_iter):
    # generate new dataset with 'means' as means and using estimated stream variances:
    newsums = gen_sums(pars, post_pan.coords, distribution=distribution)
    newdat = post_pan.copy()
    newdat['sums_treatment'] = newsums

    # run ltss:
    if distribution == "gaussian":
        sim_out = ds.ltss(newdat, ltss_func = ds.multivariate_subset_aggregation_ltss, test_side = test_side) 
    elif distribution == "binomial":
        sim_out = ds.altss(newdat, test_side=test_side, num_iterations=n_search_iter, cov_search_iter=cov_search_iter)

    result = list(list(sim_out.items())[0][1].items())[0][1][0]
    if isinstance(result, int):
        return result
    else:
        return result.item()
    
def one_sim_shuffle(panel, shuffle_on, covs, test_side, pre_bin, n_restarts = 10, loud = False, iteration_id = 0):
        
    """
    loud: If True, prints 'iteration_id' once run. """

    # convert to dataframe: 
    panel_df = panel.to_dataframe().reset_index()
    shuf_lvl_units = panel_df[[shuffle_on, "group"]].drop_duplicates()
    shuf_lvl_units.group = np.random.permutation(shuf_lvl_units["group"].values)
    shuffled = panel_df.copy().drop(["group"], axis = 1)
    shuffled = shuffled.merge(shuf_lvl_units, how = "left", on = shuffle_on)

    # turn retreated_data into a dndset
    indices = ["stream", "unit", "period"]
    if panel.staggered: 
        indices.append("treat_period")
    xr_retreat = shuffled.set_index(indices).to_xarray()
    xr_retreat = md.dndset(xr_retreat)
    xr_retreat.attrs = panel.attrs
    xr_retreat.attrs['agged'] = False
    xr_retreat.attrs["binned"] = pre_bin
    xr_retreat.attrs["diffed"] = False

    # xr_retreat = xr_retreat.sel({"stream": [i for i in np.unique(panel.stream.to_numpy())]}) # haven't figured out what this is for - this command shouldn't do anything

    # analyze dataset: 
    sim_score = ds.cor_search(xr_retreat, search_vars = covs, binned = pre_bin, n_restarts = n_restarts)   

    if loud:
        print(iteration_id)

    return sim_score['score']        


def rand_test(result, panel, n_samples = 1000, use_control_only = True, 
              test_side = 1, search_covs = None, distribution = "gaussian", show_plot = False, 
              n_processes = None, shuffle_on = None, print_count = False,
              n_restarts = 10, cov_search_iter = 4, binned = None):
    
    """
    Purpose: Runs randomization tests of the experiment results to determine whether the 
             subgroup found is statistically significant. 
    
    result: pass the full ltss results output from analyzing the real data. 
    panel: pass the data panel (xarray). 
    n_samples: number of datasets to generate
    rt_type: "generate" or "rearrange. Indicates whether old data is reassigned to 
             treat/control or if new data is generated.  
    use_control_only: (default, False unimplemented!) if True, uses only control data to 
    estimate the mean about which the randomized data sets is generated.  
    shuffle_on: What variable to shuffle on for rearrange style randomization tests? This 
                is usually the hierarchical level of treatment randomization.  
    """

    if shuffle_on is None: # make unit the default name of the variable to shuffle on:   
        shuffle_on = "unit"

    if not binned: # if binned is "none", check from the data
        binned = panel.attrs['binned']

    # Get the score from the analyzed data:
    real_score = result['score']

    # store the values of the top ltss score for each simulated dataset.
    score_sample_dist = []

    # get all combinations of covariates as bins to iterate through (and include the dimension of time treated if staggered):
    iterbins = search_covs.copy()
    if panel.staggered: 
        iterbins.append("treat_period")
        
    if n_processes is None or n_processes == 1:

        for i in range(n_samples):
            
            # get simulation result for a single random dataset: 
            sim_score = one_sim_shuffle(panel, shuffle_on, search_covs, test_side, binned, n_restarts=n_restarts, loud = True, iteration_id=i)
                            
            # add the score to the list: 
            score_sample_dist.append(sim_score)

    else: 
        inputs = [(panel, shuffle_on, search_covs, test_side, binned, n_restarts, True, i) for i in range(n_samples)] # copy input tuple n_samples times.
        parfunc = partial(one_sim_shuffle)
        with mp.Pool(processes = n_processes) as pool:
            score_sample_dist = pool.starmap(parfunc, inputs)

    score_sample_copy = [i for i in score_sample_dist]
    score_sample_copy = np.array(score_sample_copy)
    score_sample_copy = np.append(score_sample_copy, real_score)
    score_sample_copy = np.sort(score_sample_copy)
    score_sample_dist = np.array(score_sample_dist)
    score_sample_dist = np.sort(score_sample_dist)

    # Get quantile of the real data result: 
    qtil = np.mean([index for index, element in enumerate(score_sample_copy) if element == real_score])
    qtil = qtil/n_samples

    if(show_plot):
        # Plot histogram of generated data along with 95 percentiles and our real score: 
        # Plot histogram using Seaborn
        sns.histplot(score_sample_dist, bins='auto', kde=False)
        plt.xlabel('F-Score of Best Cluster')
        #plt.ylabel('Frequency')
        plt.title('Randomization Study Result')
        # Draw line at true score
        y_min, y_max = plt.gca().get_ylim()
        x_min, x_max = plt.gca().get_xlim()
        x_range = x_max - x_min
        plt.axvline(real_score, color='red', linestyle='--', linewidth=1)
        plt.text(real_score-0.05*x_range, 0.9*y_max, 'data', ha='center')
        # Draw line at 95%
        ninety5pct = score_sample_dist[(round(0.95 * n_samples)-1)]
        plt.axvline(ninety5pct, color = 'blue', linestyle='--', linewidth=1)
        plt.text(ninety5pct-0.05*x_range, 0.95*y_max, '95%', ha='center')
        plt.show()

    # return(qtil, score_sample_dist)
    return(score_sample_dist)


if __name__ == "__main__":
    
    # a test of rand_test():
    
    n_u = 10000

    data, _ = md.make_dataset(2, n_units = n_u, n_times = 2, prop_affected = 2/9, distribution="bernoulli",
                                treat_time = 1, base_mean = 0.6, aff_mean = 0.7, base_sd = 1, aff_sd = np.sqrt(2), treat_sd = np.sqrt(2), covariate = "3x3", noise_covariates = 0)

    #data = data.first_diff()
    data = data.binvars(n_bins = [2, 2])
    panel = data.aggbinvars(binom = True)

    def binom_counterfac(panel, ref_periods = None):
        
        def logit(x):
            return np.log(x/(1-x))

        def expit(x): 
            return 1/(1 + np.exp(-x))

        if ref_periods is None:
            ref_periods = [i for i in panel.attrs["all.periods"] if i not in panel.attrs["treat.periods"]]


        # generate counterfactual and replace sums_control with that:
        presums = panel.sel({'period':ref_periods}).sum('period')
        preprobs_treat = presums.sums_treatment / presums.counts_treatment
        preprobs_contr = presums.sums_control / presums.counts_control
        probs_contr = panel.sums_control / panel.counts_control

        panel["sums_control"] = expit(logit(probs_contr) - logit(preprobs_contr) + logit(preprobs_treat))

        return panel
    
    pan_w_ctfct = binom_counterfac(panel)
                                
    result = ds.altss(pan_w_ctfct, ltss_func=ds.multivar_sub_agg_altss, test_side = 1)

    result = ds.ltss(panel, ltss_func = ds.multivariate_subset_aggregation_ltss, test_side=1)

    q, gen = rand_test(result, panel, n_samples=2, show_plot = True)

    rand_test()




    # # set ltss function: 
    # def ltss_fun(effective_c, effective_b, score_func, priority_func, 
    #              detection_dim=detection_dim, test_side = 0):
    #     if(detection_dim == "loc"):
    #         return ds.multivariate_subset_aggregation_best_columns(effective_c=effective_c,
    #                                                                effective_b=effective_b,
    #                                                                score_func=score_func,
    #                                                                priority_func=priority_func,
    #                                                                test_side=test_side)
    #     elif(detection_dim == "unit"):
    #         return ds.multivariate_subset_aggregation_best_rows(effective_c=effective_c,
    #                                                             effective_b=effective_b,
    #                                                             score_func=score_func,
    #                                                             priority_func=priority_func,
    #                                                             test_side=test_side)
    #     elif(detection_dim == "all"): 
    #         return ds.multivariate_subset_aggregation_ltss(effective_c=effective_c,
    #                                                        effective_b=effective_b,
    #                                                        score_func=score_func,
    #                                                        priority_func=priority_func,
    #                                                        test_side=test_side)
    #     else:
    #         print("ERROR: detection_dims must be 'unit', 'loc' or 'all'.")
