import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

def wando_gen(config, n_extra_covs = 0, n_extra_t = 0):
    """
    config = {"eff_size": eff_size, "n_aff_covs": n_aff_covs, "reg_pct": reg_pct, "streams": streams}
    config = {"eff_size": 0.1, "n_aff_covs": 2, "reg_pct": [1,1], "streams": 2}

    Generates semi-simulated data from wang and overby data. Specifically, the covariate data is 
    retained while the outcomes are generated according to 'config'.

    Returns new dataset and a dictionary describing the affected subregion. 
    """

    wando = pd.read_csv("data/wando_cleaned.csv")

    eff_size = config["eff_size"] # get effect size
    n_aff_covs = config["n_aff_covs"] # get number of affected covs: [1, 2]
    reg_pct = config["reg_pct"] # get shape of affected subregion [1 is 1/3, 2 is 2/3]. If aff_covs == 1, the second value of reg_pct is ignored. 
    streams = config["streams"] # get number of affected streams

    # filter down to overall bankruptcy outcome: 
    wando = wando[wando["stream"]=="bankruptcy_per_capita"]

    # log(x+1) - transform the outcome: 
    wando['outcome'] = np.log(wando['outcome'] + 1) 

    # bin variables: 
    covs = ['employed_individuals','medianhouseholdincome','branch_count_percapita','internet_access_score']
    for c in covs:
        binbreaks = np.quantile(wando[c], [0,0.33,0.66,1])
        binbreaks = np.append(np.array([np.min(binbreaks)-1]), binbreaks[1:])
        wando[c+"_bin"] = pd.cut(wando[c], binbreaks, labels = ["0", "1", "2"], include_lowest = False)
    
    # manually make internet access score bin: 
    wando["internet_access_score_bin"] = wando["internet_access_score"].map({1:"0", 2:"0", 3:"1", 4:"1", 5:'1'})

    bincovs = [c + "_bin" for c in covs]
    wando_config = wando[covs[0] + "_bin"].astype(str) + wando[covs[1] + "_bin"].astype(str) + wando[covs[2] + "_bin"].astype(str) + wando[covs[3] + "_bin"].astype(str) 

    wando_ctrl = wando[~wando["ever_treat"]] 
    wando_trt = wando[wando["ever_treat"]] 

    # get sd under control: 
    logbank_std = np.std(wando_ctrl[wando_ctrl.stream=="bankruptcy_per_capita"]["outcome"])
    # get means conditional on covariate profile: 
    logbank_mean = wando_ctrl[wando_ctrl.stream=="bankruptcy_per_capita"].groupby([i for i in np.append(bincovs, "period")], dropna = True, observed = True)["outcome"].mean().reset_index()
    logbank_mean = logbank_mean.rename({"outcome":"outcome_mean"}, axis = 1)

    # get outcome mean and std dev for each individual/time: 
    wando['outcome_std'] = logbank_std
    wando = wando.merge(logbank_mean, on = [i for i in np.append(bincovs, "period")])

    # generate new periods, if n_extra_t > 0: 
    if n_extra_t > 0:
        profiles = wando[bincovs].drop_duplicates() # get just the covariate profiles
        new_periods = pd.DataFrame(np.array([str(i) for i in range(np.max(wando.period)+1, np.max(wando.period) + n_extra_t+1)])) # generate some new periods
        new_periods.columns = ['period']
        profiles['key'] = 1 # add dummy key
        new_periods['key'] = 1 # add dummy key
        extend_times = pd.merge(profiles, new_periods, on = "key").drop("key", axis = 1) 
        extend_times = pd.merge(wando[wando['period'] == np.max(wando.period)].drop(['period'], axis = 1), extend_times, on = bincovs) # attach the new periods to each unit, using the last period's mean and std dev.
        wando = pd.concat([wando, extend_times])

    # generate extra covariates: 
    if n_extra_covs > 0: 
        
        for i in range(n_extra_covs):
            orig = np.random.choice(bincovs[0:4]) # randomly draw an original cov
            wando["extracov_"+str(i)] = np.random.permutation(wando[orig].values)

    # generate new outcomes: 
    wando["outcome_1"] = np.random.normal(wando.outcome_mean, wando.outcome_std)
    wando["outcome_2"] = np.random.normal(wando.outcome_mean, wando.outcome_std)
    wando["outcome_3"] = np.random.normal(wando.outcome_mean, wando.outcome_std)

    # generate affected subgroup according to config: 
    aff_covs = np.random.choice(bincovs[:3], n_aff_covs, replace = False) # why [:3]: don't include internet_access_score as affectable for now
    # get binary variable for whether an observation is in the affected subgroup:
    aff_reg = {c:[i for i in wando[c].unique()] for c in bincovs} # initialize to all subgroups
    for i in range(len(aff_covs)):
        aff_reg[aff_covs[i]] = [i for i in np.random.choice(aff_reg[aff_covs[i]], reg_pct[i], replace = False)]
    
    def in_subgroup(row, region):
        out = 0 # tally of whether the row is not in the subregion of a variable
        for n in region: 
            out += row[n] not in region[n] 

        return out == 0

    wando["aff"] = wando.apply(lambda row: in_subgroup(row, aff_reg), axis = 1)

    # additive shift based on config: 
    if streams == 1:
        wando["outcome_1"] = wando["outcome_1"] + eff_size * (wando["aff"]) * wando.ever_treat * wando.lending_club_available
    elif streams == 2: 
        wando["outcome_1"] = wando["outcome_1"] + eff_size * (wando["aff"]) * wando.ever_treat * wando.lending_club_available
        wando["outcome_2"] = wando["outcome_2"] + eff_size * (wando["aff"]) * wando.ever_treat * wando.lending_club_available
    elif streams == 3:
        wando["outcome_1"] = wando["outcome_1"] + eff_size * (wando["aff"]) * wando.ever_treat * wando.lending_club_available
        wando["outcome_2"] = wando["outcome_2"] + eff_size * (wando["aff"]) * wando.ever_treat * wando.lending_club_available
        wando["outcome_3"] = wando["outcome_3"] + eff_size * (wando["aff"]) * wando.ever_treat * wando.lending_club_available

    wando = wando.drop(["outcome","stream","outcome_mean", "outcome_std"], axis = 1)
    id = [col for col in wando.columns if col not in ["outcome_1", "outcome_2", "outcome_3"]]
    wando_long = pd.wide_to_long(wando, stubnames='outcome', i=id, j="stream", sep ="_").reset_index()
    wando_long["stream"] = wando_long.stream.astype(str)

    wando_long.period = wando_long['period'].astype("int") # convert period to int
    #wando_long.loc[wando_long['aff'], 'group'] = 'A'

    return wando_long, aff_reg


