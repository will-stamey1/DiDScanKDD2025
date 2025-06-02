import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from random import sample
import re
import itertools
import warnings
from fractions import Fraction
from scipy.linalg import block_diag

# make subclass of xarray dataset that adds some methods: 
class dndset(xr.Dataset):
    __slots__ = ()
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.attrs["binned"] = False 
        self.attrs["agged"] = False 
        self.attrs["diffed"] = False 
        self.attrs["staggered"] = False
        self.attrs["all.periods"] = [str(j) for j in np.sort([int(i) for i in np.unique(self.period.values)])]
        self.attrs["treat.periods"] = None

        #if self.period is not None:
            # get first period during treatment: 
        #    first_per = np.min([int(i) for i in self.period.to_numpy()[self.treat_time.to_numpy()=='1']])
        #    self.attrs["treat.periods"] = [i for i in self.attrs["all.periods"] if int(i) >= first_per]

    
    def vis_cov(self, cov_name = "covariate"):
        d = self.to_dataframe().reset_index(level=list(self.dims))

        d = d[d["group"] != "C"]
        dpre = d[d["treat_time"] == 0]
        dpost = d[d["treat_time"] == 1]

        dpost = dpost.groupby(["stream", "unit", "group", cov_name])["outcome"].mean().reset_index().rename(columns={'Value': 'Mean_Value'})
        dpre = dpre[dpre["period"] == str(dpre["period"].max())].reset_index(drop = True)
        x = dpost[cov_name]
        y = dpost["outcome"] - dpre["outcome"]

        plt.scatter(x, y, label='Treatment Effect Over Covariate')  # Plot y1
        plt.xlabel('Covariate')
        plt.ylabel('Outcome')
        #plt.legend()
        plt.title('')
        plt.show()

    def visbin2d(self, cov_names = None):
        """
        cov_names selects which covariates are used to visualize. 
        
        if binned already, makes a heatmap. Otherwise, makes a scatterplot.
        """

        # default names: 
        if self.attrs["binned"] == True and cov_names is None:
            cov_names = ["binned1", "binned2"]

        if self.attrs["binned"] == False and cov_names is None:
            cov_names = ["variate1", "variate2"]

        if self.attrs["binned"]:

            dims = [np.array(self.coords[name]) for name in cov_names]
            twodbins = list(itertools.product(dims[0], dims[1]))

            shift = self.counts_treatment - self.counts_control

            d = [[shift.sel({cov_names[0]:[a], cov_names[1]:[b]}).mean().item() for a in dims[0]] for b in dims[1]]

            plt.imshow(d, cmap='viridis', interpolation='nearest')
            plt.colorbar(label='Difference')
            plt.xlabel(cov_names[0])
            plt.ylabel(cov_names[1])
            plt.title('Treatment - Control Shift')
            plt.show()

        else:
            a = self.to_dataframe().reset_index()
            a = a[a["period"] == '1']
            a = a[a['group']!='C'] # filter C's

            color_map = {'A': 'blue', 'T': 'red'}
            colors = [color_map[cat] for cat in a.group]

            plt.figure(figsize=(8, 6))
            plt.scatter(a[cov_names[0]], a[cov_names[1]], c=colors, alpha=0.5, label=a.group)

            # Customizing plot
            plt.title('Scatter Plot with Categorical Colors')
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.legend()

            plt.show()

    def biased_sampling(self, props = None):
        """Takes a binned dndset and downsamples the treatment group values in each 
        bin according to the values of `props`"""        

        # TODO possibly: Make compatible with unaggregated but binned data. 

        # props = array of same dimensions as covariate bins specifying downsampling rate for each bin.  
        
        if self.attrs["agged"] is False:
            raise RuntimeError("This dndset is not yet aggregated and so it cannot be resampled.")
        else:
            vardims = list(self.dims)[2:]

            if props is None:
                # Make a default props array where only the 0,0...,0 bin is downsampled: 
                shape = [len(self.coords[i]) for i in vardims]
                
                props = np.ones(shape)
                firstcoord = tuple([0 for i in vardims])
                props[firstcoord] = 0.5

            downsample = xr.DataArray(props, coords = self[vardims].coords)
            a = np.multiply(self[["sums_treatment", "sos_treatment", "counts_treatment"]], downsample)

            bisamp_data = self
            bisamp_data[["sums_treatment", "sos_treatment", "counts_treatment"]] = a 

            return bisamp_data

    def binvars(self, varname = None, n_bins = 2, make_char = True, suffix = None):
        """
        Takes xarray dataset - takes the values of the covariate arrays and makes new 
        variables for each specifying a bin/multinomial value.

        varname: 
        n_bins_per_dim: 
        make_char: if true, levels are characters rather than integers.  
        """

        if self.attrs["binned"] is False:

            d = self
            v = varname
            nb = n_bins

            if suffix is None:
                suffix = ""

            newnames=None
            if v is None: # assume generated data; use generic variable names
                v = [name for name in list(d.variables) if "variate" in name]
            else: # v exists; get newnames for binned variables: 
                newnames = [name + suffix for name in v]
            
            if isinstance(v, str): 
                v = [v]

            for n in v: 
                thisv = d[n].as_numpy() 
                thisv = thisv[0, :, 0] # Filter down to unique values, one per unit

                # get the number of bins for this variable
                if type(nb) is list:
                    nb_i = nb[np.where(np.array(v) == n)[0][0]]
                else: 
                    nb_i = nb

                # make quantile-split categories: 
                newv = pd.qcut(thisv.to_numpy(), q=nb_i, labels = [str(i + 1) for i in np.arange(nb_i)])

                if not make_char:
                    newv = [int(val) for val in newv]

                newv = np.array([newv for _ in range(d[n].shape[2])])
                newv = np.array([newv for _ in range(d[n].shape[0])]).transpose(0,2,1)

                if newnames: 
                    d[newnames[v.index(n)]] = xr.DataArray(newv,
                                                    dims = ['stream', 'unit', 'period']
                                        )
                else: 
                    d["binned"+str(v.index(n) + 1)] = xr.DataArray(newv,
                                                    dims = ['stream', 'unit', 'period']
                                        )
                
            d.attrs["binned"] = True

        else: # Already binned!
            raise RuntimeError("This dndset is already binned!")
            
        return d
    
    def aggbinvars(self, vars2aggby = None, outcome = "dif", binom = False, reftimes = None):
        
        if self.attrs["binned"] is False:
            raise RuntimeError("This dndset is not binned, or else has no covariates, and so it cannot be aggregated. Use 'binvars' to bin continuous variables.")
        elif self.attrs["agged"] is True:
            raise RuntimeError("This dndset has already been aggregated!")
        else:

            if self.attrs["diffed"] is False:
                warnings.warn("The outcomes in this dndset are the raw outcomes rather than the first differences, which is not currently supported by the LTSS functions suite. Applying the 'first_diff' method before aggregation is required to use those functions.")

            if binom: # don't use difs if doing binom: 
                outcome = "outcome"

            if reftimes is None: 
                reftimes = [i for i in self.attrs["all.periods"] if i not in self.attrs["treat.periods"]]

            d = self
            v = vars2aggby

            if v is None: 
                v = [name for name in list(d.variables) if "binned" in name]
            elif type(v) is not list: 
                v = [v]

            if self.staggered:
                # get this strange nan thing out: 
                d = d.fillna(np.nan)

            # Get dictionary of lists of possible values for binned variables:
            binvaldict = {}
            for var in v: 
                # commented out code was trying to order variable values. Not always applicable. How to handle? 
                #flatdata = d[var].values.flatten().astype("float")
                #binvaldict[var] = np.unique(flatdata[~np.isnan(flatdata)]).astype("int").astype("str")
                binvaldict[var] = np.array([i for i in list(set(self[var].to_numpy().flatten()))])

            # get dimensions of final new xr dataset: 
            dims = [len(np.unique(d.stream)), len(np.unique(d.period))] 
            for b in v:
                dims.append(len(binvaldict[b]))

            # get "treat_time" variable from d and reshape them for this data. 
            if self.staggered: 
                t_time = d['treat_time'][:, 0, :, 0].to_numpy()
            else: 
                t_time = d['treat_time'][:, 0, :].to_numpy()
            
            trans_order = [i for i in np.arange(-2, len(dims) - 2)]
            t_time2 = np.tile(t_time, (np.append(np.append(dims[2:], 1), 1))).transpose(trans_order)


            # make sums, sum of squares and count arrays: 
            sums_treat = np.zeros((dims))
            sos_treat = np.zeros((dims))
            counts_treat = np.zeros((dims))
            sums_control = np.zeros((dims))
            sos_control = np.zeros((dims))
            counts_control = np.zeros((dims))
            #rawsums_control = np.zeros((dims))

            # get observed categorical covariate combinations: 
            catcombs = d.sel({'stream':d['stream'].values[0], 'period':d['period'].values[0]})[np.append(v,'group')].to_dataframe()[np.append(v,'group')]
            catcombs.loc[catcombs['group']!='C', 'group'] = 'T'
            catcombsuni = catcombs.drop_duplicates().dropna() # get just the unique values of the covariates.

            for ix, row in catcombsuni.iterrows():
                # ix, row = next(catcombsuni.iterrows())

                # 1. get the units who have this combination of covariates: 
                theseunits = catcombs.index[catcombs.eq(row).all(axis=1)]

                # 2. Make selector for assigning values to only the relevant covariate profiles
                selector = [slice(None)]*2
                for var in v:
                    i = np.where(binvaldict[var] == row.drop('group')[var])[0][0] # get the index of the bin/factor level
                    selector.append(slice(i, i+1))

                # make newshape for how the results will be inserted into the new dataset:
                newshape = [d.sizes['stream'], d.sizes['period']]
                for i in v:
                    newshape.append(1)


                # Here, split based on whether "staggered":
                if self.staggered: 
                    if row["group"] == "C": # only iterating through Treatment for staggered data
                        continue
                    
                    # 3. sum up the values and add to the sums array, and add number of units to the counts array. 
                    vals = d[outcome].sel({'unit':[i[0] for i in theseunits.values]})
                    valssqrd = vals**2

                    # put in treatment data:
                    counts_treat[tuple(selector)] = len(theseunits)
                    sums_treat[tuple(selector)] = vals.sum(dim = ['unit', 'treat_period']).to_numpy().reshape(newshape)
                    sos_treat[tuple(selector)] = valssqrd.sum(dim = ['unit', 'treat_period']).to_numpy().reshape(newshape)

                    # get counterfactual (baseline) data: 
                    contrrow = row.copy()
                    contrrow.group = "C"
                    contrunits = catcombs.index[catcombs.eq(contrrow).all(axis=1)]
                    
                    # add count of control units, the number of control obs in this covariate bin: 
                    counts_control[tuple(selector)] = len(np.unique([i[0] for i in contrunits]))  

                    for u in theseunits: # iterate through treatment units and get their treat_period's counterfactual: 
                        contr_u = [c for c in contrunits if c[1] == u[1]] # get contr ix from same year
                        
                        contr_vals = d[outcome].sel({'unit':[i[0] for i in contr_u], 'treat_period':[i[1] for i in contr_u]})
                        contr_valssqrd = contr_vals**2

                        sums_control[tuple(selector)] = sums_control[tuple(selector)] + (contr_vals.mean(dim = ["unit", "treat_period"])).values.reshape(newshape)

                        # TEMPORARY AND PROBABLY INVALID: taking the average of the sum of squares of the counterfactual sets for each of the treated obs:
                        sos_control[tuple(selector)] = sos_control[tuple(selector)] + (contr_valssqrd.sum(dim = ["unit", "treat_period"])).values.reshape(newshape)

                    # multiply sums by number of control units and divide by number of treatment units: 
                    sums_control[tuple(selector)] = np.multiply(counts_control[tuple(selector)],  sums_control[tuple(selector)])
                    sums_control[tuple(selector)] = np.divide(sums_control[tuple(selector)], len(theseunits))

                    # TEMPORARY AND PROBABLY INVALID: taking the average of the sum of squares of the counterfactual sets for each of the treated obs:
                    sos_control[tuple(selector)] = sos_control[tuple(selector)] / len(theseunits)

                else: # not staggered: 

                    # 3. sum up the values and add to the sums array, and add number of units to the counts array. 
                    vals = d[outcome].sel({'unit':theseunits.values})
                    valssqrd = d["outcome"].sel({'unit':theseunits.values})**2 # vals**2

                    if row['group'] == 'C':
                        counts_control[tuple(selector)] = len(theseunits)  # same across all period/stream dimensions. 
                        sums_control[tuple(selector)] = vals.sum(dim = 'unit').to_numpy().reshape(newshape)
                        sos_control[tuple(selector)] = valssqrd.sum(dim = 'unit').to_numpy().reshape(newshape)
                        #rawsums_control[tuple(selector)] = rawvals.sum(dim = 'unit').to_numpy().reshape(newshape)
                    else:
                        counts_treat[tuple(selector)] = len(theseunits)
                        sums_treat[tuple(selector)] = vals.sum(dim = 'unit').to_numpy().reshape(newshape)
                        sos_treat[tuple(selector)] = valssqrd.sum(dim = 'unit').to_numpy().reshape(newshape)

            # convert to xr data arrays
            coords = {'stream':d['stream'].values, 'period':d['period'].values}
            
            for var in v:
                coords[var] = binvaldict[var]
            
            nuds = xr.Dataset(
                                data_vars = 
                                    dict(sums_treatment = (list(coords.keys()), sums_treat),
                                        sos_treatment = (list(coords.keys()), sos_treat),
                                        counts_treatment = (list(coords.keys()), counts_treat),
                                        sums_control = (list(coords.keys()), sums_control),
                                        sos_control = (list(coords.keys()), sos_control),
                                        counts_control = (list(coords.keys()), counts_control),
                                        #rawsums_control = (list(coords.keys()), rawsums_control),
                                        treat_time = (list(coords.keys()), t_time2)), 
                                coords = coords
                            )
            
            # convert to dndset
            nuds = dndset(nuds)

            # get attributes from self
            nuds.attrs = self.attrs 

            # update "agged":
            nuds.attrs["agged"] = True
        
        return nuds

    
    def first_diff(self, ref_times = None):
        """
        Gets first difference, changing the value of outcome for post-treatment  to 
        the difference between post treatment and the mean of the pre-treatment values. 
        """

        if self.attrs["diffed"] == True:
            raise RuntimeError("First differences have already been taken.")
        else:

            if ref_times is None:
                pre_per = [int(i) for i in self.attrs["all.periods"] if i not in self.attrs["treat.periods"]]
            else:
                pre_per = np.array([int(j) for j in ref_times])

            earliest = min([int(j) for j in self.period]) # get earliest time point name
            pre_per = np.subtract(pre_per, earliest) 

            pre_means = self.outcome[:, :, pre_per].mean(axis = 2)
            # set all preperiods to have a 'dif' value of nan: 
            self['dif'] = self.outcome - pre_means
            self['dif'] = xr.where(self['treat_time']==0, np.nan, self['dif'])

            self.attrs["diffed"] = True

        return self

    def get_ix(self, group, var = "group", dim = None):
        """
        Takes xarray dataset and returns indices for the requested group.
        group is "A" for affected, "T" for treatment (including affected)
            and "C" for control.

        "var" indicates the variable indexing on. Assumed to be "group" 
            for treatment group. Alternative is "time" for treatment time.
        """

        if var == "group":
            if group == "A":
                ix = np.where(self["group"] == "A") # get affected group indices
            if group == "T":
                ix = np.where((self["group"] == "T") | (self["group"] == "A")) # get treatment group indices
            if group == "C":
                ix = np.where(self["group"] == "C") # get control group indices
        elif var == "time":
            if group == "Post" or group == "post":
                ix = np.where(self["treat_time"] == 1)
            elif group == "Pre" or group == "pre":
                ix = np.where(self["treat_time"] == 0)

        if dim == None:
            ix = (np.unique(ix[0]), np.unique(ix[1]), np.unique(ix[2]))
        else:
            ix = (np.unique(ix[dim]))

        return ix     

def subset_by_covs(xr_data, subset, search_covs = None):

    # xr_data should be an unaggregated xr dataset

    if search_covs is None:
        search_covs = [i for i in list(xr_data.keys()) if i in list(subset.keys())]

    dset_copy = xr_data.copy()

    # subset the streams if applicable: 
    if 'stream' in list(subset.keys()):
        dset_copy = dset_copy.sel({'stream':subset['stream']})

    # initialize mask, all True 
    condition = xr.DataArray(True, dims=["unit"], coords={"unit": xr_data.unit})

    # Loop over each invariant variable and build the condition
    for var in search_covs:
        # Create a condition for the current variable based on allowed values
        valid_values = subset.get(var, [])
        
        if valid_values:
            # Create a condition where the current variable is in the allowed values
            var_condition = xr_data.sel({"period":xr_data.period[0].item(), 'stream':xr_data.stream[0].item()})[var].isin(valid_values)  # Assumes values are the same over time
            
            # Combine with the overall condition using logical AND
            condition = condition & var_condition

    return dset_copy.sel({"unit":xr_data['unit'].where(condition, drop=True).to_numpy()})


#create 3D xarray dataset
def make_dataset(n_streams = int, n_units = int, n_times = 2, treat_time = 1, 
                 start_time = 0, n_units_treat = None, distribution = "gaussian", 
                 prop_affected = 0.1, base_mean = 0, base_sd = 1, treat_mean = None, 
                 treat_sd = None, aff_mean = None, aff_sd = None, rand_affected = False,
                 aff_streams = None, 
                 covariate_mean_diff = 1, covariate = None, noise_covariates = 0, aff_periods = None):
    
    """
    treat_time: first time period treatment has effect. time periods go from 0 up. 
    distribution: sampling distribution for tensor data. "poisson", "normal" and 
                  "multinorm" currently supported.
    start_time: value of the first period dimension. 0 by default. 
    n_units_treat: number of treatment units. By default, this is one half of 
                   n_units, rounded down.
    prop_affected: proportion of the treatment group that is especially sensitive 
                    to the treatment.
    base_mean: baseline value for the Y(0) distribution, the outcome without 
            treatment. this is lambda for the poisson and mu for the normal.
    base_sd: standard deviation of baseline (control) value
    treat_mean: outcome for treatment individuals with low/no sensitivity to treatment
    treat_sd: standard deviation for treatment individuals with low/no sensitivity to
              treatment. 
    aff_mean: effect size for the sensitive group.
    aff_sd:  standard deviation of the effect. only relevant for normal case. 
    rand_affected: should the affected subgroup be randomly selected? if not, 
                   default to the first (n_units x prop_affected) units.
    aff_streams: Which streams are affected? Provide list of affected streams or list
                 of length n_streams with True and False. Default, "None", results in all 
                 streams being treated. 
    covariate_mean_diff: how different is the covariate on average for affected
                         vs unaffected subgroups?
    covariate: Default is None, which means no covariate. Otherwise, specifies what 
                kind of covariate to generate. 
                Options: 'normal' - normal distribution with homogeneous variance 
                         and a mean shift.
                         'unimidshift' - normal distribution
    aff_periods: if not None, supplies which period indices where there is an effect for affected
                 stream/unit combos. For other periods, the expected values and variances for 
                 these observations come from the "treat" values for treated but unaffected individuals.
    """

    def make_tensor(mean, sd, shape, correlation = None):
        """
        Takes base mean or effect mean and the dimensions of the data. Returns 
        a tensor with the given dimensions with values distributed according 
        to the effect argument. 

        mean: Can be an integer or a string. The string should specify the 
                effect mean as a function of time period, where t = 0 is the 
                reference time. 
                
                The string should be of the form 
                    "y ~ constant + coefficient_1 * t + coefficient_2 * t^2".
        
        sd: Like effect, can be an integer or a string, with the same 
                   logic. Can also be a matrix, in which case the matrix's 
                   d dimensions must be of the same shape as the first d 
                   elements of "shape". 
        
        correlation: "period", "stream", or "both". Specifies what dimensions are 
            captured by the standard deviation covariance matrix if applicable. 
            Only necessary when n_streams == n_periods. 
        """

        def parse_model(func):
            """
            This function reads a string specifying the functional relationship 
            between effect and period t, returning the function specified. 
            """
            func = func.replace(" ", "")
            pattern = r"(?=[+-])|\A"
            
            # Use re.findall() to find all matches
            parts = re.split(pattern, func) # split by '+' and '-'
            parts = [i.replace('+', '') for i in parts if i != ''] # remove empty substrings and take out '+'s            

            # output function has two components: an exp vector which specifies 
            # order of t for each term, and a multiplier vector which specifies 
            # coefficients.
            exp_vec, m_vec = [], []

            for i in range(len(parts)):
                p = parts[i] 
                operators = [c for c in ["*", "/", "^"] if c in p]
                
                if len(operators) == 0 and 't' in p: 
                    # p is just t: 
                    exp_vec.append(1)
                    m_vec.append(1)
                elif len(operators) == 0: 
                    # p is just a constant, add p to m_vec and 1 to t_vec
                    exp_vec.append(0)
                    m_vec.append(float(p))
                else: 
                    # reg exp to detect exponent: 
                    find_exp = r'\^(\d+(\.\d+)?)([*/]?)(?=$|[^*/])'

                    match = re.search(find_exp, p)
                    if match:
                        number = match.group(1)
                        p = p[:match.start()] + p[match.end():]
                        p = p.replace('t', '')
                    else:
                        number = 1
                        p = p.replace('t', '')

                    exp_vec.append(float(number))

                    coef = re.sub(r'[^\d.-]+', '', p)
                    #coef = re.sub(r'(-?\b\w+\b)', '', p)
                    if coef != '':
                        m_vec.append(float(coef))
                    else:         
                        m_vec.append(1)     

            # make function as requested by user: 
            def f_o_t(period):
                ts = np.array([period ** i for i in exp_vec]) # get vector of t to the power of each element of exp_vec: 
                ms = np.array(m_vec) # make coefficient array

                return np.dot(ms, ts) 
            
            return f_o_t

        if isinstance(mean, str) or isinstance(sd, str):
            # in this condition, use parse_model to get a function of t and 
            # generate data:

            if isinstance(mean, str): 
                mean_fot = parse_model(mean) # get f of t from effect string
                mean_by_t = [mean_fot(t) for t in range(start_time, shape[2] - start_time)]
            else: 
                mean_by_t = [mean for _ in range(start_time, shape[2] - start_time)]

            if isinstance(sd, str):
                sd_fot = parse_model(sd)
                sd_by_t = [sd_fot(t) for t in range(start_time, shape[2] - start_time)]
            else:
                sd_by_t = [sd for _ in range(start_time, shape[2] - start_time)]                

            data_by_period = []
            for t in range(shape[2]):
                slice_shape = (shape[0], shape[1], 1)
                data_by_period.append(dist(mean_by_t[t], sd_by_t[t], slice_shape, dist = distribution))
            
            # combine slices into a full dataset:
            return np.concatenate(data_by_period, 2)
            
        else: 
            if isinstance(sd, (int, float)):
                return dist(mean, sd, shape, dist = distribution)
            else: # sd is a covariance matrix  
                
                # check that sd is shaped corresponding to the outcome streams, the time periods, or streams X periods: 
                if np.shape(sd)[0] not in (shape[0], shape[2], shape[0] * shape[2]):
                    RuntimeError(f"If SD is a matrix, it must be of size (nstreams, nstreams), (nperiods, nperiods) or (nstreams X nperiods, nstreams X nperiods).")
                else: 

                    # determine how many multivariate normal draws we need: 
                    indep_dims = []
                    if correlation is None:
                        if np.shape(sd)[0] == shape[0]:
                            correlation = "stream"
                        elif np.shape(sd)[0] == shape[2]:
                            correlation = "period"
                        else:
                            correlation = "both"
                    if shape[0]==shape[2] and correlation is None:
                        RuntimeError(f"If n_periods = n_streams, correlation must be specified.")

                    # specify the independent dimensions depending on what part of the correlation structure is specified: 
                    if correlation == "stream":
                        indep_dims = [shape[1], shape[2]]
                        blocks = shape[2] # specify the number of blocks in the block-diagonal covariance matrix for each panel unit
                    elif correlation == "period":
                        indep_dims = [shape[0], shape[1]]
                        blocks = shape[0]
                    elif correlation == "both":
                        indep_dims = [shape[1]]
                        blocks = 1
                    
                    # convert sd into a block-diagonal matrix: 
                    sdexp = blocks * [sd]
                    sdexp = block_diag(*sdexp)

                    # sample: 
                    sample = dist(mean, sd, indep_dims)

                    # TODO: DOUBLE CHECK THESE:
                    if correlation == "stream":
                        np.transpose(sample, (1,2,0))
                    if correlation == "both":
                        sample.reshape((shape[1], shape[0], shape[2]))
                        sample.transpose((1,0,2))

                    return(sample)
        
    # Set treat_mean, treat_sd, aff_mean and aff_sd to the value of baselines by default
    if treat_mean is None:
        treat_mean = base_mean
    if treat_sd is None:
        treat_sd = base_sd
    if aff_mean is None:
        aff_mean = base_mean
    if aff_sd is None:
        aff_sd = base_sd
    if aff_periods is None:
        aff_periods = np.arange(treat_time, start_time + n_times)

    if n_units_treat == None:
        n_units_treat = int(n_units/2)
    
    # set default affected streams:
    if aff_streams is None: 
        aff_streams = [i.item() for i in np.arange(n_streams)]
    elif isinstance(aff_streams[0], bool):
        aff_streams = [i.item() for i in np.arange(n_streams) if aff_streams[i]]

    # adjust prop_affected if set to 0: 
    if prop_affected <= 0 or prop_affected > 1:
        prop_affected = 0.1
        print("prop_affected must be between 0 and 1, non-inclusive of 0. If no affected subgroups are desired, set aff_mean = 0 and leave prop_affected to default. ")

    if not(treat_time in np.arange(n_times)+start_time):
        print("ERROR: Treatment time must be in the range of times.")
    else:

        treat_shape = [n_streams, n_units_treat, n_times]

        # Set distribution function:
        def dist(mean, sd, shape, dist = distribution):
            if dist == "poisson":
                return np.random.poisson(mean, size = shape)
            elif dist == "normal" or dist == "gaussian":
                return np.random.normal(mean, sd, size = shape)
            elif dist == "multinorm":
                if isinstance(mean, (int, float)):
                    mean = [mean] * np.shape(sd)[0]
                return np.random.multivariate_normal(mean, sd, shape)
            elif dist == "bernoulli":
                return np.random.binomial(n = 1, p = mean, size = shape)
            else: 
                print("ERROR: The submitted distribution type is not supported. " + 
                      "Options 'poisson' and 'normal' are currently available.")

        # Pick a (possibly random) affected subgroup: 
        n_affected = int(round(n_units_treat * prop_affected, 0))
        if rand_affected:
            affected = sample(range(n_units_treat), n_affected)
        else:
            affected = np.arange(n_affected)

        # make treatment tensor, no affected
        treat_tensor = make_tensor(treat_mean, treat_sd, shape=treat_shape)

        # remove baseline treat value for affected sub-pop in post-treatment time period, at affected streams
        #treat_tensor[:, affected, (treat_time-start_time):] = 0
        treat_tensor[np.ix_(aff_streams, affected, (aff_periods-start_time))] = 0

        # Make effect tensor, the effect for the affected-treated
        aff_tensor = make_tensor(aff_mean, aff_sd, treat_shape) 

        # Make effect zero for all in the pre-treatment period and for all not in affected subgroup at all "affected" periods and streams: 
        masked_periods = [i.item() for i in np.arange(0, n_times) if i not in aff_periods - start_time] # (masked periods are all periods that are not in the "affected periods" set)
        aff_tensor[:, :, masked_periods] = 0
        masked_streams = [i.item() for i in np.arange(0, n_streams) if i not in aff_streams]
        aff_tensor[masked_streams, :, :] = 0
        mask = np.array([True for i in range(n_units_treat)])
        mask[affected] = False
        aff_tensor[:, mask, :] = 0

        # add the two tensors to get the full treatment tensor: 
        treat_tensor = treat_tensor + aff_tensor

        # Make control dataset: 
        contr_shape = [n_streams, n_units - n_units_treat, n_times]
        contr_tensor = make_tensor(base_mean, base_sd, shape = contr_shape)

        # bind the tensors together: 
        data = np.concatenate([treat_tensor, contr_tensor], axis = 1)

        da = xr.DataArray(data, 
                            dims = ['stream', 'unit', 'period'],
                            coords = dict(
                                stream = [str(x) for x in np.arange(0,n_streams)],
                                unit = [str(x) for x in np.arange(0,n_units)],
                                period = [str(x) for x in np.arange(start_time,start_time + n_times)]
                            )
                        )
        
        # make second dataarray with the labels
        treat_labs = np.full(treat_shape, "T")
        treat_labs[:, affected, :] = "A"
        contr_labs = np.full(contr_shape, "C") 
        labels = np.concatenate([treat_labs, contr_labs], axis = 1)
        da_labels = xr.DataArray(labels,
                                 dims = ['stream', 'unit', 'period'],
                                coords = dict(
                                            stream = [str(x) for x in np.arange(0,n_streams)],
                                            unit = [str(x) for x in np.arange(0,n_units)],
                                            period = [str(x) for x in np.arange(start_time, start_time + n_times)]
                                        )
                        )
        
        # make a treatment time variable:
        pre_treat_dummy = np.full((n_streams, n_units, (treat_time - start_time)), 0)
        post_treat_dummy = np.full((n_streams, n_units, n_times - (treat_time - start_time)), 1)
        treat_time_dummy = np.concatenate([pre_treat_dummy, post_treat_dummy], axis = 2)
        da_treat = xr.DataArray(treat_time_dummy,
                                dims = ['stream', 'unit', 'period'],
                                coords = dict(
                                            stream = [str(x) for x in np.arange(0,n_streams)],
                                            unit = [str(x) for x in np.arange(0,n_units)],
                                            period = [str(x) for x in np.arange(start_time, start_time + n_times)]
                                        )
                )

        # join the two dataarrays into a dataset: 
        xrda = xr.Dataset({
            "outcome" : da,
            "group" : da_labels,
            "treat_time" : da_treat})

        # make covariate dataarrays: 
        if covariate is not None:
            # makes (n_stream, n_streams, n_times) size matrix where each stream has a covariate value fixed across other dimensions.

            affected_dummy = np.zeros(n_units_treat)
            affected_dummy[affected] = 1

            variates = [] # allocate to store covariate(s) generated in the following chunks. 

            if covariate == "normal":    
                contr_cov = np.random.normal(size = contr_shape[1]) # make covariate values for each stream
                contr_cov = np.array([c + np.random.binomial(covariate_mean_diff, prop_affected, 1).item() for c in contr_cov])

                # makes covariate for the treatment group so that the affected subgroup has a shifted value on average.
                treat_cov = np.random.normal(size = treat_shape[1])
                treat_cov = np.array([mu + a*covariate_mean_diff for mu, a in zip(treat_cov, affected_dummy)])

                variates.append((contr_cov, treat_cov))

            elif covariate == "unimidshift": # make a uniformly distributed covarate with prop_affected-sized section in the middle, where the outcome will be shifted by aff_mean
                contr_cov = np.random.uniform(size = contr_shape[1])

                treat_cov = np.zeros(shape = n_units_treat)
                for i in range(n_units_treat):
                    if affected_dummy[i] == 1:
                        treat_cov[i] = np.random.uniform(low = 0.5 * (1 - prop_affected), high = 0.5 * (1 + prop_affected), size = 1).item()
                    else:    
                        treat_cov[i] = np.random.uniform(low = 0.5*(1 + prop_affected), high = 1 + 0.5*prop_affected, size = 1).item()
                        if treat_cov[i] >= 1:
                            treat_cov[i] -= 1

                variates.append((contr_cov, treat_cov))

            elif covariate == "unimid2d":
                contr_cov1 = np.random.uniform(size = contr_shape[1])
                contr_cov2 = np.random.uniform(size = contr_shape[1])
                treat_cov1 = np.zeros(shape = n_units_treat)
                treat_cov2 = np.zeros(shape = n_units_treat)

                for i in range(n_units_treat):
                    if affected_dummy[i] == 1:
                        treat_cov1[i] = np.random.uniform(low = 0.5 * (1 - np.sqrt(prop_affected)), high = 0.5 * (1 + np.sqrt(prop_affected)), size = 1).item()
                        treat_cov2[i] = np.random.uniform(low = 0.5 * (1 - np.sqrt(prop_affected)), high = 0.5 * (1 + np.sqrt(prop_affected)), size = 1).item()
                    else:    
                        treat_cov1[i] = np.random.uniform(low = 0.5*(1 + np.sqrt(prop_affected)), high = 1.5 - 0.5*np.sqrt(prop_affected), size = 1).item()
                        treat_cov2[i] = np.random.uniform(low = 0.5*(1 + np.sqrt(prop_affected)), high = 1.5 - 0.5*np.sqrt(prop_affected), size = 1).item()                        
                        if treat_cov1[i] >= 1:
                            treat_cov1[i] -= 1
                        if treat_cov2[i] >= 1:
                            treat_cov2[i] -= 1

                variates.append((contr_cov1, treat_cov1))
                variates.append((contr_cov2, treat_cov2))

            elif covariate == "noise": # Make a covariate which is unrelated to the treatment effect: 
                contr_cov = np.random.uniform(size = contr_shape[1])
                treat_cov = np.random.uniform(size = treat_shape[1])

                variates.append((contr_cov, treat_cov))

            elif covariate == "hacksaw": # Make a covariate with many ups and downs.
                contr_cov = np.random.uniform(size = contr_shape[1])
                treat_cov = np.random.uniform(size = treat_shape[1], low = 0, high = 0.1)

                for i in range(n_units_treat):
                    if affected_dummy[i] == 1:
                        treat_cov[i] += np.random.choice(list([0.1, 0.3, 0.5, 0.7, 0.9]))
                    else:    
                        treat_cov[i] += np.random.choice(list([0.0, 0.2, 0.4, 0.6, 0.8]))

                variates.append((contr_cov, treat_cov))
            
            elif covariate == "corner2d": # make 2d covariates with a corner square anomalous
                contr_cov1 = np.random.uniform(size = contr_shape[1])
                contr_cov2 = np.random.uniform(size = contr_shape[1])
                treat_cov1 = np.zeros(shape = n_units_treat)
                treat_cov2 = np.zeros(shape = n_units_treat)

                for i in range(n_units_treat):
                    if affected_dummy[i] == 1:
                        treat_cov1[i] = np.random.uniform(low = 0, high = np.sqrt(prop_affected), size = 1).item()
                        treat_cov2[i] = np.random.uniform(low = 0, high = np.sqrt(prop_affected), size = 1).item()
                    else:    
                        # The following lines uniformly sample from the symmetric L shape angled around the treat-effect square:
                        treat_cov1[i] = np.random.uniform(low = np.sqrt(prop_affected), high = 1, size = 1).item()
                        treat_cov2[i] = np.random.uniform(low = 0, high = 1 + np.sqrt(prop_affected), size = 1).item()   
                        if treat_cov2[i] > 1:
                            treatcov2temp = treat_cov2[i]
                            treat_cov2[i] = treat_cov1[i]
                            treat_cov1[i] = treatcov2temp - 1

                variates.append((contr_cov1, treat_cov1))
                variates.append((contr_cov2, treat_cov2))

            elif covariate == "3x3":

                acc_props = [1/9, 2/9, 4/9, 2/3, 1]

                def nearest(x, float_list, epsilon):
                    for f in float_list:
                        if abs(x - f) <= epsilon:
                            return f
                    raise RuntimeError(f"Currently, using 3x3rects requires supplying prop_affected equal to one of [1/9, 2/9, 4/9, 2/3, 1].")

                # get value nearest to the provided prop_affected (this allows user to input values not precisely equal to the fractions)
                nrst = nearest(prop_affected, acc_props, 0.02)

                # generate control data uniformly
                contr_cov1 = np.random.uniform(size = contr_shape[1])
                contr_cov2 = np.random.uniform(size = contr_shape[1])
                treat_cov1 = np.zeros(shape = n_units_treat)
                treat_cov2 = np.zeros(shape = n_units_treat)

                for i in range(n_units_treat):
                    firstset = [1/9, 4/9, 1]
                    # Generates the "affected" rectangle: 
                        #   A | U | U      A | A | U      A | A | A
                        #   U | U | U  or  A | A | U  or  A | A | A
                        #   U | U | U      U | U | U      A | A | A

                    if nrst in firstset: 
                        if affected_dummy[i] == 1:
                            treat_cov1[i] = np.random.uniform(low = 0, high = np.sqrt(prop_affected), size = 1).item()
                            treat_cov2[i] = np.random.uniform(low = 0, high = np.sqrt(prop_affected), size = 1).item()
                        else:
                            # The following lines uniformly sample from the symmetric L shape angled around the affected square:
                            treat_cov1[i] = np.random.uniform(low = np.sqrt(prop_affected), high = 1, size = 1).item()
                            treat_cov2[i] = np.random.uniform(low = 0, high = 1 + np.sqrt(prop_affected), size = 1).item()   
                            if treat_cov2[i] > 1:
                                treatcov2temp = treat_cov2[i]
                                treat_cov2[i] = treat_cov1[i]
                                treat_cov1[i] = treatcov2temp - 1

                    elif nrst == 2/9:
                        # Generates the "affected" rectangle: 
                        #   A | A | U
                        #   U | U | U
                        #   U | U | U

                        if affected_dummy[i] == 1:
                            treat_cov1[i] = np.random.uniform(low = 0, high = 2/3)
                            treat_cov2[i] = np.random.uniform(low = 0, high = 1/3)
                        else:
                            # The following lines uniformly sample from the 'U's in the diagram above:
                            treat_cov2[i] = np.random.uniform(low = 0, high = 7/3) 
                            if treat_cov2[i] > 5/3:
                                treat_cov1[i] = np.random.uniform(low = 0, high = 1/3)
                                treat_cov2[i] = treat_cov2[i] - 4/3
                            elif treat_cov2[i] > 1:
                                treat_cov1[i] = np.random.uniform(low = 1/3, high = 2/3)
                                treat_cov2[i] = treat_cov2[i] - 2/3
                            else: 
                                treat_cov1[i] = np.random.uniform(low = 2/3, high = 1)

                    else: # nrst == 2/3

                        # Generates the "affected" rectangle: 
                        #   A | A | U
                        #   A | A | U
                        #   A | A | U

                        if affected_dummy[i] == 1:
                            treat_cov1[i] = np.random.uniform(low = 0, high = 2/3, size = 1).item()
                            treat_cov2[i] = np.random.uniform(low = 0, high = 1, size = 1).item()
                        else:
                            treat_cov1[i] = np.random.uniform(low = 2/3, high = 1, size = 1).item()
                            treat_cov2[i] = np.random.uniform(low = 0, high = 1, size = 1).item()   

                variates.append((contr_cov1, treat_cov1))
                variates.append((contr_cov2, treat_cov2))

            if noise_covariates > 0:
                for c in range(noise_covariates): 
                    contr_cov = np.random.uniform(size = contr_shape[1])
                    treat_cov = np.random.uniform(size = n_units_treat)
                    variates.append((contr_cov, treat_cov))

            for c in range(len(variates)):

                contr_cov = np.array([variates[c][0] for _ in range(contr_shape[2])]) # expand across times
                contr_cov = np.array([contr_cov for _ in range(contr_shape[0])]).transpose(0, 2, 1) # expand across streams

                treat_cov = np.array([variates[c][1] for _ in range(treat_shape[2])]) # expand across times
                treat_cov = np.array([treat_cov for _ in range(treat_shape[0])]).transpose(0,2,1) # expand across streams

                cova = np.concatenate([treat_cov, contr_cov], axis = 1)

                xrda["variate" + str(c + 1)] = xr.DataArray(cova,
                                        dims = ['stream', 'unit', 'period'],
                                        coords = dict(
                                                    stream = [str(x) for x in np.arange(0,n_streams)],
                                                    unit = [str(x) for x in np.arange(0,n_units)],
                                                    period = [str(x) for x in np.arange(start_time,start_time + n_times)]
                                                )
                            )

        # Make xrda the customized xarray 'dndset' object which gives it some more methods.         
        xrda = dndset(xrda)

        # Make attributes of xrda which will help determine what functions are valid.
        xrda.attrs["binned"] = False # Will be True when variables have been discretized.
        xrda.attrs["agged"] = False # Will be True when data has been aggregated at the bin level.
        xrda.attrs["diffed"] = False # Will be True when first-differences have been taken. 
        xrda.attrs["all.periods"] = [str(i) for i in np.arange(start_time, start_time + n_times)]
        xrda.attrs["treat.periods"] = [str(i) for i in np.arange(treat_time, start_time + n_times)]

        # Returns 1: the panel, 
        #         2: binary vector for treatment status along unit dim
        #         (later, also affected streams, possibly effect size once we develop this) 
        return xrda, affected
    
def df_to_dndset(df, bin = True, diff = True, agg = True, outcome_name_col = "stream",
              binvars = None, searchvars = None, nbins = 2, ref_times = None,
              outcome_col = "outcome", unit_col = "unit", period_col = "period", 
              treat_time_col = "treat_time", treat_group_col = "group"):
    """ Takes pandas dataframe and makes into dndset. Will bin, diff and agg """
    
    if searchvars is None:
        searchvars = [i for i in df.columns if i not in [outcome_name_col, outcome_col, unit_col, period_col, treat_time_col, treat_group_col]]

    if binvars is None:
        binvars = [i for i in searchvars if not isinstance(df[i][0], str)]

    xry = df.set_index([outcome_name_col, unit_col, period_col]).to_xarray()
    xry = dndset(xry)

    # get treat times: 
    xry.attrs["treat.periods"] = [i for i in xry.attrs['all.periods'] if int(i) >= np.min([int(i) for i in df[df[treat_time_col] == 1][period_col]])]

    if bin: 
        xry = xry.binvars(varname = binvars, n_bins = nbins)
    else: 
        xry.attrs["binned"] = False
    if diff: 
        xry = xry.first_diff()
    else:
        xry.attrs["diffed"] = False
    if agg: 
        xry = xry.aggbinvars(vars2aggby = searchvars)
    else:
        xry.attrs["agged"] = False

    xry['period'] = xry.period.astype(str)

    return xry


def make_cov_matrix(matrix = None, n = 5, rhos = None, more_corr = False, desc_auto_corr = False):

    # TODO: make "rhos" argument fully functional. currently, only one and two-periods apart are 

    if more_corr and desc_auto_corr:
        RuntimeError("At least one of more_corr and desc_auto_corr must be False.")

    if rhos is None:
        rhos = [0.2, 0.1]

    sd = np.ones(n)
    sd = np.eye(n)
    
    if desc_auto_corr:
        if rhos[0] != 0: 
            for i in range(0,n):
                for j in range((i+1), n):
                    if j > i:
                        sd[i,j] = rhos[0]/((j - i))
                        sd[j,i] = rhos[0]/((j - i))
    elif more_corr:
        for i in range(0,n):
            for j in range((i+1), n):
                if (j - i >= 2) & j < n:
                    sd[i,j] = rhos[1]
                    sd[j,i] = rhos[1]
                if (j - i == 1) & (i+2 <= n):
                    sd[i,j] = rhos[0]
                    sd[j,i] = rhos[0]
    else: 
        for i in range(0,n):
            for j in range((i+1), (i+3)):
                if (j - i == 2) & (i+3 <= n):
                    sd[i,j] = rhos[1]
                    sd[j,i] = rhos[1]
                if (j - i == 1) & (i+2 <= n):
                    sd[i,j] = rhos[0]
                    sd[j,i] = rhos[0]
                #print(str(i) + ", " + str(j) + ", " + str(sd[i,j]))

    return(sd)


if __name__ == "__main__":

    # Below are settings for testing the function - delete later. 
    n_streams = 3; n_units = 20000; n_times = 2; treat_time = 1 
    start_time = 0; n_units_treat = None; distribution = "normal" 
    prop_affected = 0.1; base_mean = 0.2; base_sd = 1; treat_mean = 0.4 
    treat_sd = None; aff_mean = 0.6; aff_sd = None; rand_affected = False
    covariate_mean_diff = 1; covariate = "3x3"
    aff_periods = None; noise_covariates = 3


    #a = make_dataset(100, 100, 2, distribution="gaussian", treat_time=1, start_time=0, prop_affected=0.3, base_mean=100, aff_mean=105)
    #b = make_dataset(100, 100, 2, distribution="gaussian", treat_time=1, start_time=0, prop_affected=0.3, base_mean=100, aff_mean=100)

    # Simple mean difference demo: 

    df, aff = make_dataset(3, 10000, n_units_treat=5000, n_times = 5, start_time = -1, treat_time = 1, 
                           prop_affected=1/9, base_mean = base_mean, aff_mean=aff_mean, 
                           covariate = covariate, noise_covariates=1,
                           aff_streams = [0], distribution=distribution)

    df = df.first_diff()
    df = df.binvars(n_bins = [3, 3, 3, 3, 3], make_char = True)
    df_agg = df.aggbinvars(varname = None, outcome = "outcome")

    # get "original df":
    df = df.to_dataframe()
    df = df.reset_index()

    #a = df_agg.biased_sampling()

    df_agg.visbin2d()

    import algs.did_ss as ds

    result = ds.ltss(df_agg, ltss_func = ds.multivariate_subset_aggregation_ltss, test_side = 1)





    # DEMO GENERATING CORRELATED DATA: 

    data, _ = make_dataset(2, 100, 3, treat_mean = 2, base_sd = sd)





    # Polynomial effect demo: 
    treat_data = make_dataset(1, 100, n_times = 10, treat_time=5, distribution="gaussian", prop_affected=0.5, 
                              base_mean = "2.5 - 0.3*t + 0.05*t^2", effect_mean="-0.1*t^2 + 10 + 0.5*t")

    control_data = make_dataset(1, 100, n_times = 10, treat_time=5, distribution="gaussian", prop_affected=0.5, 
                              base_mean = "-0.3*t + 0.05*t^2")

    data = xr.concat([treat_data[0], control_data[0]], dim = "unit")

    data[0, :, :].to_pandas().to_csv('simul/sim_data/curvydata.csv')

    data.name = "curvydat"
    data.to_dataframe().to_csv('curvydata.csv')

    data = data.to_pandas()
    