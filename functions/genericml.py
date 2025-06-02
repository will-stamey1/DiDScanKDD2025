import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import sklearn.base
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

class genml(): 
    def __init__(self, random_state = None):#, y_learner: sklearn.base.BaseEstimator, t_learner: sklearn.base.BaseEstimator):
        """
        Initialize the GenML class.
        
        Parameters:
        - y_learner: model for the outcome process.
        - t_learner: model for the treatment process.
        - random_state: Random state for reproducibility.
        """
        # self.y_learner = y_learner
        # self.t_learner = t_learner
        # self.random_state = random_state
        self.model = None
        self.unit_name = None
        self.dataset = None
        self.te_preds = None
        self.blp_results = None
        self.random_state = random_state

    class LogisticRegression(nn.Module):
        def __init__(self, input_size):
            super(genml.LogisticRegression, self).__init__()
            self.fc1 = nn.Linear(input_size, 64)
            self.dropout = nn.Dropout(p=0.5)  
            self.fc2 = nn.Linear(64, 1)
            self.dropout = nn.Dropout(p=0.5)  

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)  # Apply dropout
            return torch.sigmoid(self.fc2(x))
        
    class FlexReg(nn.Module):
        def __init__(self, input_size):
            super(genml.FlexReg, self).__init__()
            self.fc1 = nn.Linear(input_size, 32)  # Input layer to hidden layer
            self.fcmid = nn.Linear(32, 16)
            self.fcmid2 = nn.Linear(16, 16)
            self.fc2 = nn.Linear(16, 1)   # Hidden layer to output layer

        def forward(self, x):
            x = torch.relu(self.fc1(x))  # Activation function
            x = self.fcmid(x)
            x = self.fcmid2(x)
            x = self.fc2(x)               # Output layer
            return x


    def fit(self, dataset, xnames, zname, yname = "dif", unit_name = "unit", splits = 100, n_estimators = 50, p_treat = None):

        #xnames = ["variate1", "variate2", "variate3", "variate4", "variate5"]
        #zname = "group"

        self.data = dataset
        self.yname = yname
        self.unit_name = unit_name

        unit_id = dataset[unit_name].to_numpy()

        y = dataset[yname].to_numpy()
        X = dataset[xnames].to_numpy()
        z = dataset[zname].to_numpy()

        # save y, X and z: 
        self.y = y
        self.X = X
        self.z = z

        ix = np.arange(len(y))

        if p_treat:
            
            if isinstance(p_treat, float) and p_treat > 0.0 and p_treat < 1.0:
                phat = np.ones(len(y)) * p_treat
            elif isinstance(p_treat, str):
                phat = np.array(dataset[p_treat]) 
            else: 
                RuntimeError("p_treat must be a float between 0.0 and 1.0, noninclusive.")

        else: # p_treat is none, i.e., not provided: estimate it. 
            phat = np.zeros(len(y))

            thesplit = train_test_split(ix,X,z,test_size=0.5,)#random_state=self.random_state)
            
            for i in range(2):
                xtrain = thesplit[2 + i]
                xgetp = thesplit[3 - i]
                ztrain = thesplit[4 + i]
                zgetp = thesplit[5 - i]

                theseix = thesplit[0+i]

                # transform to tensors: 
                xtrain_tensor = torch.FloatTensor(xtrain)
                xgetp_tensor = torch.FloatTensor(xgetp)
                ztrain_tensor = torch.FloatTensor(ztrain)
                zgetp_tensor = torch.FloatTensor(zgetp)

                # initialize pmodel: 
                input_size = xtrain.shape[1]
                pmodel = self.LogisticRegression(input_size)
                criterion = nn.BCELoss()
                optimizer = optim.Adam(pmodel.parameters(), lr=0.005, weight_decay=1e-4)

                num_epochs = 100
                for epoch in range(num_epochs):
                    pmodel.train()
                    outputs = pmodel(xtrain_tensor).squeeze() # pass in X data for t split
                    loss = criterion(outputs, ztrain_tensor) # predict treatment indicator 
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # if (epoch+1) % 10 == 0:
                    #     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

                pmodel.eval()
                with torch.no_grad(): # save the phat values: 
                    phat[theseix] = pmodel(xgetp_tensor).squeeze().detach().numpy()

        self.phat = phat

        # set aside list for models and a list for the prediction ixs and base and treat predictions for each split:
        self.pred_ix_big_list = []
        self.base_preds = []
        self.te_preds = []

        for sp in range(splits): # change this so that the number of times an observation is in prediction group is fairly evenly distributed and guaranteed > 1
            
            # make 2 folds: 
            if sp % 2 == 0: # this ensures that each split side gets used for both tasks:  
                kf = KFold(n_splits=2, shuffle=True)
                train_ix, pred_ix = next(kf.split(X))
            else: 
                hold = pred_ix
                pred_ix = train_ix
                train_ix = hold

            # save pred_ix for later: 
            self.pred_ix_big_list.append(pred_ix)

            # Currently, the treatment effects model is a T learner the outcome functions are learned separately for the treated 
            # and untreated groups and use their predicted difference as the treatment effect. 

            # split into A and M sets:

            Aix, Mix = unit_id[train_ix], unit_id[pred_ix]
            AX, MX = X[train_ix], X[pred_ix]
            Az, Mz = z[train_ix], z[pred_ix]
            Ay, My = y[train_ix], y[pred_ix]
            Aphat, Mphat = phat[train_ix], phat[pred_ix]

            # split by treatment/control: 
            AX_treat = AX[Az==1]
            AX_contr = AX[Az==0]
            Ay_treat = Ay[Az==1]
            Ay_contr = Ay[Az==0]

            base_model = RandomForestRegressor(n_estimators=n_estimators) #, random_state=self.random_state, max_depth=4, max_features=2)
            base_model.fit(AX_contr, Ay_contr)
            base_preds = base_model.predict(MX)
            self.base_preds.append(base_preds)

            treat_model = RandomForestRegressor(n_estimators=n_estimators) #, random_state=self.random_state, max_depth=4, max_features=2)
            treat_model.fit(AX_treat, Ay_treat)
            treat_preds = treat_model.predict(MX)

            ### NN MODELS, for later: 
            # # transform to tensors: 
            # xtreat_tensor = torch.FloatTensor(AX_treat)
            # xcontr_tensor = torch.FloatTensor(AX_contr)
            # ytreat_tensor = torch.FloatTensor(Ay_treat)
            # ycontr_tensor = torch.FloatTensor(Ay_contr)

            # MX_tensor = torch.FloatTensor(MX)

            # # initialize base model (control outcome): 
            # input_size = xtrain.shape[1]
            # y0model = FlexReg(input_size)
            # criterion = nn.MSELoss()
            # optimizer = optim.SGD(y0model.parameters(), lr=0.005)

            # num_epochs = 100
            # for epoch in range(num_epochs):
            #     y0model.train()
            #     outputs = y0model(xcontr_tensor).squeeze() # pass in X data
            #     loss = criterion(outputs, ycontr_tensor) # predict control outcome 
                
            #     optimizer.zero_grad()
            #     loss.backward()
            #     optimizer.step()

            #     if (epoch+1) % 10 == 0:
            #         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

            # y0model.eval()
            # with torch.no_grad(): # save the phat values: 
            #     base_preds = y0model(MX_tensor).squeeze().detach().numpy()

            # # initialize treated outcome model: 
            # input_size = xtrain.shape[1]
            # y1model = FlexReg(input_size)
            # criterion = nn.MSELoss()
            # optimizer = optim.SGD(y1model.parameters(), lr=0.005)

            # num_epochs = 100
            # for epoch in range(num_epochs):
            #     y1model.train()
            #     outputs = y1model(xtreat_tensor).squeeze() # pass in X data
            #     loss = criterion(outputs, ytreat_tensor) # predict control outcome 
                
            #     optimizer.zero_grad()
            #     loss.backward()
            #     optimizer.step()

            #     if (epoch+1) % 10 == 0:
            #         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

            # y1model.eval()
            # with torch.no_grad(): # save the phat values: 
            #     treat_preds = y1model(MX_tensor).squeeze().detach().numpy()

            # calculate treatment effect predictions and save them to self: 
            te_preds = treat_preds - base_preds
            self.te_preds.append(te_preds)

        # get average prediction across splits for each individual to determine quantile group membership: 
        # total number of unique units: 
        n_units = len(np.unique(self.data.unit))

        all_preds = []
        for ix, preds in zip(self.pred_ix_big_list, self.te_preds):
            arr = np.full(n_units, np.nan)
            arr[ix] = preds
            all_preds.append(arr)

        mean_pred = np.nanmean(np.array(all_preds), axis = 0)
        self.data['mean_pred'] = mean_pred

        return None 

    def BLP(self):
        """ 
        Evaluate BLP of the CATE on the proxy
        """

        blps = []

        for j in range(len(self.pred_ix_big_list)): # iterate through list of fit results: 

            # get pred_group data:
            te_preds = self.te_preds[j]
            base_preds = self.base_preds[j]
            pred_ix = self.pred_ix_big_list[j]

            Mphat = self.phat[pred_ix]
            My = self.y[pred_ix]
            Mz = self.z[pred_ix]

            # make blp components: 
            newX = np.hstack((np.ones(My.shape).reshape(-1,1), base_preds.reshape(-1,1)))
            H = ((Mz - Mphat)/(Mphat * (1 - Mphat)))
            YH = My * H
            DminP = (Mz - Mphat).reshape(-1,1)
            Si_ES = te_preds - np.mean(te_preds)

            # BLP1: 
            RHS = np.hstack((newX, DminP, DminP * Si_ES.reshape(-1,1)))
            blpmod = LinearRegression()
            blpmod = blpmod.fit(RHS, My)
            # BLP 2: 

            # RHS = np.hstack((X*H.reshape(-1,1), np.ones(My.shape).reshape(-1,1), Si_ES.reshape(-1,1)))
            # blpmod = LinearRegression()
            # blpmod = blpmod.fit(RHS, YH)

            blps.append(blpmod.coef_)

        self.blp_results = np.array(blps).mean(axis = 0)
        return self.blp_results

    def GATE(self, k = 5, quantile_range = None, return_group = False):
        """
        Gets GATE for specified quantile group. Also, if return_set == True, returns the list of observations in the specified quantile range. 

        k: number of quantile subgroups
        quantile_range: If a list with two numbers between 0 and 1 are specified, the specific GATE and group set for that quantile span will 
                be returned INSTEAD of the k groups. 
        """

        # set aside list for GATE for each split: 
        gate_list = []

        if quantile_range: 

            # get group membership based on mean predictions across splits: 
            quantvals = np.nanquantile(self.data['mean_pred'], quantile_range)
            memberships = np.digitize(self.data['mean_pred'], quantvals, right=True)
            
            ingroup = np.multiply(self.data['mean_pred'] > quantvals[0], self.data['mean_pred'] < quantvals[1])
            groups = self.data[self.unit_name][ingroup]

            for j in range(len(self.pred_ix_big_list)): # iterate through list of fit results: 

                # get pred set data: 
                te_preds = self.te_preds[j]
                
                # who's in the group? 
                group_true = np.multiply((np.quantile(te_preds, quantile_range[0]) <= te_preds), (np.quantile(te_preds, quantile_range[1]) > te_preds))

                # get this GATE and save it
                gate_list.append(np.mean(te_preds[group_true]))

            gates = np.array(gate_list).mean(axis = 0)


        else: 

            # get group membership based on mean predictions across splits: 
            quantiles = np.linspace(1/k, 1-1/k, k-1)
            quantvals = np.nanquantile(self.data['mean_pred'], quantiles)
            memberships = np.digitize(self.data['mean_pred'], quantvals, right=True)
            groups = []
            for i in np.unique(memberships):
                g = np.where(memberships == i)
                groups.append(np.array(self.data[self.unit_name][g[0]]))
            
            # iterate across splits
            for j in range(len(self.pred_ix_big_list)): # iterate through list of fit results: 

                # get pred set data: 
                te_preds = self.te_preds[j]
                base_preds = self.base_preds[j]
                pred_ix = self.pred_ix_big_list[j]

                Mphat = self.phat[pred_ix]
                My = self.y[pred_ix]
                Mz = self.z[pred_ix]
                newX = np.hstack((np.ones(My.shape).reshape(-1,1), base_preds.reshape(-1,1)))
                H = ((Mz - Mphat)/(Mphat * (1 - Mphat)))
                YH = My * H
                DminP = (Mz - Mphat).reshape(-1,1)

                quintiles = np.quantile(te_preds, np.linspace(1/k, 1 - 1/k, (k-1)))
                quintdummies = []
                for i in range(k):
                    if i == 0:
                        dummy = (te_preds <= quintiles[i])  # First quintile
                    elif i == (k-1):
                        dummy = (te_preds > quintiles[i-1])  # top quintile
                    else:
                        dummy = (te_preds > quintiles[i-1]) & (te_preds <= quintiles[i])  # Middle quintiles
                    quintdummies.append(dummy.astype(int))  # Convert boolean to int (0 or 1)
                quintdummies = np.transpose(np.array(quintdummies))                

                RHS = np.hstack((base_preds.reshape(-1,1), np.array(newX), np.array(DminP) * quintdummies))
                blpmod = LinearRegression()
                blpmod = blpmod.fit(RHS, My)
                gate_list.append(blpmod.coef_[-5:])

            gates = np.array(gate_list).mean(axis = 0)
        
        if return_group:
            return gates, groups
        else: 
            return gates
    
    def CLAN(self, quantile_range = [0.8, 1], return_set = False): # TODO: unfinished
        """
        Gets mean vector of covariates for specified quantile group. Also, if return_set == True, returns the list of observations in the specified quantile range. 
        """

        # CLAN: 
        avg_X_top = np.dot(np.transpose(MX), quintdummies[:,4]) / np.sum(quintdummies[:,4])
        avg_X_bottom = np.dot(np.transpose(MX), quintdummies[:,0]) / np.sum(quintdummies[:,0])

        return None

    def vis_effect(self, xvar):
        plt.scatter(self.data[xvar], self.data.mean_pred.to_numpy())
        plt.show()



if __name__ == "__main__":

    # DEMO: 

    import simul.make_data.make_data as md
    xr_data, _ = md.make_dataset(n_streams=1, n_units = 10000, n_times = 2, base_mean=0, treat_mean=0, aff_mean=20, covariate="corner2d", noise_covariates=0,
                             base_sd = 1, prop_affected = 4/9)

    # preprocess data: 
    xr_data = xr_data.binvars()
    xr_data = xr_data.first_diff()
    df = xr_data.to_dataframe().reset_index()
    df = df[df['period']=='1']
    df['group'] = df['group'].replace({"A": '1', "T": '1', "C": '0'}).astype(int)


    # Run genml and get GATE: 
    gml_mod = genml() # initialize gml object
    gml_mod.fit(df, ['variate1', 'variate2'], zname = "group", splits=20, n_estimators=100)
    gate, groups = gml_mod.GATE(k = 5, return_group = True)
    gml_mod.vis_effect(xvar = "variate1")