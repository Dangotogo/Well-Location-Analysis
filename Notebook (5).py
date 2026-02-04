#!/usr/bin/env python
# coding: utf-8

# 
# # OilyGiant Mining Company: Well Location Analysis
# 
# **Project Goal:**  
# Determine the best location for a new oil well to maximize profit.
# 
# ---
# 
# ## Steps to Choose the Location
# 1. Collect the oil well parameters in the selected regions (oil quality and volume of reserves).  
# 2. Build a model for predicting the volume of reserves in new wells.  
# 3. Select the oil wells with the highest estimated values.  
# 4. Pick the region with the highest total profit for the selected wells.  
# 
# ---
# 
# ## Analysis Approach
# - Build and train a predictive model to estimate reserves.  
# - Evaluate potential profit in each region.  
# - Assess risks and uncertainty using the **Bootstrapping technique**.  
# 
# ---
# 
# ## Expected Outcome
# Identify the region with the **highest profit margin** and **lowest risk** for drilling new wells.

# ### Import Libraries

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[2]:


#data preprocessing

data0 = pd.read_csv('/Users/hillary/Downloads/geo_data_0.csv')
data1 = pd.read_csv('/Users/hillary/Downloads/geo_data_1.csv')
data2 = pd.read_csv('/Users/hillary/Downloads/geo_data_2.csv')

display(data0)
display(data1)
display(data2)

#print(data0[data0['id'].duplicated()].value_counts() )
data0[data0['id'] == 'HZww2']
#keeping the duplicated id variable since they hold different values

print(data0.isna().sum())
print(data1.isna().sum())
print(data0.isna().sum())


# ## 2. Train data for each region
# ### 2.1 Split data into training and validation set 

# In[3]:


data = {'region_0': data0, 'region_1': data1, 'region_2': data2}

#splitting data to 75/25 for train/valid set 
data0_temp, data0_test = train_test_split(data0, test_size = 0.20, random_state = 12345)
data0_train, data0_valid = train_test_split(data0, test_size = 0.25, random_state = 12345)

data1_temp, data1_test = train_test_split(data1, test_size = 0.20, random_state = 12345)
data1_train, data1_valid = train_test_split(data1, test_size = 0.25, random_state = 12345)

data2_temp, data2_test = train_test_split(data2, test_size = 0.20, random_state = 12345)
data2_train, data2_valid = train_test_split(data2, test_size = 0.25, random_state = 12345)


#setting features and target 
# Region 0
data0_temp, data0_test  = train_test_split(data0, test_size=0.20, random_state=12345)
data0_train, data0_valid = train_test_split(data0_temp, test_size=0.25, random_state=12345)

features0_train = data0_train.drop(['product','id'], axis=1);  target0_train = data0_train['product']
features0_valid = data0_valid.drop(['product','id'], axis=1);  target0_valid = data0_valid['product']
features0_test  = data0_test.drop(['product','id'],  axis=1);  target0_test  = data0_test['product']

# Region 1
data1_temp, data1_test  = train_test_split(data1, test_size=0.20, random_state=12345)
data1_train, data1_valid = train_test_split(data1_temp, test_size=0.25, random_state=12345)

features1_train = data1_train.drop(['product','id'], axis=1);  target1_train = data1_train['product']
features1_valid = data1_valid.drop(['product','id'], axis=1);  target1_valid = data1_valid['product']
features1_test  = data1_test.drop(['product','id'],  axis=1);  target1_test  = data1_test['product']

# Region 2
data2_temp, data2_test  = train_test_split(data2, test_size=0.20, random_state=12345)
data2_train, data2_valid = train_test_split(data2_temp, test_size=0.25, random_state=12345)

features2_train = data2_train.drop(['product','id'], axis=1);  target2_train = data2_train['product']
features2_valid = data2_valid.drop(['product','id'], axis=1);  target2_valid = data2_valid['product']
features2_test  = data2_test.drop(['product','id'],  axis=1);  target2_test  = data2_test['product']


# ### 2.2 Train model and make prediction for training set

# In[4]:


model0 = LinearRegression().fit(features0_train, target0_train)
preds0_valid = model0.predict(features0_valid)

model1 = LinearRegression().fit(features1_train, target1_train)
preds1_valid = model1.predict(features1_valid)

model2 = LinearRegression().fit(features2_train, target2_train)
preds2_valid = model2.predict(features2_valid)

# Convert predictions to Series to use .nlargest and keep indexes aligned after reset
preds0_valid_s = pd.Series(preds0_valid, name='pred').reset_index(drop=True)
preds1_valid_s = pd.Series(preds1_valid, name='pred').reset_index(drop=True)
preds2_valid_s = pd.Series(preds2_valid, name='pred').reset_index(drop=True)

target0_valid_s = target0_valid.reset_index(drop=True)
target1_valid_s = target1_valid.reset_index(drop=True)
target2_valid_s = target2_valid.reset_index(drop=True)


# ### 2.3. Save the predictions and correct answers for the validation set.

# In[5]:


rmse = {
    'region_0': mean_squared_error(target0_valid, preds0_valid, squared=False),
    'region_1': mean_squared_error(target1_valid, preds1_valid, squared=False),
    'region_2': mean_squared_error(target2_valid, preds2_valid, squared=False),
}

mean_true = {
    'region_0': float(target0_valid.mean()),
    'region_1': float(target1_valid.mean()),
    'region_2': float(target2_valid.mean()),
}

mean_pred = {
    'region_0': float(preds0_valid_s.mean()),
    'region_1': float(preds1_valid_s.mean()),
    'region_2': float(preds2_valid_s.mean()),
}


# ### 2.4. Print the average volume of predicted reserves and model RMSE.

# In[6]:


print('RMSE values for each reagion:')
print(rmse)


# The results show that Model Set 1 achieves near perfect performance with an R² of 0.9996 and extremely low MSE on both the training and validation sets. 
# 
# While this suggests very strong predictive power, the results appear unusually high and may indicate potential data leakage, where the target variable or a closely related feature is influencing the model. 
# 
# In contrast, Model Sets 0 and 2 produce more realistic results with R² values around 0.20–0.28 and higher MSE (1400–1600), suggesting that a simple linear regression struggles to capture the underlying patterns in the data. Between these two, Set 0 performs slightly better. 
# 
# Overall, the findings highlight the need to carefully check for leakage in Set 1 and to choosing set 0 as the current best model since it showed lower MSE and higher R².

# ## 3. Prepare for profit calculation
# **Conditions**:
# - Budget for 200 wells: **$100M**
# - Revenue per unit (thousand barrels): **$4500**
# - Explore **500** wells; pick **200** best by predicted reserves
# 
# **Break-even threshold** (thousand barrels per selected well):
# 

# In[7]:


BUDGET = 100_000_000        # $
REVENUE_PER_UNIT = 4_500    # $ per thousand barrels
EXPLORED_PER_REGION = 500   # explore 500
SELECTED_PER_REGION = 200   # pick 200

break_even_volume = BUDGET / (SELECTED_PER_REGION * REVENUE_PER_UNIT)  # thousand bbl per well
print("\nMeans & RMSE per region:")
for name in data.keys():
    print(f"{name:>8} | mean_true={mean_true[name]:6.2f} | mean_pred={mean_pred[name]:6.2f} | RMSE={rmse[name]:6.2f}")

print(f"\nBreak-even volume per selected well (thousand bbl): {break_even_volume:.3f}")


# The model results show clear differences in prediction quality across the three regions. Region 1 stands out with a mean true value of 68.79 thousand barrels, closely matched by the mean prediction of 68.79 and a very low RMSE of just 0.89, indicating highly accurate predictions. In contrast, Regions 0 and 2 have higher average reserves (92.58 and 95.18 thousand barrels respectively), but their RMSE values are much larger (37.84 and 40.13), reflecting less precise estimates. Importantly, the break-even threshold is 111.1 thousand barrels per well, which is above the average reserve levels for all three regions, meaning that on average, none of the regions reach profitability per well without selecting the very best-performing sites. This highlights the importance of careful well selection rather than relying on average reserves.

# ## 4. Write a function to calculate profit from a set of selected oil wells and model predictions:

# In[11]:


def profit_from_selection(y_true: pd.Series,
                          y_pred: pd.Series,
                          budget: float = BUDGET,
                          rev: float = REVENUE_PER_UNIT,
                          k: int = SELECTED_PER_REGION) -> float:
    """Pick top-k by prediction, sum TRUE product, compute revenue - budget."""
    top_idx = y_pred.nlargest(k).index
    total_product_k = y_true.loc[top_idx].sum()  # thousand barrels
    return total_product_k * rev - budget

print("\nProfit on full validation (sanity check):")
print(f"{'region_0':>8} | profit = {profit_from_selection(target0_valid_s, preds0_valid_s):,.0f} USD")
print(f"{'region_1':>8} | profit = {profit_from_selection(target1_valid_s, preds1_valid_s):,.0f} USD")
print(f"{'region_2':>8} | profit = {profit_from_selection(target2_valid_s, preds2_valid_s):,.0f} USD")


# From the profit analysis on the validation set, all three regions show the potential to generate positive returns, but at different levels. Region 0 leads with an estimated profit of about 28.3 million, followed by Region 1 at 24.1 million and Region 2 at $23.3 million. Even though the margins are relatively close, Region 0 demonstrates the highest profitability, suggesting it may be the most promising candidate for development. However, since these results are based on model predictions, further evaluation with risk analysis is needed before making a final decision.

# ## 5. Bootstrapped risks and profit per region
# We run **1000 bootstrap iterations**. In each iteration:
# 1. Randomly sample 500 wells from the validation set.
# 2. Select the top-200 by predicted reserves.
# 3. Compute profit using actual reserves.
# 
# We then compute the distribution of profit, the mean, the 95% confidence interval, and the risk of losses (share of iterations with negative profit).

# In[9]:


def bootstrap_profit(y_true: pd.Series,
                     y_pred: pd.Series,
                     n_runs: int = 1000,
                     explored: int = EXPLORED_PER_REGION,
                     k: int = SELECTED_PER_REGION,
                     budget: float = BUDGET,
                     rev: float = REVENUE_PER_UNIT,
                     seed: int = 12345):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    profits = np.empty(n_runs, dtype=float)
    for i in range(n_runs):
        idx = rng.integers(0, n, explored)  # sample 500 with replacement
        s_true = y_true.iloc[idx].reset_index(drop=True)
        s_pred = y_pred.iloc[idx].reset_index(drop=True)
        profits[i] = profit_from_selection(s_true, s_pred, budget, rev, k)
    mean_p = float(profits.mean())
    ci_low, ci_high = np.quantile(profits, [0.025, 0.975])
    loss_risk = float((profits < 0).mean())
    return mean_p, ci_low, ci_high, loss_risk

print("\nBootstrap results:")
res0 = bootstrap_profit(target0_valid_s, preds0_valid_s)
res1 = bootstrap_profit(target1_valid_s, preds1_valid_s)
res2 = bootstrap_profit(target2_valid_s, preds2_valid_s)
for name, (mean_p, lo, hi, risk) in {
    'region_0': res0,
    'region_1': res1,
    'region_2': res2
}.items():
    print(f"{name:>8} | mean={mean_p:,.0f} USD | 95% CI=[{lo:,.0f}, {hi:,.0f}] | loss risk={risk*100:.2f}%")


# Looking at the bootstrap results, Region 1 appears to be the strongest option when considering both profitability and risk. While all three regions have fairly similar mean profits (around $3.8M–$4.3M), Region 1 has the lowest loss risk at 2.2%, which is below the 2.5% threshold set by the project requirements. In comparison, Region 0 and Region 2 show higher risks of 5.0% and 8.3%, respectively, meaning they are less reliable despite their competitive average profits. The confidence intervals also show a wide range of possible outcomes, which highlights the uncertainty of the predictions, but Region 1’s balance of profitability and acceptable risk makes it the best candidate for development.

# In[14]:


res0 = bootstrap_profit(target0_valid_s, preds0_valid_s)
res1 = bootstrap_profit(target1_valid_s, preds1_valid_s)
res2 = bootstrap_profit(target2_valid_s, preds2_valid_s)

# Choose region per rule: risk < 2.5%, then highest mean profit
eligible = []
for name, r in zip(['region_0','region_1','region_2'], [res0, res1, res2]):
    mean_p, lo, hi, risk = r
    if risk < 0.025:
        eligible.append((name, mean_p, risk))
if eligible:
    best = sorted(eligible, key=lambda x: x[1], reverse=True)[0]
    print(f"\nRecommended region: {best[0]} (mean profit={best[1]:,.0f} USD; loss risk={best[2]*100:.2f}%)")
else:
    print("\nNo region meets the loss-risk threshold (< 2.5%).")


# Based on the bootstrap analysis, Region 1 comes out as the recommended choice for development. While its average profit of around $4.1 million is lower compared to the earlier validation profits, it maintains a low risk of loss at only 2.20%, which meets the project’s risk threshold of 2.5%. This balance between profitability and stability makes Region 1 a safer investment compared to the other regions, which showed higher risks of negative returns. As a college student analyzing these results, I see that this highlights the importance of considering not just the raw profit numbers but also the uncertainty and potential risks when making business decisions.
