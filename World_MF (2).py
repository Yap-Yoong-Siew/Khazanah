#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import tqdm

# In[2]:


data = pd.read_excel('Optimization (All).xlsx', index_col=0, parse_dates=True)
N = 13 # number of managers
K = 6 # number of managers to pick

output = pd.DataFrame(data = [], columns=['Manager 1', 'Manager 2',  'Manager 3', 'Manager 4', 'Manager 5', 'Manager 6', '10Y Ann Ret', '10Y Ann Excess Ret', '10Y Vol', '10Y Excess Vol', '10Y Sharpe', '10Y IR'])
output.loc[0] = [np.nan] * len(output.columns)

combinations = itertools.combinations(data.columns[:-3], K)


# In[3]:


for i, combo in enumerate(combinations):
    # weights = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
    num_of_managers = data[list(combo)].shape[1]
    weights = np.full((num_of_managers, 1), 1/num_of_managers)
    data['Total Portfolio Return'] = np.dot(data[list(combo)], weights)
    data['Excess Return'] = data['Total Portfolio Return'] - data['Market']
    returns = np.prod(1 + data['Total Portfolio Return']) ** (1/5) - 1
    market_ret = np.prod(1 + data['Market']) ** (1/5) - 1
    risk_free_returns = np.prod(1 + data['Rf']) ** (1/5) - 1
    excess_ret = returns - market_ret
    vol = data['Total Portfolio Return'].std() * np.sqrt(59/60) * np.sqrt(12)
    excess_vol = data['Excess Return'].std() * np.sqrt(59/60) * np.sqrt(12)
    sharpe = (returns - risk_free_returns) / vol
    ir = excess_ret / excess_vol
    
    # append results to output dataframe
    output.loc[i] = list(combo) + [returns, excess_ret, vol, excess_vol, sharpe, ir]

output


# In[4]:


output.to_csv('output.csv')


# In[28]:


output_sort = output.copy()
output_sort = output_sort.sort_values('10Y IR', ascending=False).iloc[:10]
output_sort


# In[30]:


df2 = output.copy()
string = 'EN-BlackRock'

# filter out rows that don't contain 'HO-DFA' in any of the Manager columns
df_filtered = df2[df2['Manager 1'].str.contains(string) |
                 df2['Manager 2'].str.contains(string) |
                 df2['Manager 3'].str.contains(string) |
                 df2['Manager 4'].str.contains(string) |
                 df2['Manager 5'].str.contains(string) |
                 df2['Manager 6'].str.contains(string)]


# In[33]:


# Create the Sharpe Ratio plot
plt.scatter(output['10Y Vol'], output['10Y Ann Ret'])
plt.xlabel('10Y Vol')
plt.ylabel('10Y Ret')
plt.title('10Y Absolute Risk-Return Profile')

# Highlight
plt.scatter(output_sort['10Y Vol'], output_sort['10Y Ann Ret'], color='red', marker='o')
plt.savefig('charts/top_10_absolute.png')
plt.show()

# Create the Information Ratio plot
plt.scatter(output['10Y Excess Vol'], output['10Y Ann Excess Ret'])
plt.xlabel('10Y Active Risk')
plt.ylabel('10Y Excess Ret')
plt.title('10Y Relative Risk-Return Profile')

# Highlight
plt.scatter(output_sort['10Y Excess Vol'], output_sort['10Y Ann Excess Ret'], color='red', marker='o')
plt.savefig('charts/top_10_relative.png')
plt.show()


# In[32]:


for string in data.columns[:-4]:
    df2 = output.copy()
    df_filtered = df2[df2['Manager 1'].str.contains(string) |
                 df2['Manager 2'].str.contains(string) |
                 df2['Manager 3'].str.contains(string) |
                 df2['Manager 4'].str.contains(string) |
                 df2['Manager 5'].str.contains(string) |
                 df2['Manager 6'].str.contains(string)]
    
    # Create the Sharpe Ratio plot
    plt.scatter(output['10Y Vol'], output['10Y Ann Ret'])
    plt.xlabel('10Y Vol')
    plt.ylabel('10Y Ret')
    plt.title('{} - 10Y Absolute Risk-Return Profile'.format(string))

    # Highlight
    plt.scatter(df_filtered['10Y Vol'], df_filtered['10Y Ann Ret'], color='red', marker='o')
    plt.savefig(f'charts/{string}_absolute.png')
    plt.show()

    # Create the Information Ratio plot
    plt.scatter(output['10Y Excess Vol'], output['10Y Ann Excess Ret'])
    plt.xlabel('10Y Active Risk')
    plt.ylabel('10Y Excess Ret')
    plt.title('{} - 10Y Relative Risk-Return Profile'.format(string))

    # Highlight
    plt.scatter(df_filtered['10Y Excess Vol'], df_filtered['10Y Ann Excess Ret'], color='red', marker='o')
    plt.savefig(f'charts/{string}_relative.png')
    plt.show()


# In[ ]:


# weights = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]

# manager_list = ['EN-Robeco',
#                 'EN-BlackRock',
#                 'EN-SSGA 1.0',
#                 'EN-Invesco',
#                 'HO-PanAgora',
#                 'HO-Lazard']


# data['Total Portfolio Return'] = np.dot(data[manager_list], weights)
# data['Excess Return'] = data['Total Portfolio Return'] - data['Market']

# output2 = pd.DataFrame(data = [], columns=['Manager 1', 'Manager 2',  'Manager 3', 'Manager 4', 'Manager 5', 'Manager 6', '10Y Ann Ret', '10Y Ann Excess Ret', '10Y Vol', '10Y Excess Vol', '10Y Sharpe', '10Y IR'])
# output2.loc[0] = [np.nan] * len(output2.columns)

# returns = np.prod(1 + data['Total Portfolio Return']) ** (1/5) - 1
# market_ret = np.prod(1 + data['Market']) ** (1/5) - 1
# risk_free_returns = np.prod(1 + data['Rf']) ** (1/5) - 1
# excess_ret = returns - market_ret
# vol = data['Total Portfolio Return'].std() * np.sqrt(59/60) * np.sqrt(12)
# excess_vol = data['Excess Return'].std() * np.sqrt(59/60) * np.sqrt(12)
# sharpe = (returns - risk_free_returns) / vol
# ir = excess_ret / excess_vol

# output2['Manager 1'] = manager_list[0]
# output2['Manager 2'] = manager_list[1]
# output2['Manager 3'] = manager_list[2]
# output2['Manager 4'] = manager_list[3]
# output2['Manager 5'] = manager_list[4]
# output2['Manager 6'] = manager_list[5]
# output2['10Y Ann Ret'] = returns
# output2['10Y Ann Excess Ret'] = excess_ret
# output2['10Y Vol'] = vol
# output2['10Y Excess Vol'] = excess_vol
# output2['10Y Sharpe'] = sharpe
# output2['10Y IR'] = ir
# output2


# In[ ]:




