#!/usr/bin/env python
# coding: utf-8

# # EFM Holdings Dashboard

# ## 1. Set up environment

# In[27]:


# Set up environment - import libraries, set options, define paths, filenames, import files

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from efm_functions_v2 import *
import os
from copy import deepcopy
from IPython.display import HTML

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# Set options
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 130)


# ## 2. Import files

# In[7]:


# Define file paths and file names, import files
data_path = "Data/"
output_path = "Data/Output/"
bbg_path = "Data/bbg_historical_holdings/dm_sectors/"
crd_path = 'Data/crd_daily_holdings/'

efm_dict = {'MOMG':'BlackRock US MF',
            'MOMN':'Dimensional US MF',
            'MOMP':'Invesco US MF', 
            'MOMS':'Acadian US Value',
            'MOMU':'AB US Growth',
            'MOMX':'BlackRock EU Systematic'}

# Import Data

"""
efm_holdings_data
"""
# "efm_holdings_data" will consist of dataframes from "df_list" which will be a list of DataFrames for each manager   
df_list = []
    
for filename in os.listdir(crd_path):
    df = pd.read_csv(crd_path + filename)
    df_list.append(df)

# "efm_holdings_data" will be a DataFrame (df) containing historical stock holdings of efms defined above, and will be saved as defined below as .csv
efm_holdings_data = pd.concat(df_list).set_index('Effective Date')
efm_holdings_data.index = pd.to_datetime(efm_holdings_data.index)
efm_holdings_data['SEDOL'] = efm_holdings_data['SEDOL'].astype(str)

# Save holdings data as csv file with the output being named with today's date
save_csv(efm_holdings_data, output_path, 'efm_holdings_data')


"""
stock_data
"""
# "stock_data" will contain stock data of MSCI and US and Europe as defined below
stock_data = pd.read_excel(data_path + 'MXUS EU as of Mar 24 20231.xlsx')
stock_data['SEDOL'] = stock_data['SEDOL'].astype(str)


"""
bbg_data
"""
# "bbg_data" contains stock holdings of EFMs from the PORT function on Bloomberg which only contains weekly data up to the past 40 weeks
bbg_df_list = []

# Loop through all the Excel files in the folder
for filename in os.listdir(bbg_path):
    df = pd.read_excel(bbg_path + filename)
    bbg_df_list.append(df)

# "bbg_data" will contain historical stock holdings from bloomberg
bbg_data = pd.concat(bbg_df_list)

# Save holdings data as csv file with the output being named with today's date
# save_csv(bbg_data, output_path, 'bbg_holdings_data')
print('Files successfully imported.')


# ## 3. Process Data

# In[9]:


# Merge efm_holdings_data with stock_data to update holdings data with sector and subinudstry data from stock_data
holdings_merged_1 = pd.merge(efm_holdings_data, stock_data, on='SEDOL', how='left')
holdings_merged_1.index = efm_holdings_data.index

# Count proportion of empty sectors
holdings_merged_1['Sector'].isnull().sum() / len(holdings_merged_1)

"""
bbg_processed
"""

# Process the bloomberg data to tag the sector names, as the data is separated by sector rows and does not have sector column
bbg_processed_list = []

for bbg_df in bbg_df_list:
    df1 = bbg_df.copy()
    df1 = df1.loc[:, ['Name', 'SEDOL1', 'ISIN', 'Ticker']]
    
    # find the index of the separator rows
    sector_indexes = df1[df1['Name'].str.startswith('›')].index.tolist()
    df2 = df1[~df1['Name'].str.startswith('›')]
    df2 = df2.drop(index=0)
    
    # create a list of sectors
    sectors = []
    for i in range(len(sector_indexes)):
        start = sector_indexes[i]
        if i == len(sector_indexes) - 1:
            end = len(df1)
        else:
            end = sector_indexes[i+1]
        sector_name = df1.iloc[start]['Name'][2:]
        sectors.extend([sector_name] * (end - start - 1))
    
    # add the "Sector" column to the dataframe
    df2['Sector'] = sectors
    bbg_processed_list.append(df2)
    
bbg_processed = pd.concat(bbg_processed_list)
bbg_processed = bbg_processed.rename(columns={'SEDOL1': 'SEDOL'})
bbg_processed['SEDOL'] = bbg_processed['SEDOL'].astype(str)
save_csv(bbg_processed, output_path, 'bbg_processed')

# Merge the dataframes to obtain sector and subinudstry data
bbg_processed.drop_duplicates(subset=['SEDOL'], inplace=True)
holdings_merged_2 = pd.merge(holdings_merged_1, bbg_processed, on='SEDOL', how='left')
holdings_merged_2.index = holdings_merged_1.index
holdings_merged_2['Sector_x'] = holdings_merged_2['Sector_x'].fillna(holdings_merged_2['Sector_y'])
holdings_merged_2 = holdings_merged_2.drop('Sector_y', axis=1)
holdings_merged_2 = holdings_merged_2.rename(columns={'Sector_x': 'Sector'})
holdings = holdings_merged_2
save_csv(holdings, output_path, 'holdings')

# Count proportion of empty sectors
proportion = holdings['Sector'].isnull().sum() / len(holdings)

# Replace NaN values in the 'Sector' column with 'Not Classified'
holdings['Sector'].fillna('Null', inplace=True)

# Convert values in the 'Market Value' column to negative floats if they are enclosed in parentheses
holdings['Market Value'] = holdings['Market Value'].str.replace(',', '').str.replace('(', '-', regex=False).str.replace(')', '', regex=False).astype(float)
holdings['Quantity'] = holdings['Quantity'].str.replace(',', '').str.replace('(', '-', regex=False).str.replace(')', '', regex=False).astype(float)

# Display information on processed data
fund_list = list(holdings['Account Code'].unique())
latest_date = holdings.index.max()
number_of_rows = len(holdings)
print(f'Funds: {fund_list} \n'
      f'Latest date: {latest_date:"%Y-%m-%d"} \n'
      f'Missing sectors: {proportion:.2%} \n'
      f'{number_of_rows} rows')


# In[13]:


# find the index of the separator rows
df1 = pd.read_excel('Data/bbg_dm_industry_group.xlsx')

industry_group_indexes = df1[df1['Name'].str.startswith('›')].index.tolist()
df2 = df1[~df1['Name'].str.startswith('›')]
df2 = df2.drop(index=0)

# create a list of sectors
industry_groups = []
for i in range(len(industry_group_indexes)):
    start = industry_group_indexes[i]
    if i == len(industry_group_indexes) - 1:
        end = len(df1)
    else:
        end = industry_group_indexes[i+1]
    ig_name = df1.iloc[start]['Name'][2:]
    industry_groups.extend([ig_name] * (end - start - 1))

# add the "Sector" column to the dataframe
df2['Industry Group'] = industry_groups
    
df2 = df2.rename(columns={'SEDOL1': 'SEDOL'})
df2['SEDOL'] = df2['SEDOL'].astype(str)
save_csv(df2, output_path, 'df2')

# Merge the dataframes to obtain sector and subinudstry data
df2.drop_duplicates(subset=['SEDOL'], inplace=True)
holdings_merged_2 = pd.merge(holdings, df2, on='SEDOL', how='left')
holdings_merged_2.index = holdings.index
holdings = holdings_merged_2

# Replace Industry Group values in the 'Sector' column with 'Not Classified'
holdings['Sector'].fillna('Null', inplace=True)

# Display information on processed data
fund_list = list(holdings['Account Code'].unique())
latest_date = holdings.index.max()
number_of_rows = len(holdings)
print(f'Funds: {fund_list} \n'
      f'Latest date: {latest_date:"%Y-%m-%d"} \n'
      f'Missing sectors: {proportion:.2%} \n'
      f'{number_of_rows} rows')


# In[14]:


holdings.to_csv('holdings_industry_group.csv')


# In[138]:


holdings
    

# ## 4. Calculate and plot sector weights over time

# In[28]:


industry_weights_df = industry_group_weights(holdings)


# In[145]:


plot_industry_group_weights(industry_weights_df, efm_dict, output_path)


# In[29]:


change_windows = [1, 7, 30, 60]
df_changes_list = tabulate_changes_all_ig(industry_weights_df, efm_dict, output_path, change_windows)


# In[4]:


# Calculate sector weights for each day, save into .csv file in output
sector_weights_df = sector_weights(holdings)
save_csv(sector_weights_df, output_path, 'holdings_by_sector')

# Plot sector weight changes for each fund, save as .jpgs in output
plot_sector_weights(sector_weights_df, efm_dict, output_path)


# ## 6. Tabulate Changes for selected windows

# In[5]:


change_windows = [1, 7, 30, 60]
df_changes_list = tabulate_changes_all(sector_weights_df, efm_dict, output_path, change_windows)


# ## 7. Calculate Top Holdings Changes

# In[110]:


change_windows = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                  11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                  21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
holdings_changes_dict = holdings_changes(holdings, change_windows)
top_holdings_changes_dict = top_holdings_changes(holdings_changes_dict)
top_holdings_dashboard_dict = top_holdings_dashboard(top_holdings_changes_dict, efm_dict)


# In[124]:


change_windows = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                  11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                  21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

holdings_changes_dict = stock_changes(holdings, change_windows)
holdings_changes_df = stock_changes_df(holdings_changes_dict, efm_dict)
holdings_changes_df.to_csv('holdings_changes_df_6.csv')


# In[125]:


holdings_changes_df


# In[131]:


ema_dict = holdings_ema(holdings_changes_df)
ema_dict


# In[130]:





# In[126]:


# Calculate the exponential moving average for each SEDOL across the 30 windows
ema = holdings_changes_df.groupby('SEDOL')['Quantity % Change'].apply(lambda x: x.ewm(span=30).mean())

# Create a new DataFrame with the unique SEDOLs and their respective 30-day EMAs
df_ema = pd.DataFrame({'30-Day EMA': ema})
df_ema.index.name = 'SEDOL'

df_ema


# In[117]:


holdings


# In[ ]:


for account, acct_group in holdings.groupby('Account Code'):
    

