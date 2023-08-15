#!/usr/bin/env python
# coding: utf-8

# # EFM Holdings Dashboard

# ## 1. Set up environment

# In[1]:


# Set up environment - import libraries, set options, define paths, filenames, import files

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from efm_functions import *
import os
from copy import deepcopy
from IPython.display import HTML
from sklearn.metrics.pairwise import cosine_similarity
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# Set options
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 130)


# ## 2. Import files

# In[2]:


# Define file paths and file names, import files
data_path = "Data/"
output_path = "Data/Output/"
bbg_path = "Data/bbg_historical_holdings/dm_sectors/"
crd_path = 'Data/crd_daily_holdings_new/'
market_regime_path = 'Data/market_regime_invesco.csv'
benchmark_path = 'Data/Benchmark Weights/MXUS/'



#%%

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
    if not filename.endswith(".zip"):
        df = pd.read_csv(crd_path + filename)
        
        df['Effective Date'] = pd.to_datetime(df['Effective Date'])
        df = df.sort_values(by='Effective Date')
        
        if filename == 'momp_si_20230605.csv':
            df = df[df['Effective Date'] >= '2022-02-11']
            
        elif filename == 'momu_si_20230605.csv':
            df = df[df['Effective Date'] >= '2022-12-21']
            
        elif filename == 'momx_si_20230605.csv':
            df = df[df['Effective Date'] >= '2022-12-30']
            
        elif filename == 'momg_si_20230605.csv':
            df = df[df['Effective Date'] >= '2021-12-01']
        
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
    if not filename.endswith(".zip"):
        df = pd.read_excel(bbg_path + filename)
        bbg_df_list.append(df)

# "bbg_data" will contain historical stock holdings from bloomberg
bbg_data = pd.concat(bbg_df_list)

# Save holdings data as csv file with the output being named with today's date
# save_csv(bbg_data, output_path, 'bbg_holdings_data')
print('Files successfully imported.')



# ## 3. Process Data

# In[3]:


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


# In[4]:


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

# Replace NaN values in the 'Sector' column with 'Not Classified'
holdings['Sector'].fillna('Null', inplace=True)

# Display information on processed data
fund_list = list(holdings['Account Code'].unique())
latest_date = holdings.index.max()
number_of_rows = len(holdings)
print(f'Funds: {fund_list} \n'
      f'Latest date: {latest_date:"%Y-%m-%d"} \n'
      f'Missing sectors: {proportion:.2%} \n'
      f'{number_of_rows} rows')



# holdings.to_csv('holdings_industry_group.csv')


# ## 4. Calculate and plot sector weights over time

# In[6]:


industry_weights_df = industry_group_weights(holdings)




# In[8]:


plot_industry_group_weights(industry_weights_df, efm_dict, output_path)


#%% process 2018 to 2021 data

df = pd.read_excel("Data//Invesco OMFL_Month Ends.xlsx")

df_sector_weights = read_invesco_data_to_sectors(df, "Sector")

#%% process benchmark data 2021 to 2023


data_list = []

for subdirectories in os.listdir(benchmark_path):
    year_path =  os.path.join(benchmark_path, subdirectories)
    files_for_year = os.listdir(year_path)
    
    sector_weights_year = pd.Series(dtype=float)
    for file in files_for_year:
        file_path = os.path.join(year_path, file)
        date_str = file.split(" as of ")[1].split(".xlsx")[0]
        date_str = date_str[:-1]  # Remove the last character (extra "1")
        # Convert to datetime object
        try:
            date_obj = pd.to_datetime(date_str)
        except:
            continue
        # Process the file
        sector_weights_series = process_benchmark_files(file_path)
        for sector, weight in sector_weights_series.items():
            data_list.append([date_obj, sector, weight])
# Create a DataFrame with 'Date' as the index
benchmark_df = pd.DataFrame(data_list, columns=['Date', 'Sector', 'Sector Weight'])
benchmark_df.set_index('Date', inplace=True)
benchmark_df['Sector'].replace('--', 'Null', inplace=True)
benchmark_df.to_csv("benchmark_df.csv")




#%%
# Calculate sector weights for each day, save into .csv file in output
sector_weights_df = sector_weights(holdings)
sector_weights_df = pd.concat([df_sector_weights, sector_weights_df], axis=0)
sector_weights_df.index.name = 'Date'
# save_csv(sector_weights_df, output_path, 'holdings_by_sector')

# Plot sector weight changes for each fund, save as .jpgs in output
plot_sector_weights(sector_weights_df, efm_dict, output_path)

#%% market regime plot without subtract benchmark

market_regime = pd.read_csv(market_regime_path)
market_regime['Date'] = pd.to_datetime(market_regime['Date'], format="%Y %B")
market_regime.sort_values('Date', inplace=True)
market_regime.set_index('Date', inplace=True)

colors = {
    'Recovery': 'lightblue',
    'Expansion': 'lightgreen',
    'Slowdown': 'plum',
    'Contraction': 'lightcoral'
}
plot_sector_weights_with_regimes(sector_weights_df, efm_dict, market_regime, colors, output_path)

#%% market regime plot WITH subtract benchmark
sector_weights_df = sector_weights_df[sector_weights_df['Account Code'] == 'MOMP']

sector_weights_df = sector_weights_df.drop(['Account Code'], axis=1)
sector_weights_df.to_csv("invesco_sector_weights.csv")
#%% create df for active weight

df_invesco = pd.read_csv("invesco_sector_weights.csv", parse_dates=['Date'])
df_invesco = df_invesco[552: 4758]
df_benchmark = pd.read_csv("benchmark_df.csv", parse_dates=['Date'])
df_active = df_invesco.copy()

def subtract_weights(row):
    matching_value = df_benchmark.loc[
        (df_benchmark['Date'].dt.month == row['Date'].month) &
        (df_benchmark['Date'].dt.year == row['Date'].year) &
        (df_benchmark['Sector'] == row['Sector']), 
        'Sector Weight'
    ]
    
    if row['Sector'] == "Not Applicable":
        return row['Sector Weight'] - 0
    elif matching_value.size > 0:
        return row['Sector Weight'] - matching_value.values[0]
    else:
        return row['Sector Weight']

df_active['Sector Weight'] = df_invesco.apply(subtract_weights, axis=1)
df_active.set_index('Date', inplace=True)
sector_weights_df = df_active
plot_active_sector_weights(sector_weights_df, output_path)
plot_invesco_activeweights_with_regimes(sector_weights_df, market_regime, colors, output_path)

#%%

# # Filter sector_weights_df to match the date range of benchmark_df
# sector_weights_df = sector_weights_df[sector_weights_df['Date'].isin(benchmark_df['Date'])]

# # Merge the benchmark and invesco dataframes on Date and Sector
# merged_df = pd.merge(sector_weights_df, benchmark_df, on=['Date', 'Sector'], how='left', suffixes=('_invesco', '_benchmark'))

# # Calculate the difference between invesco and benchmark weights
# merged_df['Active Weight'] = merged_df['Sector Weight_invesco'] - merged_df['Sector Weight_benchmark'].fillna(0)

# # Extract the relevant columns to form the active_weights_df
# active_weights_df = merged_df[['Date', 'Sector', 'Active Weight']]

# active_weights_df.head()



#%% PREPROCESSING FOR regime similarity score using cosines similarity

def fill_missing_elements(row):
    missing_elements = np.setdiff1d(np.arange(1, 14), row)
    row = np.concatenate([row, missing_elements])
    return row


market_regime.index = pd.to_datetime(market_regime.index)

# sector_regime_df = pd.merge_asof(sector_weights_df.sort_index(), market_regime.sort_index(), 
#                             left_index=True, right_index=True, direction='backward') do this line later

sector_weights_df = sector_weights_df.groupby(sector_weights_df.index).apply(lambda x: x.sort_values(by='Sector Weight', ascending=False))

sector_mapping = {
    'Utilities': 1,
    'Null': 2,
    'Communication Services': 3,
    'Consumer Discretionary': 4,
    'Consumer Staples': 5,
    'Energy': 6,
    'Financials': 7,
    'Health Care': 8,
    'Industrials': 9,
    'Materials': 10,
    'Information Technology': 11,
    'Real Estate': 12,
    'Not Classified' : 13
}

sector_weights_df['Sector'] = pd.Categorical(sector_weights_df['Sector'], categories=sector_mapping.keys()).codes + 1
sector_weights_df = sector_weights_df.groupby(level=0)['Sector'].apply(lambda x: x.to_numpy())
sector_weights_df = pd.merge_asof(sector_weights_df.sort_index(), market_regime.sort_index(), 
                            left_index=True, right_index=True, direction='backward')
sector_weights_df['Sector'] = sector_weights_df['Sector'].apply(fill_missing_elements)

#%% IMPLEMENTATION FOR regime similarity score using cosines similarity 


# Define the cosine similarity calculator function
def cosine_similarity_calculator(input_vector, dataframe, n):
    # Ensure the input_vector is a numpy array
    input_vector = np.array(input_vector)
    
    # Group vectors by regime
    grouped = dataframe.groupby('Regime')
    
    # For each regime
    similarities = {}
    for name, group in grouped:
        sliced_input_vector = input_vector[:n]
        sliced_group = group['Sector'].apply(lambda x: x[:n])
        # Calculate cosine similarity between input vector and each vector in the group
        group_similarities = sliced_group.apply(lambda x: cosine_similarity([sliced_input_vector], [x])[0,0])
        # Compute average cosine similarity for the group
        avg_similarity = group_similarities.mean()
        similarities[name] = avg_similarity

    # Return the regime with the highest average cosine similarity
    predicted_regime = max(similarities, key=similarities.get)
    return predicted_regime, similarities

sector_weights_df['Sector'] = sector_weights_df['Sector'].apply(np.array)
n=13
recovery_vector = [ 7, 10,  9,  4, 13,  1, 12 , 6,  5 , 3 , 8 ,11,  2]

predicted_regime, similarities = cosine_similarity_calculator(recovery_vector, sector_weights_df, n)
print("testing for recovery")

print(predicted_regime)
print(similarities)

expansion_vector = [ 7, 10,  2,  9, 12,  4 , 6 , 1, 13 , 5,  3,  8, 11]

predicted_regime, similarities = cosine_similarity_calculator(expansion_vector, sector_weights_df, n)
print("testing for expansion")

print(predicted_regime)
print(similarities)

slowdown_vector = [ 8 , 5 , 7,  6,  2, 13,  4 , 9 , 3 ,12, 10 , 1 ,11]

predicted_regime, similarities = cosine_similarity_calculator(slowdown_vector, sector_weights_df, n)
print("testing for slowdown")

print(predicted_regime)
print(similarities)

contraction_vector = [ 8,  6,  5 , 7, 13, 10 , 1 , 9 ,12 , 4,  3, 11 , 2]
contraction_vector = [ 8,  6,  5 , 2, 1,  13, 12, 10,  3 , 9 , 7 , 4 ,11]
predicted_regime, similarities = cosine_similarity_calculator(contraction_vector, sector_weights_df, n)
print("testing for contraction")
print(predicted_regime)
print(similarities)
# In[11]:


change_windows = [1, 7, 30, 60] # days from current
df_changes_list = tabulate_changes_all(sector_weights_df, efm_dict, output_path, change_windows)


# ## 7. Calculate Top Holdings Changes

# In[12]:


change_windows = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                  11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                  21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
holdings_changes_dict = holdings_changes(holdings, change_windows)
top_holdings_changes_dict = top_holdings_changes(holdings_changes_dict)
top_holdings_dashboard_dict = top_holdings_dashboard(top_holdings_changes_dict, efm_dict)


# In[ ]:


change_windows = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                  11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                  21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

holdings_changes_dict = stock_changes(holdings, change_windows)
holdings_changes_df = stock_changes_df(holdings_changes_dict, efm_dict)
holdings_changes_df.to_csv('holdings_changes_df.csv')


# In[ ]:


holdings_changes_df


# In[ ]:


ema_dict = holdings_ema(holdings_changes_df)
ema_dict


# In[ ]:





# In[ ]:


# Calculate the exponential moving average for each SEDOL across the 30 windows
ema = holdings_changes_df.groupby('SEDOL')['Quantity % Change'].apply(lambda x: x.ewm(span=30).mean())

# Create a new DataFrame with the unique SEDOLs and their respective 30-day EMAs
df_ema = pd.DataFrame({'30-Day EMA': ema})
df_ema.index.name = 'SEDOL'

df_ema

