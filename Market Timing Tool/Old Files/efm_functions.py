import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os

def save_csv(df, output_path, name):
    today = datetime.datetime.now().strftime('%Y_%m_%d_%H%M_%S')
    df.to_csv(output_path + name + '_' + today + '.csv')
    
def format_pct(val):
    return '{:.2%}'.format(val)

def get_country(ticker):
    if isinstance(ticker, str) and 'UW Equity' in ticker:
        return 'United States'
    else:
        return 'Europe'

def sector_weights(holdings):
    sector_weights_dict = {}
    for date, group in holdings.groupby(level=0):
        #print(f'Processing: {date}')
        sector_weights_dict[date] = {}
        for account, acct_group in group.groupby('Account Code'):
            sector_weights_dict[date][account] = {}
            account_sum = acct_group['Market Value'].sum()
            for sector, sector_group in acct_group.groupby('Sector'):
                sector_weight = sector_group['Market Value'].sum() / account_sum if account_sum != 0 else 0
                sector_weights_dict[date][account][sector] = sector_weight
    
    sector_weights_df = pd.DataFrame(
        [(date, account, sector, sector_weights_dict[date][account][sector]) 
         for date in sector_weights_dict 
         for account in sector_weights_dict[date] 
         for sector in sector_weights_dict[date][account]],
         columns=['Effective Date', 'Account Code', 'Sector', 'Sector Weight']).set_index('Effective Date')

    return sector_weights_df

def industry_group_weights(holdings):
    sector_weights_dict = {}
    for date, group in holdings.groupby(level=0):
        #print(f'Processing: {date}')
        sector_weights_dict[date] = {}
        for account, acct_group in group.groupby('Account Code'):
            sector_weights_dict[date][account] = {}
            account_sum = acct_group['Market Value'].sum()
            for sector, sector_group in acct_group.groupby('Industry Group'):
                sector_weight = sector_group['Market Value'].sum() / account_sum if account_sum != 0 else 0
                sector_weights_dict[date][account][sector] = sector_weight
    
    sector_weights_df = pd.DataFrame(
        [(date, account, sector, sector_weights_dict[date][account][sector]) 
         for date in sector_weights_dict 
         for account in sector_weights_dict[date] 
         for sector in sector_weights_dict[date][account]],
         columns=['Effective Date', 'Account Code', 'Industry Group', 'Industry Group Weight']).set_index('Effective Date')

    return sector_weights_df

def plot_sector_weights(sector_weights_df, efm_dict, output_path):
    
    df_filtered_dict = {}
    df_fund_sectors_list = []
    
    for efm in efm_dict.keys():
        
        df_filtered = sector_weights_df.loc[(sector_weights_df['Account Code'] == efm)]
        df_filtered.groupby('Sector')['Sector Weight'].plot(legend=True, figsize=(10, 6))

        plt.title(f'{efm_dict[efm]} Sector Weights over time')
        plt.xlabel('Effective Date')
        plt.ylabel('Sector Weight')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        
        # save plot as a jpg image in output directory
        plot_title = f'Sector_weights_over_time_for_{efm_dict[efm]}.jpg'
        plt.savefig(os.path.join(output_path, plot_title), dpi=300, bbox_inches='tight')
        plt.show()
        
        df_filtered_dict[efm] = {}
        
        df_fund_sectors = pd.DataFrame(columns=[sector for sector in list(sector_weights_df['Sector'].unique())])
        
        for sector in sector_weights_df['Sector'].unique():
            
            df_filtered = sector_weights_df.loc[(sector_weights_df['Account Code'] == efm) &
                                                (sector_weights_df['Sector'] == sector)]

            df_filtered_dict[efm][sector] = df_filtered
            
            df_fund_sectors[sector] = df_filtered['Sector Weight']
        
        df_fund_sectors_list.append(df_fund_sectors)
        
    df_fund_sectors = pd.concat(df_fund_sectors_list, axis=1)
    
    save_csv(df_fund_sectors, output_path, 'fund_holdings_by_sector')
    
        # df_filtered_dict[account] = df_filtered
    # return df_filtered_dict

def plot_industry_group_weights(sector_weights_df, efm_dict, output_path):
    
    df_filtered_dict = {}
    df_fund_sectors_list = []
    
    for efm in efm_dict.keys():
        
        df_filtered = sector_weights_df.loc[(sector_weights_df['Account Code'] == efm)]


        # Plot the lines
        ax = df_filtered.groupby('Industry Group')['Industry Group Weight'].plot(legend=True, figsize=(10, 6))#, cmap='tab20')
        
        # Add labels next to the lines
        for group_name, group_data in df_filtered.groupby('Industry Group')['Industry Group Weight']:
            x = group_data.index
            y = group_data.values
            plt.text(x[-1], y[-1], f'{group_name}', ha='left', va='center')
            
        
        # Display the plot
        plt.show()


        plt.title(f'{efm_dict[efm]} Industry Group Weights over time')
        plt.xlabel('Effective Date')
        plt.ylabel('Industry Group Weight')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        
        # save plot as a jpg image in output directory
        plot_title = f'Industry_group_weights_over_time_for_{efm_dict[efm]}.jpg'
        plt.savefig(os.path.join(output_path, plot_title), dpi=300, bbox_inches='tight')
        plt.show()
        
        df_filtered_dict[efm] = {}
        
        df_fund_sectors = pd.DataFrame(columns=[sector for sector in list(sector_weights_df['Industry Group'].unique())])
        
        for sector in sector_weights_df['Industry Group'].unique():
            
            df_filtered = sector_weights_df.loc[(sector_weights_df['Account Code'] == efm) &
                                                (sector_weights_df['Industry Group'] == sector)]

            df_filtered_dict[efm][sector] = df_filtered
            
            df_fund_sectors[sector] = df_filtered['Industry Group Weight']
        
        df_fund_sectors_list.append(df_fund_sectors)
        
    df_fund_sectors = pd.concat(df_fund_sectors_list, axis=1)
    
    save_csv(df_fund_sectors, output_path, 'fund_holdings_by_industry_group')
    
def tabulate_changes(sector_weights_df, efm_dict, output_path, diff):
    
    df_filtered_dict = {}
    
    for efm in efm_dict.keys():
        df_filtered_dict[efm_dict[efm]] = {}
        
        for sector in sector_weights_df['Sector'].unique():
            df_filtered = sector_weights_df[(sector_weights_df['Sector'] == sector) & (sector_weights_df['Account Code'] == efm)]
            if not df_filtered.empty and len(df_filtered) > diff:
                change = df_filtered['Sector Weight'].iloc[-1] - df_filtered['Sector Weight'].iloc[-diff-1]
                df_filtered_dict[efm_dict[efm]][sector] = change
    
    sector_changes_df = pd.DataFrame(df_filtered_dict)
    sector_changes_df['Average'] = sector_changes_df.mean(axis=1)
    
    return sector_changes_df

def tabulate_changes_all(sector_weights_df, efm_dict, output_path, change_windows):
    
    df_changes_list = []
    
    for diff in change_windows:
        sector_changes_df = tabulate_changes(sector_weights_df, efm_dict, output_path, diff)
        df_styled = sector_changes_df.style.background_gradient(cmap='RdYlGn').format(format_pct).set_caption(f'Sector changes in the last {diff} days')
        display(df_styled)
        df_changes_list.append(sector_changes_df)
        save_csv(sector_changes_df, output_path, f'sector_changes_{diff}_days')
        
    return df_changes_list

def calculate_top_holdings(holdings, output_path):
    
    stock_weight_dict = {}
    top_holdings_dict = {}
    all_top_dict = {}
    
    for account, acct_group in holdings.groupby('Account Code'):
        
        stock_weight_dict[account] = {}
        top_holdings_dict[account] = {}
        
        for sedol, sedol_group in acct_group.groupby('Ticker_x'):
            stock_weight = sedol_group['Market Value'].sum() / acct_group['Market Value'].sum()
            stock_weight_dict[account][sedol] = stock_weight
        
        stock_weight_df = pd.DataFrame(stock_weight_dict)
        top_holdings_dict[account] = stock_weight_df.nlargest(10, account).index
        
    for sedol, sedol_group in holdings.groupby('Ticker_x'):
        stock_weight = sedol_group['Market Value'].sum() / holdings['Market Value'].sum()
        all_top_dict[sedol] = stock_weight
        
    all_top = pd.DataFrame(all_top_dict.items(), columns = ['Ticker_x', 'Stock Weight'])  
    top_holdings_dict['All'] = all_top.nlargest(10, 'Stock Weight')['Ticker_x']
        
    top_holdings = pd.DataFrame(top_holdings_dict)
    save_csv(top_holdings, output_path, 'top_holdings')
    
    return top_holdings

def top_holdings_day(holdings):
    
    top_holdings_dict = {}
    for acct, acct_group in holdings.groupby('Account Code'):
        top_holdings_dict[acct] = {}
        for date, date_group in acct_group.groupby(level=0):
            date_string = date.strftime('%Y-%m-%d')
            df = date_group.sort_values('Market Value', ascending=False).iloc[:10]
            df = df[['Ticker_x', 'Quantity', 'Market Value']]
            df.rename(columns={'Ticker_x': 'Ticker'}, inplace=True)
            top_holdings_dict[acct][date_string] = df
    
    return top_holdings_dict

def holdings_make_dict(holdings):
    
    unique_ticker = list(holdings['Ticker_x'].unique())
    
    holdings_dict = {}
    for acct, acct_group in holdings.groupby('Account Code'):
        holdings_dict[acct] = {}
        for date, date_group in acct_group.groupby(level=0):
            date_string = date.strftime('%Y-%m-%d')
            holdings_dict[acct][date_string] = date_group
            unique_t = list(date_group['Ticker_x'].unique())
            absent_ticker = [ticker for ticker in unique_ticker if ticker not in unique_t]
            for ticker in absent_ticker:
                
                
                holdings_dict[acct][date_string].loc[len(holdings_dict[acct][date_string])] = [date, acct, ticker] + [np.nan]*(holdings_dict[acct][date_string].shape[1] - 3)
            
            
    return holdings_dict
# def top_holdings_changes(top_holdings_dict):
    
#     top_holdings_changes_dict = {}
    
#     for account in top_holdings_dict.keys():

#         # Obtain list of dates and sort in descending order
#         dates = list(top_holdings_dict[account].keys())
#         dates.sort(reverse=True)

#         # Find latest and second last dates
#         latest_date = dates[0]
#         second_last_date = dates[1]

#         # Obtain the DataFrames for the two dates
#         df_latest = top_holdings_dict[account][latest_date]
#         df_second_last = top_holdings_dict[account][second_last_date]

#         # Subtract the two DataFrames
#         df_diff = df_latest - df_second_last
        
#         top_holdings_changes_dict[account] = df_diff

        
#     return top_holdings_changes_dict
            
# def holdings_changes(holdings, change_windows):

#     holdings_by_group_dict = {}
#     for acct, acct_group in holdings.groupby('Account Code'):
#         holdings_by_group_dict[acct] = {}
#         for date, date_group in acct_group.groupby(level=0):
#             date_string = date.strftime('%Y-%m-%d')
#             df = date_group[['Ticker_x', 'Quantity', 'Market Value']]
#             df = df.rename(columns={'Ticker_x': 'Ticker'})
#             holdings_by_group_dict[acct][date_string] = df
    
#     holdings_changes_dict = {}
#     for account in holdings_by_group_dict.keys():
#         # Obtain list of dates and sort in descending order
#         dates = list(holdings_by_group_dict[account].keys())
#         dates.sort(reverse=True)
        
#         df_diff_dict = {}
#         for n in change_windows:
            
#             # Find latest and second last dates
#             d2 = dates[0]
#             d1 = dates[n]

#             # Obtain the DataFrames for the two dates
#             df2 = holdings_by_group_dict[account][d2].reset_index().drop(columns='Effective Date')
#             df1 = holdings_by_group_dict[account][d1].reset_index().drop(columns='Effective Date')
#             df3 = pd.merge(df1, df2, on='Ticker', how='outer', suffixes=('_1', '_2'))
#             df3 = df3.fillna(0)
            
#             # Subtract the two DataFrames
#             q_diff = df3['Quantity_2'] - df3['Quantity_1']
#             market_diff = df3['Market Value_2'] - df3['Market Value_1']

#             df3['Quantity Change'] = q_diff
#             df3['Market Value Change'] = market_diff

#             df_diff_dict[str(n)] = df3

#         holdings_changes_dict[account] = df_diff_dict
        
#     return holdings_changes_dict

def top_holdings_changes(holdings_changes_dict):
    
    top_holdings_changes_dict = {}
    for account in holdings_changes_dict.keys():
        top_holdings_changes_dict[account] = {}
        
        for frame in holdings_changes_dict[account].keys():
            df = holdings_changes_dict[account][frame].sort_values('Market Value_2', ascending=False).iloc[:10].reset_index(drop=True)
            df.index += 1  # add 1 to each index value
            df = df[['Ticker', 'Market Value_2', 'Quantity Change', 'Market Value Change']]
            top_holdings_changes_dict[account][frame] = df
        
    return top_holdings_changes_dict

def top_holdings_dashboard(top_holdings_changes_dict, efm_dict):
    
    top_holdings_dashboard_dict = {}
    change_columns = ['Quantity Change', 'Market Value Change']
    
    for account in top_holdings_changes_dict.keys():
        top_holdings_dashboard_dict[account] = {}
        
        for column in change_columns:
            df_merged = pd.DataFrame()
            window_list = []
            for window in top_holdings_changes_dict[account]:
                df = top_holdings_changes_dict[account][window][['Ticker', column]]
                window_list.append(window)
                if df_merged.empty:
                    df_merged = df.copy()
                else:
                    df_merged = pd.merge(df_merged, df, how="outer", on="Ticker", suffixes=(f'_{window}_1', f'{window}_2'))
            
            window_list = [f"{window}-day change" for window in window_list]
            df_merged = df_merged.set_index('Ticker')
            column_mapping = {old_name: new_name for old_name, new_name in zip(df_merged.columns, window_list)}
            df_merged = df_merged.rename(columns=column_mapping)
            df_merged.name = f'{column} for {efm_dict[account]}'
            df_styled = df_merged.style.background_gradient(cmap='RdYlGn', axis=None).format('{:.0f}').set_caption(df_merged.name)
            top_holdings_dashboard_dict[account][column] = df_styled
            display(df_styled) 
                
    return top_holdings_dashboard_dict

def stock_changes(holdings, change_windows):
    
    stock_list = list(holdings['SEDOL'].unique())
    stock_unique = pd.DataFrame(columns=['SEDOL'], data=stock_list)
    
    holdings_by_group_dict = {}
    for acct, acct_group in holdings.groupby('Account Code'):
        holdings_by_group_dict[acct] = {}
        for date, date_group in acct_group.groupby(level=0):
            date_string = date.strftime('%Y-%m-%d')
            df = date_group[['SEDOL', 'Ticker_x', 'Quantity', 'Market Value']]
            df = df.rename(columns={'Ticker_x': 'Ticker'})
            holdings_by_group_dict[acct][date_string] = df
    
    holdings_changes_dict = {}
    for account in holdings_by_group_dict.keys():
        
        
        # Obtain list of dates and sort in descending order
        dates = list(holdings_by_group_dict[account].keys())
        dates.sort(reverse=True)
        
        df_diff_dict = {}
        for n in change_windows:
            
            # Define 2 dates
            d2 = dates[n-1]
            d1 = dates[n]
            
            # Obtain the DataFrames for the two dates
            df2 = holdings_by_group_dict[account][d2].reset_index().drop(columns='Effective Date')
            df1 = holdings_by_group_dict[account][d1].reset_index().drop(columns='Effective Date')
            
            df2 = pd.merge(df2, stock_unique, on='SEDOL', how='outer')
            df1 = pd.merge(df1, stock_unique, on='SEDOL', how='outer')
            
            df3 = pd.merge(df1, df2, on='SEDOL', how='outer', suffixes=('_1', '_2'))
            df3 = df3.fillna(0)

            # Subtract the two DataFrames
            q_diff = (df3['Quantity_2'] / df3['Quantity_1']) - 1

            df3['Quantity % Change'] = q_diff
            df3 = df3.replace([np.inf, -np.inf], 1)
            
            df3 = df3[['Ticker_2', 'SEDOL', 'Quantity_2', 'Quantity % Change']]

            df_diff_dict[str(n-1)] = df3

        holdings_changes_dict[account] = df_diff_dict
        
    return holdings_changes_dict

def stock_changes_df(holdings_changes_dict, efm_dict):
    
    df_list = []
    for account in holdings_changes_dict.keys():

        for window in holdings_changes_dict[account]:
            if window == '-1':
                pass
            else:
                df = holdings_changes_dict[account][window]
                df['Fund'] = account
                df['window'] = window
                
                df_list.append(df)
            
    holdings_changes_df = pd.concat(df_list)

                
    return holdings_changes_df

def holdings_ema(holdings_changes_df):
    
    ema_dict = {}
    
    for fund, fund_group in holdings_changes_df.groupby('Fund'):
        ema_dict[fund] = {}
        for sedol, sedol_group in fund_group.groupby('SEDOL'):
            average = sedol_group['SEDOL'].mean()
            ema_dict[fund][sedol] = average
            
    return ema_dict
            



"""
Holdings Change Test
d_f = holdings_by_group_dict['MOMG']['2023-03-24'].reset_index().drop(columns='Effective Date').sort_values('Ticker')
d_1 = holdings_by_group_dict['MOMG']['2023-03-23'].reset_index().drop(columns='Effective Date').sort_values('Ticker')
d_3 = holdings_by_group_dict['MOMG']['2023-03-21'].reset_index().drop(columns='Effective Date').sort_values('Ticker')
d_5 = holdings_by_group_dict['MOMG']['2023-03-17'].reset_index().drop(columns='Effective Date').sort_values('Ticker')

dates = list(holdings_by_group_dict['MOMG'])
dates.sort(reverse=True)

# Find latest and second last dates
d2 = dates[0]
d1 = dates[50]

# Obtain the DataFrames for the two dates
df2 = holdings_by_group_dict[account][d2].reset_index().drop(columns='Effective Date')
df1 = holdings_by_group_dict[account][d1].reset_index().drop(columns='Effective Date')

# Subtract the two DataFrames
df3 = df2.copy()
df3 = df3.rename(columns={'Market Value':'MV2',
                          'Quantity':'Q2'})
df3['Q1'] = df1['Quantity']
df3['MV1'] = df1['Market Value']

q_diff = df2['Quantity'] - df1['Quantity']                 
market_diff = df2['Market Value'] - df1['Market Value']

df3['Quantity Difference'] = q_diff
df3['Market Value Difference'] = market_diff

df3.sort_values('Ticker')

df = holdings_by_group_dict[account][d2].reset_index().drop(columns='Effective Date')
di = holdings_by_group_dict[account][d1].reset_index().drop(columns='Effective Date')
"""

# missing_sedols = [sedol for sedol in efm_holdings_data['SEDOL'] if sedol.strip() not in bbg_processed['SEDOL'].str.strip().unique()]

# name, holdings, bbg, stock
# Linde, BZ12WP8, BNZHB81, BNZHB81
# Atlassian, US0494681010, BQ1PC76
# Columbia Banking, 2176608
# iShares S&P 500, 2593025
# iShares MSCI Europe, BN90WN8
# Wacker Chemie, B11Y568
# DNO ASA, B15GGN4
# Frontline LTD, BDDJSX3
# Sacyr, BR0WX94
# Hafnia ltd, BJK0P85

# def sector_weights(holdings):
#     sector_weights_dict = {}
#     for date in list(holdings.index.unique()):
#         print(f'Processing: {date}')
#         sector_weights_dict[date] = {}
#         for account in list(holdings['Account Code'].unique()):
#             sector_weights_dict[date][account] = {}
#             holdings_filtered = holdings.loc[(holdings.index == date) &
#                                  (holdings['Account Code'] == account)]         
#             account_sum = holdings_filtered['Market Value'].sum()
#             for sector in list(holdings['Sector'].unique()):
#                 holdings_filtered = holdings.loc[(holdings.index == date) &
#                                  (holdings['Account Code'] == account) &
#                                  (holdings['Sector'] == sector)]
                
#                 if account_sum != 0:
#                     sector_weight = holdings_filtered['Market Value'].sum() / account_sum
#                 else:
#                     sector_weight = 0
                
#                 sector_weights_dict[date][account][sector] = sector_weight
    
#     sector_weights_df = pd.DataFrame(
#         [(date, account, sector, sector_weights_dict[date][account][sector]) 
#          for date in sector_weights_dict 
#          for account in sector_weights_dict[date] 
#          for sector in sector_weights_dict[date][account]],
#          columns=['Effective Date', 'Account Code', 'Sector', 'Sector Weight']).set_index('Effective Date')

#     return sector_weights_df


# AAPL
# MSFT
# BRKB/B
# FB
# GOOGL
# AMZN
# NVDA
# LLY
