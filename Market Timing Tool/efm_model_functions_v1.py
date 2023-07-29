import pandas as pd
import numpy as np

def clean_df(df):
    """
    This function renames 'Unnamed: 0' to 'SEDOL', converts 'Periods' to datetime, 
    sorts the index by date, and adds 'SEDOL' index
    
    Parameters
    ----------
    df : DataFrame
        df to be cleaned

    Returns
    -------
    df_c : DataFrame
        cleaned df
        
    This function 
    """
    
    # Create copy of dataframe
    df_c = df.copy()
    
    # Rename 'Unnamed:' 0 to 'SEDOL'
    df_c = df_c.rename(columns={'Unnamed: 0':'SEDOL'})
    
    # Convert 'Periods' to datetime format
    df_c['Effective Date'] = pd.to_datetime(df_c['Effective Date'])
    
    # Sets date as the index and sorts it
    df_c.set_index('Effective Date', inplace=True)
    df_c.sort_index(inplace=True)
    
    # Adds 'SEDOL' as second index
    df_c.set_index('SEDOL', append=True, inplace=True)
    
    return df_c

def transform_series(series, new_min, new_max):
    
    old_min = series.min()
    old_max = series.max()
    
    # Apply linear transformation to Scores column
    series_transformed = ((series - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
    
    return series_transformed

def linear_transform_df(df, columns):
    dfc = df.copy()
    for col in columns:
        for date, date_group in dfc.groupby('Effective Date'):
            df_pos_col = date_group[date_group[col] >= 0][col]
            df_neg_col = date_group[date_group[col] < 0][col]
            
            df_pos_col_transformed = transform_series(df_pos_col, 0, 3).reindex(df_pos_col.index)
            df_neg_col_transformed = transform_series(df_neg_col, -3, 0).reindex(df_neg_col.index)
            
            df_col_transformed = pd.concat([df_pos_col_transformed, df_neg_col_transformed]).sort_index()
            
            dfc.loc[date, f"{col} Transformed"] = df_col_transformed
        
    return dfc


def process_holdings(efm_holdings_raw):
    
    # Create copy
    efm_holdings = efm_holdings_raw.copy()
    
    # Convert dates to correct format
    efm_holdings['Effective Date'] = pd.to_datetime(efm_holdings['Effective Date'])
    
    # Convert strings to numbers
    efm_holdings['Market Value'] = efm_holdings['Market Value'].str.replace(',', '').str.replace('(', '-', regex=False).str.replace(')', '', regex=False).astype(float)
    efm_holdings['Quantity'] = efm_holdings['Quantity'].str.replace(',', '').str.replace('(', '-', regex=False).str.replace(')', '', regex=False).astype(float)
    
    # Take only relevant columns
    efm_holdings = efm_holdings[['Account Code', 'SEDOL', 'Effective Date', 'Quantity', 'Market Value']]
    
    # Make into pivot table
    df_pivoted = pd.pivot_table(efm_holdings, values=['Quantity', 'Market Value'], index=['Effective Date', 'SEDOL'], columns='Account Code', aggfunc='sum')
    
    # Fill in all dates with all unique sedols in data
    unique_sedols = df_pivoted.index.get_level_values('SEDOL').unique()  # Get unique SEDOLs

    # Generate a new index with all unique dates and SEDOLs combinations
    new_index = pd.MultiIndex.from_product([df_pivoted.index.get_level_values('Effective Date').unique(), unique_sedols], names=['Effective Date', 'SEDOL'])
    
    # Reindex the dataframe using the new index
    efm_holdings = df_pivoted.reindex(new_index)
    
    # Fill nans with 0
    efm_holdings = efm_holdings.fillna(0)
    
    # Flatten columns
    efm_holdings.columns = [f'{col[1]} {col[0]}' for col in efm_holdings.columns]
    
    return efm_holdings

def calc_quantity_change_score(efm_holdings):
    
    # Calculate daily pct change
    for column in efm_holdings.columns:
        if "Quantity" in column:
            efm_holdings[f'{column} % Change'] = efm_holdings.groupby('SEDOL')[column].pct_change()
            
    # Replace infinites with 1
    efm_holdings = efm_holdings.replace([np.inf], 1)
    
    # Fill nans with 0
    efm_holdings = efm_holdings.fillna(0)
    
    # Calculate EMAs
    sedols = efm_holdings.index.get_level_values('SEDOL').unique()

    for i, sedol in enumerate(sedols):

        # Filter the DataFrame for the current SEDOL
        sedol_df = efm_holdings[efm_holdings.index.get_level_values('SEDOL') == sedol]
        
        for column in efm_holdings.columns:
            if ("Quantity % Change" in column) & ("EMA" not in column):
        
                # Calculate the EMA for the column
                ema_values = sedol_df[column].ewm(span=30, adjust=False).mean()
                
                # Assign the EMA values back to the original DataFrame
                efm_holdings.loc[sedol_df.index, f'{column} EMA'] = ema_values
  
        # Calculate the percentage progress
        progress = (i + 1) / len(sedols) * 100
        
        # Print the percentage progress
        print(f"Progress: {progress:.2f}%", end='\r')
   
    # Get mean of EMAs
    ema_columns = efm_holdings.filter(like='EMA').columns
    efm_holdings['Average EFM % Change EMA'] = efm_holdings[ema_columns].mean(axis=1)
    
    # Linear transform to get scores
    efm_holdings = linear_transform_df(efm_holdings, ['Average EFM % Change EMA'])
    efm_holdings = efm_holdings.rename(columns={'Average EFM % Change EMA Transformed': 'EFM Quantity Change Score'})

        
    return efm_holdings

def calc_abs_weight_score(efm_holdings):
    
    for column in efm_holdings.columns:
        if "Market Value" in column:
            efm_holdings[f'{column} Abs Weight'] = efm_holdings[column] / efm_holdings.groupby('Effective Date')[column].transform('sum')
            
    # Get mean of Weights
    wt_columns = efm_holdings.filter(like='Weight').columns
    efm_holdings['Average EFM Absolute Weight'] = efm_holdings[wt_columns].mean(axis=1)
    
    # Linear transform to get scores
    efm_holdings = linear_transform_df(efm_holdings, ['Average EFM Absolute Weight'])
    efm_holdings = efm_holdings.rename(columns={'Average EFM Absolute Weight Transformed': 'EFM Absolute Weight Score'})
            
    return efm_holdings

def daily_to_monthly(efm_holdings):
    
    dates = list(efm_holdings.index.get_level_values('Effective Date').unique())
    date_series = pd.to_datetime(dates)
    monthly_groups = date_series.to_period('M').unique()
    
    latest_days = []
    for month in monthly_groups:
        group = date_series[date_series.to_period('M') == month]
        latest_day = group.max()
        latest_days.append(latest_day)
        
    monthly_df = efm_holdings[efm_holdings.index.get_level_values('Effective Date').isin(latest_days)].copy()
    
    return monthly_df
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
