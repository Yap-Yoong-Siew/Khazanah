# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 10:43:42 2023

@author: intern.yoongsiew
"""


import pandas as pd
import numpy as np
from xgboost import XGBClassifier, XGBRFClassifier, cv, XGBRegressor, XGBRFRegressor
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from collections import Counter
from supervised.automl import AutoML
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
warnings.filterwarnings("ignore", category=UserWarning, message="`use_label_encoder` is deprecated")

#%%


directory = 'C:\\Users\\intern.yoongsiew\\OneDrive - Khazanah Nasional Berhad\\Documents\\Khazanah repo\\Khazanah\\daily_resolution'
filename = 'daily_marco_daily.csv'
filepath = os.path.join(directory, filename)

macro_df = pd.read_csv(filepath, index_col=0, parse_dates=True)
for col in macro_df.columns:
    macro_df[col] = pd.to_numeric(macro_df[col], errors='coerce')
    
name_mapping = {"VIX Index" : "VIX",
                ".SPFUTNET G Index" : "Futures Net Long",
                "USGGBE10 Index" : "US 10 Y breakeven",
                "LF98OAS Index" : "US High Yield spread",
                "PCUSEQTR Index" : "Put Call Ratio",
                "NAPMNEWO Index" : "ISM Mfgs New Orders",
                "SKEW Index" : "SKEW Index",
                "COMFCOMF Index" : "Econ Activity Tracker",
                "GTII10 Govt" : "US 10Y Real Yield",
                "USYC2Y10 Index" : "Yield Curve",
                "CL1 COMB Comdty" : "Oil",
                ".NETBULL G Index" : "AAII Net",
                "BLERP12M Index" : "Implied Recession Proba",
                "NAPMPMI Index" : "Supply Chain Pressure",
                "CESIUSD Index" : "BBG Eco US surprise"
                }
macro_df.rename(columns=name_mapping, inplace=True)

macro_df = macro_df.iloc[1:] # remove NaT the PX_LAST
# macro_df = macro_df.fillna(method='ffill')


#%% get percent change 
macro_df['future_1m'] = macro_df['SPX Index'].shift(-4)
# macro_df = macro_df.iloc[1:]
# macro_df = macro_df.fillna(method='ffill')
# macro_df['perct_change'] = (macro_df['future_1m'] - macro_df['SPX Index']) / macro_df['SPX Index'] * 100
# macro_df.drop(['SPX Index', 'future_1m'], axis=1, inplace=True)
macro_df = macro_df.resample('M').last()


#%% 3 month change
def add_3m_change(df, columns):
    """
    Adds new columns to the dataframe for the 3-month change of the specified columns.
    """
    for col in columns:
        # Calculate the 3-month change for the column
        col_3m_ago = df[col].shift(3)
        col_change = df[col] - col_3m_ago
        
        # Add the new column to the dataframe
        new_col_name = f'{col}_3m_change'
        col_index = df.columns.get_loc(col)
        df.insert(col_index + 1, new_col_name, col_change)
        
    return df

def add_1m_change(df, columns):
    """
    Adds new columns to the dataframe for the 3-month change of the specified columns.
    """
    for col in columns:
        # Calculate the 3-month change for the column
        col_1m_ago = df[col].shift(1)
        col_change = df[col] - col_1m_ago
        
        # Add the new column to the dataframe
        new_col_name = f'{col}_1m_change'
        col_index = df.columns.get_loc(col)
        df.insert(col_index + 1, new_col_name, col_change)
        
        
    return df



# Define the columns to calculate the 3-month change for
# columns = ['futures_net_long', 'ism_mfg_new_order', '10y_real_yield', 'oil', 'HY_spread', 'implied_recession_proba', 'supply_chain_pressure', 'vix']
columns = ['Futures Net Long', 'ISM Mfgs New Orders', 'US 10Y Real Yield', 'US 10 Y breakeven', 'Oil', 'Econ Activity Tracker', 'US High Yield spread', 'Implied Recession Proba', 'Supply Chain Pressure', 'VIX']

# Add the new columns to the dataframe
macro_df = add_3m_change(macro_df, columns)

# Define the columns to calculate the 1-month change for
# columns = ['futures_net_long']
columns = ['Futures Net Long']
# Add the new columns to the dataframe
macro_df = add_1m_change(macro_df, columns)




#%%

factor_return_df = pd.read_excel("Factor return by period - Periods for Adib.xlsx", skiprows=1)
factor_return_df.drop(['#'], axis=1, inplace=True)
factor_return_df['Period'] = pd.to_datetime(factor_return_df['Period'])
factor_return_df.set_index('Period', inplace=True)
factor_return_df = factor_return_df.iloc[:-1]
factor_return_df = factor_return_df.resample('M').mean()
factor_return_df = factor_return_df.iloc[:, :42]

for factor in factor_return_df.columns:
    # df['Rolling_Mean_3'] = df['Values'].rolling(window=3).mean()

    factor_return_df[factor] = factor_return_df[factor].rolling(window=6).mean()
    # factor_return_df[factor] = factor_return_df[factor].shift(-6)
    

#%% plot factors to check for stationary

# Loop through each column in the dataframe
for col in factor_return_df.columns:
    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(14,6))
    
    # Plot the column on the axis
    ax.plot(factor_return_df[col])
    
    # Set the title and axis labels
    ax.set_title(col)
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    
    # Show the plot
    plt.show()
    
#%%

# Loop through each column in the dataframe
for col in macro_df.columns:
    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(14,6))
    
    # Plot the column on the axis
    ax.plot(macro_df[col])
    
    # Set the title and axis labels
    ax.set_title(col)
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    
    # Show the plot
    plt.show()
#%% merge both df

merged_df = macro_df.join(factor_return_df, how='inner')

#%% Pearson
pearson_corr  = merged_df.corr()
pearson_corr  = pearson_corr.loc[macro_df.columns, factor_return_df.columns]
# Spearman correlation
spearman_corr = merged_df.corr(method='spearman')
spearman_corr = spearman_corr.loc[macro_df.columns, factor_return_df.columns]
# Kendall correlation
kendall_corr = merged_df.corr(method='kendall')
kendall_corr = kendall_corr.loc[macro_df.columns, factor_return_df.columns]
#%%  visualize the correlation


# Set up the matplotlib figure
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))

# Pearson correlation heatmap
sns.heatmap(pearson_corr, annot=True, cmap="coolwarm", center=0, ax=axes[0])
axes[0].set_title('Pearson Correlation')

# Spearman correlation heatmap
sns.heatmap(spearman_corr, annot=True, cmap="coolwarm", center=0, ax=axes[1])
axes[1].set_title('Spearman Correlation')

# Kendall correlation heatmap
sns.heatmap(kendall_corr, annot=True, cmap="coolwarm", center=0, ax=axes[2])
axes[2].set_title('Kendall Correlation')

# Adjust layout
plt.tight_layout()

plt.show()

#%% save to png

# def save_heatmap_to_file(corr_matrix, title, filename):
#     plt.figure(figsize=(6, 5))
#     sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0)
#     plt.title(title)
#     plt.tight_layout()
#     plt.savefig(filename, dpi=300)  # Save with high resolution (300 dpi)


# save_heatmap_to_file(pearson_corr, "Pearson Correlation", "pearson_correlation.png")
# save_heatmap_to_file(spearman_corr, "Spearman Correlation", "spearman_correlation.png")
# save_heatmap_to_file(kendall_corr, "Kendall Correlation", "kendall_correlation.png")

#%%

with pd.ExcelWriter('correlation_matrices.xlsx') as writer:
    pearson_corr.to_excel(writer, sheet_name='Pearson Correlation')
    spearman_corr.to_excel(writer, sheet_name='Spearman Correlation')
    kendall_corr.to_excel(writer, sheet_name='Kendall Correlation')

#%% xgboost

macro_df = merged_df.drop(factor_return_df.columns, axis=1)
factor_return_df = merged_df.drop(macro_df.columns, axis=1)

for factors in factor_return_df.columns:
    factor = factor_return_df[[factors]]
    train_test_df = macro_df.join(factor, how='inner')
    train_test_df.dropna(inplace=True)
    y_name = train_test_df.columns[-1]
    print(f"working for {y_name} now")
    X = train_test_df.drop([y_name], axis=1)
    y = train_test_df.iloc[:, [-1]]
    train_test_ratio = 0.7
    
    X_train = X.iloc[ : round(len(train_test_df) * train_test_ratio)]
    X_test = X.iloc[round(len(train_test_df) * train_test_ratio) : ]
    y_train = y.iloc[ : round(len(train_test_df) * train_test_ratio)]
    y_test = y.iloc[round(len(train_test_df) * train_test_ratio) : ]
    
    # Perform standardization on the features
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)

        
    # tomek = TomekLinks()
    # X_train_smote, y_train_smote = tomek.fit_resample(X_train, y_train)
    
    # print("Before tomek:", Counter(y_train))
    # print("After tomek:", Counter(y_train_smote))
    
    # class_weights = dict(pd.Series(y_train).value_counts(normalize=True))

    
    model = XGBRFRegressor(random_state=42, 
                           objective="reg:squarederror",
                           colsample_bytree=0.6,      # Lower ratios avoid over-fitting
                           subsample=0.8,             # Lower ratios avoid over-fitting
                           max_depth=6,               # Lower values avoid over-fitting
                           gamma=0.3,                 # Larger values avoid over-fitting
                           learning_rate=0.1,         # Lower values avoid over-fitting
                           min_child_weight=5) 
    
    param_grid = {
        'colsample_bytree': [0.3, 0.7],
        'subsample': [0.6, 0.8],
        'max_depth': [5, 6],
        'gamma': [0.1, 0.3],
        'learning_rate': [0.05, 0.1],
        'min_child_weight': [1, 5]
    }
    
    
    np.random.seed(42)
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
    
    grid_search.fit(X_train, y_train)
    
    # print(grid_search.best_params_)
    model = XGBRegressor(objective="reg:squarederror", **grid_search.best_params_)
    model.fit(X_train, y_train)
    
    explainer = shap.TreeExplainer(model)

    # Compute SHAP values
    shap_values = explainer.shap_values(X_test)

    # Plot summary plot
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(f'factor_6m_past_diff/{y_name}.png', bbox_inches='tight', dpi=300)
    plt.close()
