#%% import
"""
Created on Wed Jul 12 09:22:19 2023

@author: intern.yoongsiew
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier, XGBRFClassifier, cv
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
import shap
import os
warnings.filterwarnings("ignore", category=UserWarning, message="`use_label_encoder` is deprecated")

#%%  month with monthly change xlsx read


# # Define the directory containing the xlsx files
# directory = 'C:\\Users\\intern.yoongsiew\\OneDrive - Khazanah Nasional Berhad\\Documents\\Khazanah repo\\Khazanah'


# # Define the skiprows, columns to drop, and columns to rename for each file
# file_info = {
#     'vix.xlsx': {'skiprows': 6, 'drop_cols': ['PX_VOLUME'], 'rename_cols': {'PX_LAST': 'vix'}},
#     'futures net long until 2021 only.xlsx': {'skiprows': 5, 'drop_cols': [], 'rename_cols': {'PX_LAST': 'futures_net_long'}},
#     'us 10y breakeven.xlsx': {'skiprows': 5, 'drop_cols': ['PX_BID'], 'rename_cols': {'PX_LAST': 'us10ybreakeven'}},
#     'sp500 momentum.xlsx': {'skiprows': 6, 'drop_cols': ['PX_VOLUME'], 'rename_cols': {'PX_LAST': 'sp500_momentum'}},
#     'high yield spread.xlsx': {'skiprows': 6, 'drop_cols': [], 'rename_cols': {'PX_BID': 'HY_spread'}},
#     'pull call ratio.xlsx': {'skiprows': 6, 'drop_cols': ['PX_MID'], 'rename_cols': {'PX_LAST': 'putcall_ratio'}},
#     'ISM manu new orders.xlsx': {'skiprows': 5, 'drop_cols': ['FIRST_REVISION'], 'rename_cols': {'PX_LAST': 'ism_mfg_new_order'}},
#     'skew index.xlsx': {'skiprows': 6, 'drop_cols': ['PX_VOLUME'], 'rename_cols': {'PX_LAST': 'skew_index'}},
#     'economic activity tracker.xlsx': {'skiprows': 5, 'drop_cols': ['FIRST_REVISION'], 'rename_cols': {'PX_LAST': 'econ_activity_tracker'}},
#     'us 10y real yield.xlsx': {'skiprows': 5, 'drop_cols': ['YLD_CNV_MID'], 'rename_cols': {'PX_MID': '10y_real_yield'}},
#     'yield curve.xlsx': {'skiprows': 6, 'drop_cols': ['PX_BID'], 'rename_cols': {'PX_LAST': 'yield_curve'}},
#     'oil prices.xlsx': {'skiprows': 6, 'drop_cols': ['PX_VOLUME'], 'rename_cols': {'PX_LAST': 'oil'}},
#     'bbg econ surprise.xlsx': {'skiprows': 6, 'drop_cols': ['PX_VOLUME'], 'rename_cols': {'PX_LAST': 'bbg_eco_surprise'}},
#     'aaii net.xlsx': {'skiprows': 5, 'drop_cols': [], 'rename_cols': {'PX_LAST': 'aaii_net'}},
#     'implied recession probability.xlsx': {'skiprows': 5, 'drop_cols': ['PX_MID'], 'rename_cols': {'PX_LAST': 'implied_recession_proba'}},
#     'supply chain pressure.xlsx': {'skiprows': 5, 'drop_cols': ['FIRST_REVISION'], 'rename_cols': {'PX_LAST': 'supply_chain_pressure'}},
#     'spx data.xlsx': {'skiprows': 6, 'drop_cols': ['PX_VOLUME'], 'rename_cols': {'PX_LAST': 'spx'}},
# }
  
# def load_df(directory, file_info):
#     # Create an empty dictionary to store the dataframes
#     dfs = {}
    
#     # Loop through each file in the directory
#     for filename in os.listdir(directory):
#         if filename.endswith('.xlsx'):
#             # Get the file path
#             filepath = os.path.join(directory, filename)
            
#             # Get the file info for this file
#             info = file_info.get(filename)
#             if info is None:
#                 continue
            
#             # Read the file into a dataframe
#             df = pd.read_excel(filepath, skiprows=info['skiprows'], index_col=0, parse_dates=True)
            
#             # Drop the specified columns
#             df = df.drop(columns=info['drop_cols'])
            
#             # Rename the specified columns
#             df = df.rename(columns=info['rename_cols'])
            
#             # Resample the dataframe to monthly frequency
#             df = df.resample('M').mean()
            
#             # Add the dataframe to the dictionary
#             dfs[filename] = df
#     return dfs

# dfs = load_df(directory, file_info)

#%% concat

# df_concat = pd.concat(dfs.values(), axis=1)


#%% daily freq monthly change  csv read

directory = 'C:\\Users\\intern.yoongsiew\\OneDrive - Khazanah Nasional Berhad\\Documents\\Khazanah repo\\Khazanah\\daily_resolution'
filename = 'daily_marco_daily.csv'
filepath = os.path.join(directory, filename)

df_concat = pd.read_csv(filepath, index_col=0, parse_dates=True)
for col in df_concat.columns:
    df_concat[col] = pd.to_numeric(df_concat[col], errors='coerce')
    
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
df_concat.rename(columns=name_mapping, inplace=True)

df_concat = df_concat.iloc[1:] # remove NaT the PX_LAST
df_concat = df_concat.fillna(method='ffill')
#%% get percent change 
df_concat['future_1m'] = df_concat['SPX Index'].shift(-25)
# df_concat = df_concat.iloc[1:]
# df_concat = df_concat.fillna(method='ffill')
df_concat['perct_change'] = (df_concat['future_1m'] - df_concat['SPX Index']) / df_concat['SPX Index'] * 100


#%%
# df_concat['implied_recession_proba'] = df_concat['implied_recession_proba'] /10

# df_concat['perct_change'] = df_concat['spx'].pct_change() * 100
# df_concat['perct_change'] = df_concat['perct_change'].shift(-1)
# df_concat['perct_change'] = df_concat['perct_change'].fillna(0)
df_concat.drop(['SPX Index', 'future_1m'], axis=1, inplace=True)




#%% 3 month change
def add_3m_change(df, columns):
    """
    Adds new columns to the dataframe for the 3-month change of the specified columns.
    """
    for col in columns:
        # Calculate the 3-month change for the column
        col_3m_ago = df[col].shift(25 * 3)
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
        col_1m_ago = df[col].shift(25)
        col_change = df[col] - col_1m_ago
        
        # Add the new column to the dataframe
        new_col_name = f'{col}_1m_change'
        col_index = df.columns.get_loc(col)
        df.insert(col_index + 1, new_col_name, col_change)
        
        
    return df



# Define the columns to calculate the 3-month change for
columns = ['futures_net_long', 'ism_mfg_new_order', '10y_real_yield', 'oil', 'HY_spread', 'implied_recession_proba', 'supply_chain_pressure', 'vix']
columns = ['Futures Net Long', 'ISM Mfgs New Orders', 'US 10Y Real Yield', 'US 10 Y breakeven', 'Oil', 'Econ Activity Tracker', 'US High Yield spread', 'Implied Recession Proba', 'Supply Chain Pressure', 'VIX']

# Add the new columns to the dataframe
df_concat = add_3m_change(df_concat, columns)

# Define the columns to calculate the 1-month change for
columns = ['futures_net_long']
columns = ['Futures Net Long']
# Add the new columns to the dataframe
df_concat = add_1m_change(df_concat, columns)
#%%
# df_concat.drop(['VIX'], axis=1, inplace=True)
df_concat = df_concat.loc['2003-10-31':'2021-11-30']
df_concat = df_concat.fillna(method='ffill')
#%% plot indicators to check for stationary

# Loop through each column in the dataframe
for col in df_concat.columns:
    # Create a new figure and axis
    fig, ax = plt.subplots()
    
    # Plot the column on the axis
    ax.plot(df_concat[col])
    
    # Set the title and axis labels
    ax.set_title(col)
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    
    # Show the plot
    plt.show()



#%% positive training preprocessing

up_percentage_threshold = 3

df_concat['target'] = np.where(df_concat['perct_change'] > up_percentage_threshold, 1, 0)

X = df_concat.drop(['target', 'perct_change'], axis=1)
y = df_concat['target']


train_test_ratio = 0.7

X_train = X.iloc[ : round(len(df_concat) * train_test_ratio)]
X_test = X.iloc[round(len(df_concat) * train_test_ratio) : ]
y_train = y.iloc[ : round(len(df_concat) * train_test_ratio)]
y_test = y.iloc[round(len(df_concat) * train_test_ratio) : ]

# train_indices = np.arange(len(X_train))
# np.random.shuffle(train_indices)
# X_train = X_train.iloc[train_indices]
# y_train = y_train.iloc[train_indices]


# Perform standardization on the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("Before SMOTE:", Counter(y_train))
print("After SMOTE:", Counter(y_train_smote))


tomek = TomekLinks()
X_train_smote, y_train_smote = tomek.fit_resample(X_train, y_train)

print("Before tomek:", Counter(y_train))
print("After tomek:", Counter(y_train_smote))

class_weights = dict(pd.Series(y_train).value_counts(normalize=True))

#%%
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42, 
                               max_depth=6, 
                               min_samples_leaf=5) 

param_grid = {
    'max_depth': [5, 6],
    'min_samples_leaf': [1, 5]
}

np.random.seed(42)
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='balanced_accuracy')

grid_search.fit(X_train_smote, y_train_smote)

print(grid_search.best_params_)
model = RandomForestClassifier(random_state=42, **grid_search.best_params_)  
model.fit(X_train_smote, y_train_smote)





#%% xgboost + random forest

neg_class_samples = np.sum(y_train == 0)
pos_class_samples = np.sum(y_train == 1)
scale_pos_weight = neg_class_samples / pos_class_samples


model = XGBRFClassifier(random_state=42, 
                        use_label_encoder=False, 
                        objective="binary:logistic",
                        colsample_bytree=0.6,      # Lower ratios avoid over-fitting
                        subsample=0.8,             # Lower ratios avoid over-fitting
                        max_depth=6,               # Lower values avoid over-fitting
                        gamma=0.3,                 # Larger values avoid over-fitting
                        learning_rate=0.1,         # Lower values avoid over-fitting
                        min_child_weight=5,
                        scale_pos_weight=scale_pos_weight) 



param_grid = {
    'colsample_bytree': [0.3, 0.7],
    'subsample': [0.6, 0.8],
    'max_depth': [5, 6],
    'gamma': [0.1, 0.3],
    'learning_rate': [0.05, 0.1],
    'min_child_weight': [1, 5]
}


np.random.seed(42)
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='balanced_accuracy')

grid_search.fit(X_train_smote, y_train_smote)

# model.fit(X_train_smote, y_train_smote)
print(grid_search.best_params_)
model = XGBClassifier(objective="binary:logistic", **grid_search.best_params_)  
model.fit(X_train_smote, y_train_smote)


#%% k fold cross validation


data_dmatrix = xgb.DMatrix(data=X,label=y)
params = {"objective":"binary:logistic",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 10}

xgb_cv = cv(dtrain=data_dmatrix, params=params, nfold=3,
                    num_boost_round=50, early_stopping_rounds=10, metrics="auc", as_pandas=True, seed=123)
#%% see the results

predictions = model.predict(X_train)
cm = confusion_matrix(y_train, predictions)
cm_df = pd.DataFrame(cm, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])
print(cm_df)

X_test_scaled = scaler.fit_transform(X_test)
predictions_proba = model.predict_proba(X_test_scaled)
predictions = model.predict(X_test_scaled)
cm = confusion_matrix(y_test, predictions, labels=model.classes_)
# cm_df = pd.DataFrame(cm, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])
# print(cm_df)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=model.classes_)
disp.plot()
plt.show()
print(classification_report(y_test, predictions))

predictions_proba = predictions_proba[:, 1] 
graph_df = pd.DataFrame(predictions_proba, columns=['probabilities'])
graph_df.index = y_test.index

graph_df = pd.concat([graph_df, y_test], axis=1)
graph_df_weekly = graph_df.resample('W').last()

#%% plot like UBS paper


# Create a figure and axis
fig, ax = plt.subplots(figsize=(14,7))

# Plot the probabilities
ax.plot(graph_df.index, graph_df['probabilities'], color='grey')

# Set the background color based on the target values
colors = ['lightgreen' if val == 1 else None for val in graph_df['target']]
ax.fill_between(graph_df.index, 0, 1, where=graph_df['target'] == 1, facecolor='lightgreen')

# Set the x-axis label
ax.set_xlabel('Date')

# Set the y-axis label
ax.set_ylabel('Probabilities')

# Set the plot title
ax.set_title('Market rally')

# Display the plot
plt.show()

#%% plot like UBS paper (down fall)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(14,7))

# Plot the probabilities
ax.plot(graph_df.index, graph_df['probabilities'], color='grey')

# Set the background color based on the target values
colors = ['lightgreen' if val == 1 else None for val in graph_df['target']]
ax.fill_between(graph_df.index, 0, 1, where=graph_df['target'] == 1, facecolor='lightcoral')

# Set the x-axis label
ax.set_xlabel('Date')

# Set the y-axis label
ax.set_ylabel('Probabilities')

# Set the plot title
ax.set_title('down fall')

# Display the plot
plt.show()



#%% negative training

down_percentage_threshold = -5


df_concat['target'] = np.where(df_concat['perct_change'] < down_percentage_threshold, 1, 0)

X = df_concat.drop(['target', 'perct_change'], axis=1)
y = df_concat['target']


# tsne = TSNE(n_components=2, random_state=42ArithmeticError
# X_tsne = tsne.fit_transform(X)

# plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
# plt.colorbar()
# plt.title('t-SNE Visualization negative direction')
# plt.show()



train_test_ratio = 0.7

X_train = X.iloc[ : round(len(df_concat) * train_test_ratio)]
X_test = X.iloc[round(len(df_concat) * train_test_ratio) : ]
y_train = y.iloc[ : round(len(df_concat) * train_test_ratio)]
y_test = y.iloc[round(len(df_concat) * train_test_ratio) : ]


# train_indices = np.arange(len(X_train))
# np.random.shuffle(train_indices)
# X_train = X_train.iloc[train_indices]
# y_train = y_train.iloc[train_indices]


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("Before SMOTE:", Counter(y_train))
print("After SMOTE:", Counter(y_train_smote))
    
tomek = TomekLinks()
X_train_smote, y_train_smote = tomek.fit_resample(X_train, y_train)

print("Before tomek:", Counter(y_train))
print("After tomek:", Counter(y_train_smote))

class_weights = dict(pd.Series(y_train).value_counts(normalize=True))


#%%

explainer = shap.TreeExplainer(model)

# Compute SHAP values
shap_values = explainer.shap_values(X_test)

# Plot summary plot
shap.summary_plot(shap_values, X_test)

#%%


shap.dependence_plot(
    "US High Yield spread",
    shap_values, X_test)
    
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

for value in df_concat.columns:
    shap.dependence_plot(
        value,
        shap_values, X_test)





