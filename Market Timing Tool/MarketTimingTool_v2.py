#!/usr/bin/env python
# coding: utf-8

# ### 1. Market Timing Tool
# - Determine today's regime from Invesco's sector rebalancing
# 
# ### 2. EFM Analysis
# - Charts, tables, dashboards on sector and stock movements
# 
# ### 3. Peer Score
# - Calculation of Peer Score

# In[2]:


import pandas as pd
from efm_model_functions_v1 import *
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[3]:


momg = pd.read_csv('Inputs/EFM Holdings/2111 - 2307/momg_si_20230605.csv')
momn = pd.read_csv('Inputs/EFM Holdings/2111 - 2307/momn_si_20230605.csv')
momp = pd.read_csv('Inputs/EFM Holdings/2111 - 2307/momp_si_20230605.csv')
moms = pd.read_csv('Inputs/EFM Holdings/2111 - 2307/moms_si_20230605.csv')
momu = pd.read_csv('Inputs/EFM Holdings/2111 - 2307/momu_si_20230605.csv')
momx = pd.read_csv('Inputs/EFM Holdings/2111 - 2307/momx_si_20230605.csv')
efm_holdings_raw = pd.concat([momg, momn, momp, moms, momu, momx])


# In[3]:


efm_holdings_raw


# In[4]:


efm_holdings = process_holdings(efm_holdings_raw)


# In[5]:


efm_holdings


# In[12]:


pd.DataFrame(efm_holdings.index.get_level_values('SEDOL').unique(), columns=['SEDOL']).to_csv('EFM_sedols.csv')


# In[4]:


quantity = calc_quantity_change_score(efm_holdings)


# In[5]:


abs_weight = calc_abs_weight_score(quantity)


# In[6]:


abs_weight


# In[40]:


efm_monthly = daily_to_monthly(abs_weight)


# In[41]:


efm_monthly


# In[42]:


efm_monthly.to_csv('efm_monthly.csv')


# In[ ]:




