# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 12:23:23 2023

@author: intern.yoongsiew
"""

# -*- coding: utf-8 -*-
"""
Sample code that shows how to import values for Asset Attributes

@author: ricardo.martinez@msci.com
"""
# %reset -f

from suds.client import Client
import datetime as dt
from datetime import datetime
import time
import sys
import csv
import os
import pandas as pd
import numpy as np
#%%


header_line_number = 2  # Assuming the first line is 0

with open('Factor value at stock level - Constituents (1).csv', 'r') as f:
    lines = f.readlines()
    header = lines[header_line_number]

# Convert header line to a list of column names
header_list = header.strip().split(',')
qfl = "QFL "


#%%
#logging.basicConfig(filename='import.log', level=logging.INFO)
#logging.getLogger('suds.client').setLevel(logging.DEBUG)
#logging.getLogger('suds.transport').setLevel(logging.DEBUG)

url = "https://www.barraone.com/axis2/services/BDTService?wsdl"
client = Client(url, location=url, timeout=50000)

usr = "AHamid"
pwd = "Khazanah2022$"
cid = "w5m9mk5qau"

file = os.path.join(os.path.dirname(os.getcwd()), 'Khazanah', 'Factor value at stock level - Constituents (1).csv')
# attributeName = 'qfl Earnings Yield'  # the name of your price attribute (should exist in B1/BPM)
riskModel = 'GEMLTESG'  # risk model to validate assetIds


def autoconvert(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s




def createValues(asset, value):
	myMid = client.factory.create('MID')
	myMid._ID = asset
	myMid._IDType = 'SEDOL'
	myMid._Priority = 1

	attrValue = client.factory.create('AttrValue')
	attrValue._Value = value
	attrValue._Currency = "USD"
	attrValue.MID = [myMid]
	
	return attrValue


def doUpload(attributeName, column_index):
    oldDate = ""
    allAttList = []
    attValList = []
    with open(file) as rowdata:
        next(rowdata)  # skip first line of headers
        next(rowdata)  # skip first line of headers
        next(rowdata)  # skip first line of headers
        hf_info=csv.reader(rowdata, delimiter=',')

        for row in hf_info:
		
            date = row[2]
            asset = row[4]
            value = autoconvert(row[column_index])
            if isinstance(value, str):
                continue
			
			#print (date)
			
            if (oldDate == ""):
                oldDate = date
			
            if (date == oldDate):			
                myAttVal = createValues(asset, value)				
                attValList.append(myAttVal)
				
            else:
                att1 = client.factory.create('AssetAttribute')
                att1._Name = attributeName
                att1._Owner = usr
                att1._EffectiveStartDate =  datetime.strptime(oldDate, '%m/%d/%Y') 
				
                attValues = client.factory.create('AttrValues')
                attValues.AttrValue = attValList
                att1.AttrValues = attValues				
                allAttList.append(att1)
                oldDate = date
                attValList = []
                myAttVal = createValues(asset, value)				
                attValList.append(myAttVal)
				
        att1 = client.factory.create('AssetAttribute')
        att1._Name = attributeName
        att1._Owner = usr
        try:
            att1._EffectiveStartDate = datetime.strptime(oldDate, '%m/%d/%Y') 
        except ValueError:
            print(f'Error processing date: {oldDate}')    
		
        attValues = client.factory.create('AttrValues')
        attValues.AttrValue = attValList
        att1.AttrValues = attValues				
        allAttList.append(att1)

    jobId = client.service.SubmitImportJob(User=usr, Client=cid, Password=pwd, ModelName=riskModel, JobName="AttrUpload_"+attributeName, AssetAttribute=allAttList)
    print('Processing job:', jobId)
	
    time.sleep(5)  # wait time of 5 secs is required before GetImportJobStatus

    sleepTime = 30
    while sleepTime > 0:
        try:
            sleepTime = client.service.GetImportJobStatus(usr, cid, pwd, jobId)
        except suds.WebFault as detail:
            print (detail)
        except:
            print ("Unexpected error:", sys.exc_info()[0])
            raise
        if sleepTime > 0:
            time.sleep(sleepTime)
            
    logResponse = client.service.GetImportJobLog(usr, cid, pwd, jobId)
    print("Job Name:", logResponse._JobName)
    if sleepTime == 0:
        print('Job successful. Getting import log...')
        print("Date, Name, Owner, Total, Rejects, Blanks, Duplicates, Deleted, Msg")
        
        for ejr in logResponse.LogGroups.ImportLogGroup:
            print(ejr._EffectiveDate, " ", ejr._Name, " ", ejr._Owner, " ", ejr._Total, " ", ejr._Rejected, " ", ejr._Blank, " ", ejr._Duplicate, " ", ejr._Deleted, " ", ejr._ResultMsg)
            for grp in ejr.Details.ImportLogDetail:
                if grp._ResultMsg.startswith('Risk model'):
                    print(">> ", grp._ResultMsg, " ", grp._Detail1)
                else:
                    print(">> ", grp._ResultMsg, " ", grp._Detail1, " ", grp._Detail2)
                    
    else:
        print('Job failed. Please see log file for error details.')
        print("Date, Name, Owner")
        for ejr in logResponse.LogGroups.ImportLogGroup:
            print(ejr._EffectiveDate, ", ", ejr._Name, ", ", ejr._Owner)
            for grp in ejr.Details.ImportLogDetail:
                print(">> ", grp._ResultMsg, " ", grp._Detail1)
#%%
for i, element in enumerate(header_list[5:]):
    if i < 60:
        continue
    # Do something with element
    print(f"{i+5}th column name is : {qfl + element}")
    doUpload(qfl + element, i+5)

    #%%

import math
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


class DecisionTree():

    def __init__(self, X, y, min_samples_leaf=5, max_depth=6, idxs=None):
        assert max_depth >= 0, 'max_depth must be nonnegative'
        assert min_samples_leaf > 0, 'min_samples_leaf must be positive'
        self.min_samples_leaf, self.max_depth = min_samples_leaf, max_depth
        if isinstance(y, pd.Series): y = y.values
        if idxs is None: idxs = np.arange(len(y))
        self.X, self.y, self.idxs = X, y, idxs
        self.n, self.c = len(idxs), X.shape[1]
        self.value = y.mean()
        self.best_score_so_far = float('inf')
        
        if self.max_depth > 0:
            self._maybe_insert_child_nodes()
            
            
    def _maybe_insert_child_nodes(self):
        for j in range(self.c):
            self._find_better_split(j)
        if self.is_leaf: #do not insert children
            return
        x = self.X.values[self.idxs, self.split_feature_idx]
        left_idx = np.nonzero(x <= self.threshold)[0]
        right_idx = np.nonzero(x > self.threshold)[0]
        self.left = DecisionTree(self.X, self.y, self.min_samples_leaf, self.max_depth - 1, self.idxs[left_idx])
        self.right = DecisionTree(self.X, self.y, self.min_samples_leaf, self.max_depth - 1, self.idxs[right_idx])
        
    def _find_better_split(self, feature_idx):
        pass
    
    @property
    def is_leaf(self):
        return self.best_score_so_far == float('inf')
#%%
from sklearn.datasets import load_diabetes
X, y = load_diabetes(as_frame=True, return_X_y=True)



t = DecisionTree(X, y, min_samples_leaf=5, max_depth=5)



























