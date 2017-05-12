import pandas as pd
import datetime
import collections
import numpy as np
import numbers
import random
from pandas.tools.plotting import scatter_matrix
from itertools import combinations
import os,sys
reload(sys)
sys.setdefaultencoding("utf-8")
path='/home/daoos/Documents/xiaoxiang_fintech/day8/';
sys.path.append(path+"/Notes/")
from scorecard_fucntions import *
# encoding='utf-8'


#########################################################################################################
#Step 0: Initiate the data processing work, including reading csv files, checking the consistency of Idx#
#########################################################################################################
data1 = pd.read_csv(path+'/PPD_LogInfo_3_1_Training_Set.csv', header=0)
data2 = pd.read_csv(path+'/PPD_Training_Master_GBK_3_1_Training_Set.csv', header= 0,encoding = 'gbk')
data3 = pd.read_csv(path+'/PPD_Userupdate_Info_3_1_Training_Set.csv', header=0)
data1_Idx, data2_Idx, data3_Idx = set(data1), set(data2), set(data3)
# data1_Idx, data2_Idx, data3_Idx = set(data1_Idx), set(data2_Idx), set(data3_Idx)
#
check_Idx_integrity = (data1_Idx - data2_Idx)|(data2_Idx - data1_Idx)|(data1_Idx - data3_Idx)|(data3_Idx - data1_Idx)

#set([85832, 82505, 10922, 78259, 14662]) is only in data1_Idx, so we remove them from the modeling base

###########################################################################################################
# Step 1: Derivate the features using PPD_LogInfo_3_1_Training_Set &  PPD_Userupdate_Info_3_1_Training_Set#
###########################################################################################################

### Extract the applying date of each applicant
data1['logInfo'] = data1['LogInfo3'].map(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))
data1['Listinginfo'] = data1['Listinginfo1'].map(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))
data1['ListingGap'] = data1[['logInfo','Listinginfo']].apply(lambda x: (x[1]-x[0]).days,axis = 1)

#maxListingGap = max(data1['ListingGap'])
timeWindows = TimeWindowSelection(data1, 'ListingGap', range(30,361,30))

'''
We use 180 as the maximum time window to work out some features in data1.
The used time windows can be set as 7 days, 30 days, 60 days, 90 days, 120 days, 150 days and 180 days.
We calculate the count of total and count of distinct of each raw field within selected time window.
'''
time_window = [7, 30, 60, 90, 120, 150, 180]
var_list = ['LogInfo1','LogInfo2']
data1GroupbyIdx = pd.DataFrame({'Idx':data1['Idx'].drop_duplicates()})

for tw in time_window:
    data1['TruncatedLogInfo'] = data1['Listinginfo'].map(lambda x: x + datetime.timedelta(-tw))
    temp = data1.loc[data1['logInfo'] >= data1['TruncatedLogInfo']]
    for var in var_list:
        #count the frequences of LogInfo1 and LogInfo2
        count_stats = temp.groupby(['Idx'])[var].count().to_dict()
        data1GroupbyIdx[str(var)+'_'+str(tw)+'_count'] = data1GroupbyIdx['Idx'].map(lambda x: count_stats.get(x,0))

        # count the distinct value of LogInfo1 and LogInfo2
        Idx_UserupdateInfo1 = temp[['Idx', var]].drop_duplicates()
        uniq_stats = Idx_UserupdateInfo1.groupby(['Idx'])[var].count().to_dict()
        data1GroupbyIdx[str(var) + '_' + str(tw) + '_unique'] = data1GroupbyIdx['Idx'].map(lambda x: uniq_stats.get(x,0))

        # calculate the average count of each value in LogInfo1 and LogInfo2
        data1GroupbyIdx[str(var) + '_' + str(tw) + '_avg_count'] = data1GroupbyIdx[[str(var)+'_'+str(tw)+'_count',str(var) + '_' + str(tw) + '_unique']].\
            apply(lambda x: x[0]*1.0/x[1], axis=1)


data3['ListingInfo'] = data3['ListingInfo1'].map(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))
data3['UserupdateInfo'] = data3['UserupdateInfo2'].map(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))
data3['ListingGap'] = data3[['UserupdateInfo','ListingInfo']].apply(lambda x: (x[1]-x[0]).days,axis = 1)
collections.Counter(data3['ListingGap'])
hist_ListingGap = np.histogram(data3['ListingGap'])
hist_ListingGap = pd.DataFrame({'Freq':hist_ListingGap[0],'gap':hist_ListingGap[1][1:]})
hist_ListingGap['CumFreq'] = hist_ListingGap['Freq'].cumsum()
hist_ListingGap['CumPercent'] = hist_ListingGap['CumFreq'].map(lambda x: x*1.0/hist_ListingGap.iloc[-1]['CumFreq'])

'''
we use 180 as the maximum time window to work out some features in data1. The used time windows can be set as
7 days, 30 days, 60 days, 90 days, 120 days, 150 days and 180 days
Besides, we combine MOBILEPHONE&PHONE into PHONE.
Within selected time window, we calculate the
 (1) the frequences of updating
 (2) the distinct of each item
 (3) some important items like IDNUMBER,HASBUYCAR, MARRIAGESTATUSID, PHONE
'''
data3['UserupdateInfo1'] = data3['UserupdateInfo1'].map(ChangeContent)
data3GroupbyIdx = pd.DataFrame({'Idx':data3['Idx'].drop_duplicates()})

time_window = [7, 30, 60, 90, 120, 150, 180]
for tw in time_window:
    data3['TruncatedLogInfo'] = data3['ListingInfo'].map(lambda x: x + datetime.timedelta(-tw))
    temp = data3.loc[data3['UserupdateInfo'] >= data3['TruncatedLogInfo']]

    #frequency of updating
    freq_stats = temp.groupby(['Idx'])['UserupdateInfo1'].count().to_dict()
    data3GroupbyIdx['UserupdateInfo_'+str(tw)+'_freq'] = data3GroupbyIdx['Idx'].map(lambda x: freq_stats.get(x,0))

    # number of updated types
    Idx_UserupdateInfo1 = temp[['Idx','UserupdateInfo1']].drop_duplicates()
    uniq_stats = Idx_UserupdateInfo1.groupby(['Idx'])['UserupdateInfo1'].count().to_dict()
    data3GroupbyIdx['UserupdateInfo_' + str(tw) + '_unique'] = data3GroupbyIdx['Idx'].map(lambda x: uniq_stats.get(x, x))

    #average count of each type
    data3GroupbyIdx['UserupdateInfo_' + str(tw) + '_avg_count'] = data3GroupbyIdx[['UserupdateInfo_'+str(tw)+'_freq', 'UserupdateInfo_' + str(tw) + '_unique']]. \
        apply(lambda x: x[0] * 1.0 / x[1], axis=1)

    #whether the applicant changed items like IDNUMBER,HASBUYCAR, MARRIAGESTATUSID, PHONE
    Idx_UserupdateInfo1['UserupdateInfo1'] = Idx_UserupdateInfo1['UserupdateInfo1'].map(lambda x: [x])
    Idx_UserupdateInfo1_V2 = Idx_UserupdateInfo1.groupby(['Idx'])['UserupdateInfo1'].sum()
    for item in ['_IDNUMBER','_HASBUYCAR','_MARRIAGESTATUSID','_PHONE']:
        item_dict = Idx_UserupdateInfo1_V2.map(lambda x: int(item in x)).to_dict()
        data3GroupbyIdx['UserupdateInfo_' + str(tw) + str(item)] = data3GroupbyIdx['Idx'].map(lambda x: item_dict.get(x, x))

# Combine the above features with raw features in PPD_Training_Master_GBK_3_1_Training_Set
allData = pd.concat([data2.set_index('Idx'), data3GroupbyIdx.set_index('Idx'), data1GroupbyIdx.set_index('Idx')],axis= 1)
allData.to_csv(path+'/allData_0.csv',encoding = 'gbk')


##################################################################################
# Step 2: Makeup missing value for categorical variables and continuous variables#
##################################################################################
allData = pd.read_csv(path+'/allData_0.csv',header = 0,encoding = 'gbk')
allFeatures = list(allData.columns)
allFeatures.remove('ListingInfo')
allFeatures.remove('target')
allFeatures.remove('Idx')
#check columns and remove them if they are a constant
for col in allFeatures:
    if len(set(allData[col])) == 1:
        print '====delete====>'+col
        del allData[col]
        allFeatures.remove(col)

for col in allFeatures:
    if len(set(allData[col])) == 0:
        print '====delete====>' + col
        del allData[col]
        allFeatures.remove(col)
#devide the whole independent variables into categorical type and numerical type
numerical_var = []
# categorical_var = []
# allFeatures = list(allData.columns)
for var in allFeatures:
    uniq_vals = list(set(allData[var]))
    if np.nan in uniq_vals:
        uniq_vals.remove(np.nan)
        # categorical_var.append(var)
    if len(uniq_vals) >= 10 and isinstance(uniq_vals[0],numbers.Real):
        numerical_var.append(var)

categorical_var = [i for i in allFeatures if i not in numerical_var]



'''
For each categorical variable, if the missing value occupies more than 50%, we remove it.
Otherwise we will use missing as a special status
'''
missing_pcnt_threshould_1 = 0.5
for col in categorical_var:
    missingRate = MissingCategorial(allData,col)
    print '{0} has missing rate as {1}'.format(col,missingRate)
    if missingRate > missing_pcnt_threshould_1:
        categorical_var.remove(col)
        del allData[col]
    if 0 < missingRate < missing_pcnt_threshould_1:
        allData[col] = allData[col].map(lambda x: "'"+str(x)+"'")

allData_bk = allData.copy()

'''
For continuous variable, if the missing value is more than 30%, we remove it.
Otherwise we use random sampling method to make up the missing
'''
missing_pcnt_threshould_2 = 0.3
for col in numerical_var:
    missingRate = MissingContinuous(allData, col)
    print '{0} has missing rate as {1}'.format(col, missingRate)
    if missingRate > missing_pcnt_threshould_2:
        numerical_var.remove(col)
        del allData[col]
        print 'we delete variable {} because of its high missing rate'.format(col)
    else:
        if missingRate > 0:
            not_missing = allData.loc[allData[col] == allData[col]][col]
            makeuped = allData[col].map(lambda x: MakeupRandom(x, list(not_missing)))
            del allData[col]
            allData[col] = makeuped


allData.to_csv(path+'/allData_1b.csv', header=True,encoding='gbk', columns = allData.columns, index=False)


####################################
# Step 3: Group variables into bins#
####################################
#for each categorical variable, if it has distinct values more than 5, we use the ChiMerge to merge it

trainData = pd.read_csv(path+'/allData_1b.csv',header = 0, encoding='gbk')
allFeatures = list(trainData.columns)
allFeatures.remove('ListingInfo')
allFeatures.remove('target')
allFeatures.remove('Idx')

for col in categorical_var:
    trainData[col] = trainData[col].map(lambda x: str(x).upper())


'''
For cagtegorical variables, follow the below steps
1, if the variable has distinct values more than 5, we calculate the bad rate and encode the variable with the bad rate
2, otherwise:
(2.1) check the maximum bin, and delete the variable if the maximum bin occupies more than 90%
(2.2) check the bad percent for each bin, if any bin has 0 bad samples, then combine it with samllest non-zero bad bin,
        and then check the maximum bin again
'''
target='target'
deleted_features = []  #delete the categorical features in one of its single bin occupies more than 90%
encoded_features = []
merged_features = []
var_IV = {}  #save the IV values for binned features
WOE_dict = {}
for col in categorical_var:
    print 'we are processing {}'.format(col)
    if len(set(trainData[col]))>5:
        print '{} is encoded with bad rate'.format(col)
        col0 = str(col)+'_encoding'
        trainData[col0] = BadRateEncoding(trainData, col, 'target')['encoding']
        numerical_var.append(col0)
        encoded_features.append(col0)
        del trainData[col]
    else:
        maxPcnt = MaximumBinPcnt(trainData, col)
        if maxPcnt > 0.9:
            print '{} is deleted because of large percentage of single bin'.format(col)
            deleted_features.append(col)
            categorical_var.remove(col)
            del trainData[col]
            continue
        bad_bin = trainData.groupby([col])[target].sum()
        if min(bad_bin) == 0:
            print '{} has 0 bad sample!'.format(col)
            mergeBin = MergeBad0(trainData, col, target)
            col1 = str(col)+'_mergeByBadRate'
            trainData[col1] = trainData[col].map(mergeBin)
            maxPcnt = MaximumBinPcnt(trainData, col1)
            print maxPcnt
            if maxPcnt > 0.9:
                print '{} is deleted because of large percentage of single bin'.format(col)
                deleted_features.append(col)
                categorical_var.remove(col)
                del trainData[col]
                continue
            WOE_IV = CalcWOE(trainData, col1, target)
            WOE_dict[col1] = WOE_IV['WOE']
            var_IV[col1] = WOE_IV['IV']
            merged_features.append(col)
            del trainData[col]
        else:
            WOE_IV = CalcWOE(trainData, col, target)
            WOE_dict[col] = WOE_IV['WOE']
            var_IV[col] = WOE_IV['IV']


'''
For continous variables, we do the following work:
1, split the variable by ChiMerge (by default into 5 bins)
2, check the bad rate, if it is not monotone, we decrease the number of bins until the bad rate is monotone
3, delete the variable if maximum bin occupies more than 90%
'''
for col in numerical_var:
    print "{} is in processing".format(col)
    col1 = str(col) + '_Bin'
    cutOffPoints = ChiMerge_MaxInterval(trainData, col, 'target')
    trainData[col1] = trainData[col].map(lambda x: AssignBin(x, cutOffPoints))
    #check whether the bad rate is monotone
    BRM = BadRateMonotone(trainData, col1, target)
    if not BRM:
        for bins in range(4,1,-1):
            cutOffPoints = ChiMerge_MaxInterval(trainData, col, 'target',max_interval = bins)
            trainData[col1] = trainData[col].map(lambda x: AssignBin(x, cutOffPoints))
            BRM = BadRateMonotone(trainData, col1, target)
            if BRM:
                break
    #check whether any single bin occupies more than 90% of the total
    maxPcnt = MaximumBinPcnt(trainData, col1)
    if maxPcnt > 0.9:
        del trainData[col1]
        deleted_features.append(col)
        numerical_var.remove(col)
        print 'we delete {} because the maximum bin occupies more than 90%'.format(col)
        continue
    WOE_IV = CalcWOE(trainData, col1, target)
    var_IV[col] = WOE_IV['IV']
    WOE_dict[col] = WOE_IV['WOE']
    del trainData[col]

check_var = 'ThirdParty_Info_Period2_7_Bin'
br = BadRateEncoding(trainData, check_var, target)['br_rate']
bins_sort = sorted(br.keys())
bad_rate = [br[k]['bad_rate'] for k in bins_sort]



woe = WOE_dict['ThirdParty_Info_Period2_7']
bins_sort = sorted(woe.keys())
woe_bin = [woe[k]['WOE'] for k in bins_sort]

trainData.groupby([check_var])[check_var].count()

########################################################
# Step 4: Select variables with IV > 0.02 and assign WOE#
########################################################
iv_threshould = 0.02
varByIV = [k for k, v in var_IV.items() if v > iv_threshould]

WOE_encoding = []
for k in varByIV:
    if k in trainData.columns:
        trainData[str(k)+'_WOE'] = trainData[k].map(lambda x: WOE_dict[k][x]['WOE'])
        WOE_encoding.append(str(k)+'_WOE')
    elif k+str('_Bin') in trainData.columns:
        k2 = k+str('_Bin')
        trainData[str(k) + '_WOE'] = trainData[k2].map(lambda x: WOE_dict[k][x]['WOE'])
        WOE_encoding.append(str(k) + '_WOE')
    else:
        print "{} cannot be found in trainData"

#### we can check the correlation matrix plot
col_to_index = {WOE_encoding[i]:'var'+str(i) for i in range(len(WOE_encoding))}
#sample from the list of columns, since too many columns cannot be displayed in the single plot
corrCols = random.sample(WOE_encoding,15)
sampleDf = trainData[corrCols]
for col in corrCols:
    sampleDf.rename(columns = {col:col_to_index[col]}, inplace = True)
scatter_matrix(sampleDf, alpha=0.2, figsize=(6, 6), diagonal='kde')

#alternatively, we check each pair of independent variables, and selected the variabale with higher IV if they are highly correlated
compare = list(combinations(varByIV, 2))
removed_var = []
roh_thresould = 0.8
for pair in compare:
    (x1, x2) = pair
    roh = np.corrcoef([trainData[str(x1)+"_WOE"],trainData[str(x2)+"_WOE"]])[0,1]
    if abs(roh) >= roh_thresould:
        if var_IV[x1]>var_IV[x2]:
            removed_var.append(x2)
        else:
            removed_var.append(x1)
