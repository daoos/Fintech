import pandas as pd
import datetime
import collections
import numpy as np
import random
import ChiMerge as CM
# from sklearn.preprocessing import MDLP



def TimeWindowSelection(df, daysCol, time_windows):
    '''
    :param df: the dataset containg variabel of days
    :param daysCol: the column of days
    :param time_windows: the list of time window
    :return:
    '''
    freq_tw = {}
    for tw in time_windows:
        freq = sum(df[daysCol].apply(lambda x: int(x<=tw)))
        freq_tw[tw] = freq
    return freq_tw


def ChangeContent(x):
    y = x.upper()
    if y == '_MOBILEPHONE':
        y = '_PHONE'
    return y

def MissingCategorial(df,x):
    missing_vals = df[x].map(lambda x: int(x!=x))
    return sum(missing_vals)*1.0/df.shape[0]

def MissingContinuous(df,x):
    missing_vals = df[x].map(lambda x: int(np.isnan(x)))
    return sum(missing_vals) * 1.0 / df.shape[0]

def MakeupRandom(x, sampledList):
    if x==x:
        return x
    else:
        return random.sample(sampledList,1)


path='/home/daoos/Documents/xiaoxiang_fintech/day7/';
data1 = pd.read_csv(path+'/PPD_LogInfo_3_1_Training_Set.csv', header = 0)
data2 = pd.read_csv(path+'/PPD_Training_Master_GBK_3_1_Training_Set.csv', header = 0)
data3 = pd.read_csv(path+'/PPD_Userupdate_Info_3_1_Training_Set.csv', header = 0)

# data1_Idx, data2_Idx, data3_Idx = set(data1_Idx), set(data2_Idx), set(data3_Idx)
data1_Idx, data2_Idx, data3_Idx = set(data1), set(data2), set(data3)
check_Idx_integrity = (data1_Idx - data2_Idx)|(data2_Idx - data1_Idx)|(data1_Idx - data3_Idx)|(data3_Idx - data1_Idx)

#set([85832, 82505, 10922, 78259, 14662]) is only in data1_Idx, so we remove them from the modeling base


### Step 1: Derivate the features using PPD_LogInfo_3_1_Training_Set
### Extract the applying date of each applicant
data1['logInfo'] = data1['LogInfo3'].map(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))
data1['Listinginfo'] = data1['Listinginfo1'].map(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))
data1['ListingGap'] = data1[['logInfo','Listinginfo']].apply(lambda x: (x[1]-x[0]).days,axis = 1)
#maxListingGap = max(data1['ListingGap'])
timeWindows = TimeWindowSelection(data1, 'ListingGap', range(30,361,30))
#we use 180 as the maximum time window to work out some features in data1
#the used time windows can be set as 7 days, 30 days, 60 days, 90 days, 120 days, 150 days and 180 days
#we calculate the count of total and count of distinct of each raw field within selected time window
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


### Step 2: derivate features from PPD_Userupdate_Info_3_1_Training_Set
data3['ListingInfo'] = data3['ListingInfo1'].map(lambda x: datetime.datetime.strptime(x,'%Y/%m/%d'))
data3['UserupdateInfo'] = data3['UserupdateInfo2'].map(lambda x: datetime.datetime.strptime(x,'%Y/%m/%d'))
data3['ListingGap'] = data3[['UserupdateInfo','ListingInfo']].apply(lambda x: (x[1]-x[0]).days,axis = 1)
collections.Counter(data3['ListingGap'])
hist_ListingGap = np.histogram(data3['ListingGap'])
hist_ListingGap = pd.DataFrame({'Freq':hist_ListingGap[0],'gap':hist_ListingGap[1][1:]})
hist_ListingGap['CumFreq'] = hist_ListingGap['Freq'].cumsum()
hist_ListingGap['CumPercent'] = hist_ListingGap['CumFreq'].map(lambda x: x*1.0/hist_ListingGap.iloc[-1]['CumFreq'])
#we use 180 as the maximum time window to work out some features in data1
#the used time windows can be set as 7 days, 30 days, 60 days, 90 days, 120 days, 150 days and 180 days
#into PHONE
#within selected time window, we calculate the
# (1) the frequences of updating
# (2) the distinct of each item
# (3) some important items like IDNUMBER,HASBUYCAR, MARRIAGESTATUSID, PHONE
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

### Combine the above features with raw features in PPD_Training_Master_GBK_3_1_Training_Set
allData = pd.concat([data2.set_index('Idx'), data3GroupbyIdx.set_index('Idx'), data1GroupbyIdx.set_index('Idx')],axis= 1)
allData.to_csv(path+'/allData.csv')
allFeatures = list(allData.columns)
allFeatures.remove('ListingInfo')
allFeatures.remove('target')


### Step 3: makeup missing value
categorical_var = ['UserInfo_2','UserInfo_4','UserInfo_7','UserInfo_8','UserInfo_9','UserInfo_19','UserInfo_20','UserInfo_22','UserInfo_23','UserInfo_24',
                   'Education_Info2','Education_Info3','Education_Info4','Education_Info6','Education_Info7','Education_Info8','WeblogInfo_19',
                   'WeblogInfo_20','WeblogInfo_21']
numerical_var = [i for i in allFeatures if i not in categorical_var]


#for each categorical variable, if the missing value occupies more than 50%, we remove it. Otherwise we will use missing as a special status
missing_pcnt_threshould_1 = 0.5
for var in categorical_var:
    missingRate = MissingCategorial(allData,var)
    print var, ' has missing rate as ',missingRate
    if missingRate > missing_pcnt_threshould_1:
        categorical_var.remove(var)

#for continuous variable, if the missing value is more than 70%, we remove it. Otherwise we use random sampling method
missing_pcnt_threshould_2 = 0.3
for var in numerical_var:
    missingRate = MissingContinuous(allData, var)
    if missingRate > missing_pcnt_threshould_2:
        numerical_var.remove(var)
    else:
        if missingRate > 0:
            not_missing = allData.loc[allData[var] == allData[var]][var]
            allData[var] = allData[var].map(lambda x: MakeupRandom(x,not_missing))


allData.to_csv(path+'/allData.csv')

col = 'UserInfo_18'
col_bin = CM.ChiMerge_MaxInterval(allData, col, 'target')
col_bin = CM.ChiMerge_MinChisq(allData, col, 'target')
