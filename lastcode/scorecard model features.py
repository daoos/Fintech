# coding=utf8
import pandas as pd
import datetime
import collections
import numpy as np
import numbers
import random
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import sys
import pickle

reload(sys)
sys.setdefaultencoding("utf-8")
sys.path.append('/home/daoos/Documents/xiaoxiang_fintech/lastdata/')
from scorecard_functions import *
from sklearn.linear_model import LogisticRegressionCV

# -*- coding: utf-8 -*-


#########################################################################################################
# Step 0: Initiate the data processing work, including reading csv files, checking the consistency of Idx
# 步骤0：启动数据处理工作，包括读取csv文件，检查Idx的一致性#
#########################################################################################################
print '步骤0：启动数据处理工作，包括读取csv文件，检查Idx的一致性'
folderOfData = '/home/daoos/Documents/xiaoxiang_fintech/lastdata/'

data1 = pd.read_csv(folderOfData + 'PPD_LogInfo_3_1_Training_Set.csv', header=0)
data2 = pd.read_csv(folderOfData + 'PPD_Training_Master_GBK_3_1_Training_Set.csv', header=0, encoding='gbk')
data3 = pd.read_csv(folderOfData + 'PPD_Userupdate_Info_3_1_Training_Set.csv', header=0)

######################################################################################################################################################
# Step 1: Derivate the features using PPD_Training_Master_GBK_3_1_Training_Set， PPD_LogInfo_3_1_Training_Set &  PPD_Userupdate_Info_3_1_Training_Set#
######################################################################################################################################################
# compare whether the four city variables match
print '合并训练数据开始'
data2['city_match'] = data2.apply(lambda x: int(x.UserInfo_2 == x.UserInfo_4 == x.UserInfo_8 == x.UserInfo_20), axis=1)
del data2['UserInfo_2']
del data2['UserInfo_4']
del data2['UserInfo_8']
del data2['UserInfo_20']

### Extract the applying date of each applicant
data1['logInfo'] = data1['LogInfo3'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
data1['Listinginfo'] = data1['Listinginfo1'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
data1['ListingGap'] = data1[['logInfo', 'Listinginfo']].apply(lambda x: (x[1] - x[0]).days, axis=1)

# maxListingGap = max(data1['ListingGap'])
timeWindows = TimeWindowSelection(data1, 'ListingGap', range(30, 361, 30))

'''
We use 180 as the maximum time window to work out some features in data1.
The used time windows can be set as 7 days, 30 days, 60 days, 90 days, 120 days, 150 days and 180 days.
We calculate the count of total and count of distinct of each raw field within selected time window.
'''
time_window = [7, 30, 60, 90, 120, 150, 180]
var_list = ['LogInfo1', 'LogInfo2']
data1GroupbyIdx = pd.DataFrame({'Idx': data1['Idx'].drop_duplicates()})

for tw in time_window:
    data1['TruncatedLogInfo'] = data1['Listinginfo'].map(lambda x: x + datetime.timedelta(-tw))
    temp = data1.loc[data1['logInfo'] >= data1['TruncatedLogInfo']]
    for var in var_list:
        # count the frequences of LogInfo1 and LogInfo2
        count_stats = temp.groupby(['Idx'])[var].count().to_dict()
        data1GroupbyIdx[str(var) + '_' + str(tw) + '_count'] = data1GroupbyIdx['Idx'].map(
            lambda x: count_stats.get(x, 0))

        # count the distinct value of LogInfo1 and LogInfo2
        Idx_UserupdateInfo1 = temp[['Idx', var]].drop_duplicates()
        uniq_stats = Idx_UserupdateInfo1.groupby(['Idx'])[var].count().to_dict()
        data1GroupbyIdx[str(var) + '_' + str(tw) + '_unique'] = data1GroupbyIdx['Idx'].map(
            lambda x: uniq_stats.get(x, 0))

        # calculate the average count of each value in LogInfo1 and LogInfo2
        data1GroupbyIdx[str(var) + '_' + str(tw) + '_avg_count'] = data1GroupbyIdx[
            [str(var) + '_' + str(tw) + '_count', str(var) + '_' + str(tw) + '_unique']]. \
            apply(lambda x: x[0] * 1.0 / x[1], axis=1)

data3['ListingInfo'] = data3['ListingInfo1'].map(lambda x: datetime.datetime.strptime(x, '%Y/%m/%d'))
data3['UserupdateInfo'] = data3['UserupdateInfo2'].map(lambda x: datetime.datetime.strptime(x, '%Y/%m/%d'))
data3['ListingGap'] = data3[['UserupdateInfo', 'ListingInfo']].apply(lambda x: (x[1] - x[0]).days, axis=1)
collections.Counter(data3['ListingGap'])
hist_ListingGap = np.histogram(data3['ListingGap'])
hist_ListingGap = pd.DataFrame({'Freq': hist_ListingGap[0], 'gap': hist_ListingGap[1][1:]})
hist_ListingGap['CumFreq'] = hist_ListingGap['Freq'].cumsum()
hist_ListingGap['CumPercent'] = hist_ListingGap['CumFreq'].map(lambda x: x * 1.0 / hist_ListingGap.iloc[-1]['CumFreq'])

'''
we use 180 as the maximum time window to work out some features in data3. The used time windows can be set as
7 days, 30 days, 60 days, 90 days, 120 days, 150 days and 180 days
Because we observe some mismatch of letter's upercase/lowercase, like QQ & qQ, Idnumber & idNumber, so we firstly make them consistant。
Besides, we combine MOBILEPHONE&PHONE into PHONE.
Within selected time window, we calculate the
 (1) the frequences of updating
 (2) the distinct of each item
 (3) some important items like IDNUMBER,HASBUYCAR, MARRIAGESTATUSID, PHONE
'''
data3['UserupdateInfo1'] = data3['UserupdateInfo1'].map(ChangeContent)
data3GroupbyIdx = pd.DataFrame({'Idx': data3['Idx'].drop_duplicates()})

time_window = [7, 30, 60, 90, 120, 150, 180]
for tw in time_window:
    data3['TruncatedLogInfo'] = data3['ListingInfo'].map(lambda x: x + datetime.timedelta(-tw))
    temp = data3.loc[data3['UserupdateInfo'] >= data3['TruncatedLogInfo']]

    # frequency of updating
    freq_stats = temp.groupby(['Idx'])['UserupdateInfo1'].count().to_dict()
    data3GroupbyIdx['UserupdateInfo_' + str(tw) + '_freq'] = data3GroupbyIdx['Idx'].map(lambda x: freq_stats.get(x, 0))

    # number of updated types
    Idx_UserupdateInfo1 = temp[['Idx', 'UserupdateInfo1']].drop_duplicates()
    uniq_stats = Idx_UserupdateInfo1.groupby(['Idx'])['UserupdateInfo1'].count().to_dict()
    data3GroupbyIdx['UserupdateInfo_' + str(tw) + '_unique'] = data3GroupbyIdx['Idx'].map(
        lambda x: uniq_stats.get(x, x))

    # average count of each type
    data3GroupbyIdx['UserupdateInfo_' + str(tw) + '_avg_count'] = data3GroupbyIdx[
        ['UserupdateInfo_' + str(tw) + '_freq', 'UserupdateInfo_' + str(tw) + '_unique']]. \
        apply(lambda x: x[0] * 1.0 / x[1], axis=1)

    # whether the applicant changed items like IDNUMBER,HASBUYCAR, MARRIAGESTATUSID, PHONE
    Idx_UserupdateInfo1['UserupdateInfo1'] = Idx_UserupdateInfo1['UserupdateInfo1'].map(lambda x: [x])
    Idx_UserupdateInfo1_V2 = Idx_UserupdateInfo1.groupby(['Idx'])['UserupdateInfo1'].sum()
    for item in ['_IDNUMBER', '_HASBUYCAR', '_MARRIAGESTATUSID', '_PHONE']:
        item_dict = Idx_UserupdateInfo1_V2.map(lambda x: int(item in x)).to_dict()
        data3GroupbyIdx['UserupdateInfo_' + str(tw) + str(item)] = data3GroupbyIdx['Idx'].map(
            lambda x: item_dict.get(x, x))

# Combine the above features with raw features in PPD_Training_Master_GBK_3_1_Training_Set
allData = pd.concat([data2.set_index('Idx'), data3GroupbyIdx.set_index('Idx'), data1GroupbyIdx.set_index('Idx')],
                    axis=1)
allData.to_csv(folderOfData + 'allData_0.csv', encoding='gbk')

print '合并训练数据结束'
##################################################################################
# Step 2: Makeup missing value for categorical variables and continuous variables#
##################################################################################
print 'Step2:不全缺失字段开始'
allData = pd.read_csv(folderOfData + 'allData_0.csv', header=0, encoding='gbk')
allFeatures = list(allData.columns)
allFeatures.remove('target')
allFeatures.remove('Idx')
allFeatures.remove('ListingInfo')

# check columns and remove them if they are a constant
# else determine whethert it is continuous or categorical type
numerical_var = []
for col in allFeatures:
    if len(set(allData[col])) == 1:
        print 'delete {} from the dataset because it is a constant'.format(col)
        del allData[col]
        allFeatures.remove(col)
    else:
        # uniq_vals = list(set(allData[col]))
        # if np.nan in uniq_vals:
        # uniq_vals.remove(np.nan)
        uniq_valid_vals = [i for i in allData[col] if i == i]
        uniq_valid_vals = list(set(uniq_valid_vals))
        if len(uniq_valid_vals) >= 10 and isinstance(uniq_valid_vals[0], numbers.Real):
            numerical_var.append(col)

categorical_var = [i for i in allFeatures if i not in numerical_var]

'''
For each categorical variable, if the missing value occupies more than 50%, we remove it.
Otherwise we will use missing as a special status
'''
missing_pcnt_threshould_1 = 0.5
for col in categorical_var:
    missingRate = MissingCategorial(allData, col)
    print '{0} has missing rate as {1}'.format(col, missingRate)
    if missingRate > missing_pcnt_threshould_1:
        categorical_var.remove(col)
        del allData[col]
    if 0 < missingRate < missing_pcnt_threshould_1:
        # In this way we convert NaN to NAN, which is a string instead of np.nan
        allData[col] = allData[col].map(lambda x: str(x).upper())

allData_bk = allData.copy()
'''
For continuous variable, if the missing value is more than 30%, we remove it.
Otherwise we use random sampling method to make up the missing
'''
missing_pcnt_threshould_2 = 0.3
deleted_var = []
for col in numerical_var:
    missingRate = MissingContinuous(allData, col)
    print '{0} has missing rate as {1}'.format(col, missingRate)
    if missingRate > missing_pcnt_threshould_2:
        deleted_var.append(col)
        print 'we delete variable {} because of its high missing rate'.format(col)
    else:
        if missingRate > 0:
            not_missing = allData.loc[allData[col] == allData[col]][col]
            # makeuped = allData[col].map(lambda x: MakeupRandom(x, list(not_missing)))
            missing_position = allData.loc[allData[col] != allData[col]][col].index
            not_missing_sample = random.sample(not_missing, len(missing_position))
            allData.loc[missing_position, col] = not_missing_sample
            # del allData[col]
            # allData[col] = makeuped
            missingRate2 = MissingContinuous(allData, col)
            print 'missing rate after making up is:{}'.format(str(missingRate2))

if deleted_var != []:
    for col in deleted_var:
        numerical_var.remove(col)
        del allData[col]

allData.to_csv(folderOfData + 'allData_1.csv', header=True, encoding='gbk', columns=allData.columns, index=False)

print 'Step2:不全缺失字段结束'
####################################
# Step 3: Group variables into bins#
####################################
# for each categorical variable, if it has distinct values more than 5, we use the ChiMerge to merge it
print 'Step3:分组变量开始'
trainData = pd.read_csv(folderOfData + 'allData_1.csv', header=0, encoding='gbk')
allFeatures = list(trainData.columns)
allFeatures.remove('ListingInfo')
allFeatures.remove('target')
allFeatures.remove('Idx')
# devide the whole independent variables into categorical type and numerical type
numerical_var = []
for var in allFeatures:
    uniq_vals = list(set(trainData[var]))
    if np.nan in uniq_vals:
        uniq_vals.remove(np.nan)
    if len(uniq_vals) >= 10 and isinstance(uniq_vals[0], numbers.Real):
        numerical_var.append(var)

categorical_var = [i for i in allFeatures if i not in numerical_var]

for col in categorical_var:
    # for Chinese character, upper() is not valid
    if col not in ['UserInfo_7', 'UserInfo_9', 'UserInfo_19', 'UserInfo_22', 'UserInfo_23', 'UserInfo_24',
                   'Education_Info3', 'Education_Info7', 'Education_Info8']:
        trainData[col] = trainData[col].map(lambda x: str(x).upper())

'''
For cagtegorical variables, follow the below steps
1, if the variable has distinct values more than 5, we calculate the bad rate and encode the variable with the bad rate
2, otherwise:
(2.1) check the maximum bin, and delete the variable if the maximum bin occupies more than 90%
(2.2) check the bad percent for each bin, if any bin has 0 bad samples, then combine it with samllest non-zero bad bin,
        and then check the maximum bin again
'''
deleted_features = []  # delete the categorical features in one of its single bin occupies more than 90%
encoded_features = {}
merged_features = {}
var_IV = {}  # save the IV values for binned features
var_WOE = {}
for col in categorical_var:
    print 'we are processing {}'.format(col)
    if len(set(trainData[col])) > 5:
        print '{} is encoded with bad rate'.format(col)
        col0 = str(col) + '_encoding'

        # (1), calculate the bad rate and encode the original value using bad rate
        encoding_result = BadRateEncoding(trainData, col, 'target')
        trainData[col0], br_encoding = encoding_result['encoding'], encoding_result['br_rate']

        # (2), push the bad rate encoded value into numerical varaible list
        numerical_var.append(col0)

        # (3), save the encoding result, including new column name and bad rate
        encoded_features[col] = [col0, br_encoding]

        # (4), delete the original value
        # del trainData[col]
        deleted_features.append(col)
    else:
        maxPcnt = MaximumBinPcnt(trainData, col)
        if maxPcnt > 0.9:
            print '{} is deleted because of large percentage of single bin'.format(col)
            deleted_features.append(col)
            categorical_var.remove(col)
            # del trainData[col]
            continue
        bad_bin = trainData.groupby([col])['target'].sum()
        if min(bad_bin) == 0:
            print '{} has 0 bad sample!'.format(col)
            col1 = str(col) + '_mergeByBadRate'
            # (1), determine how to merge the categories
            mergeBin = MergeBad0(trainData, col, 'target')
            # (2), convert the original data into merged data
            trainData[col1] = trainData[col].map(mergeBin)
            maxPcnt = MaximumBinPcnt(trainData, col1)
            if maxPcnt > 0.9:
                print '{} is deleted because of large percentage of single bin'.format(col)
                deleted_features.append(col)
                categorical_var.remove(col)
                del trainData[col]
                continue
            # (3) if the merged data satisify the requirement, we keep it
            merged_features[col] = [col1, mergeBin]
            WOE_IV = CalcWOE(trainData, col1, 'target')
            var_WOE[col1] = WOE_IV['WOE']
            var_IV[col1] = WOE_IV['IV']
            # del trainData[col]
            deleted_features.append(col)
        else:
            WOE_IV = CalcWOE(trainData, col, 'target')
            var_WOE[col] = WOE_IV['WOE']
            var_IV[col] = WOE_IV['IV']

'''
For continous variables, we do the following work:
1, split the variable by ChiMerge (by default into 5 bins)
2, check the bad rate, if it is not monotone, we decrease the number of bins until the bad rate is monotone
3, delete the variable if maximum bin occupies more than 90%
'''
var_cutoff = {}
for col in numerical_var:
    print "{} is in processing".format(col)
    col1 = str(col) + '_Bin'
    # (1), split the continuous variable and save the cutoff points. Particulary, -1 is a special case and we separate it into a group
    if -1 in set(trainData[col]):
        special_attribute = [-1]
    else:
        special_attribute = []
    cutOffPoints = ChiMerge_MaxInterval(trainData, col, 'target', special_attribute=special_attribute)
    var_cutoff[col] = cutOffPoints
    trainData[col1] = trainData[col].map(lambda x: AssignBin(x, cutOffPoints, special_attribute=special_attribute))

    # (2), check whether the bad rate is monotone
    BRM = BadRateMonotone(trainData, col1, 'target', special_attribute=special_attribute)
    if not BRM:
        for bins in range(4, 1, -1):
            cutOffPoints = ChiMerge_MaxInterval(trainData, col, 'target', max_interval=bins,
                                                special_attribute=special_attribute)
            trainData[col1] = trainData[col].map(
                lambda x: AssignBin(x, cutOffPoints, special_attribute=special_attribute))
            BRM = BadRateMonotone(trainData, col1, 'target', special_attribute=special_attribute)
            if BRM:
                break
        var_cutoff[col] = cutOffPoints

    # (3), check whether any single bin occupies more than 90% of the total
    maxPcnt = MaximumBinPcnt(trainData, col1)
    if maxPcnt > 0.9:
        # del trainData[col1]
        deleted_features.append(col)
        numerical_var.remove(col)
        print 'we delete {} because the maximum bin occupies more than 90%'.format(col)
        continue
    WOE_IV = CalcWOE(trainData, col1, 'target')
    var_IV[col] = WOE_IV['IV']
    var_WOE[col] = WOE_IV['WOE']
    # del trainData[col]

trainData.to_csv(folderOfData + 'allData_2.csv', header=True, encoding='gbk', columns=trainData.columns, index=False)

filewrite = open(folderOfData + 'var_WOE.pkl', 'w')
pickle.dump(var_WOE, filewrite)
filewrite.close()

filewrite = open(folderOfData + 'var_IV.pkl', 'w')
pickle.dump(var_IV, filewrite)
filewrite.close()



filewrite = open(folderOfData + 'merged_features.pkl', 'w')
pickle.dump(merged_features, filewrite)
filewrite.close()

filewrite = open(folderOfData + 'encoded_features.pkl', 'w')
pickle.dump(encoded_features, filewrite)
filewrite.close()

filewrite = open(folderOfData + 'numerical_var.pkl', 'w')
pickle.dump(numerical_var, filewrite)
filewrite.close()

filewrite = open(folderOfData + 'categorical_var.pkl', 'w')
pickle.dump(categorical_var, filewrite)
filewrite.close()

print 'Step3:分组变量结束'
#########################################################
# Step 4: Select variables with IV > 0.02 and assign WOE#
#########################################################
print 'Step4:挑选出变量IV>0.02开始'
trainData = pd.read_csv(folderOfData + 'allData_2.csv', header=0, encoding='gbk')

num2str = ['SocialNetwork_13', 'SocialNetwork_12', 'UserInfo_6', 'UserInfo_5', 'UserInfo_10', 'UserInfo_17',
           'city_match']
for col in num2str:
    trainData[col] = trainData[col].map(lambda x: str(x))

for col in var_WOE.keys():
    print col
    col2 = str(col) + "_WOE"
    if col in var_cutoff.keys():
        cutOffPoints = var_cutoff[col]
        special_attribute = []
        if - 1 in cutOffPoints:
            special_attribute = [-1]
        binValue = trainData[col].map(lambda x: AssignBin(x, cutOffPoints, special_attribute=special_attribute))
        trainData[col2] = binValue.map(lambda x: var_WOE[col][x])
    else:
        trainData[col2] = trainData[col].map(lambda x: var_WOE[col][x])

trainData.to_csv(folderOfData + 'allData_3.csv', header=True, encoding='gbk', columns=trainData.columns, index=False)

filewrite = open(folderOfData + 'var_cutoff.pkl', 'w')
pickle.dump(var_cutoff, filewrite)
filewrite.close()

### (i) select the features with IV above the thresould
trainData = pd.read_csv(folderOfData + 'allData_3.csv', header=0, encoding='gbk')
iv_threshould = 0.02
varByIV = [k for k, v in var_IV.items() if v > iv_threshould]

### (ii) check the collinearity of any pair of the features with WOE after (i)

var_IV_selected = {k: var_IV[k] for k in varByIV}
var_IV_sorted = sorted(var_IV_selected.iteritems(), key=lambda d: d[1], reverse=True)
var_IV_sorted = [i[0] for i in var_IV_sorted]

removed_var = []
roh_thresould = 0.6
for i in range(len(var_IV_sorted) - 1):
    if var_IV_sorted[i] not in removed_var:
        x1 = var_IV_sorted[i] + "_WOE"
        for j in range(i + 1, len(var_IV_sorted)):
            if var_IV_sorted[j] not in removed_var:
                x2 = var_IV_sorted[j] + "_WOE"
                roh = np.corrcoef([trainData[x1], trainData[x2]])[0, 1]
                if abs(roh) >= roh_thresould:
                    print 'the correlation coeffient between {0} and {1} is {2}'.format(x1, x2, str(roh))
                    if var_IV[var_IV_sorted[i]] > var_IV[var_IV_sorted[j]]:
                        removed_var.append(var_IV_sorted[j])
                    else:
                        removed_var.append(var_IV_sorted[i])

var_IV_sortet_2 = [i for i in var_IV_sorted if i not in removed_var]

### (iii) check the multi-colinearity according to VIF > 10
for i in range(len(var_IV_sortet_2)):
    x0 = trainData[var_IV_sortet_2[i] + '_WOE']
    x0 = np.array(x0)
    X_Col = [k + '_WOE' for k in var_IV_sortet_2 if k != var_IV_sortet_2[i]]
    X = trainData[X_Col]
    X = np.matrix(X)
    regr = LinearRegression()
    clr = regr.fit(X, x0)
    x_pred = clr.predict(X)
    R2 = 1 - ((x_pred - x0) ** 2).sum() / ((x0 - x0.mean()) ** 2).sum()
    vif = 1 / (1 - R2)
    if vif > 10:
        print "Warning: the vif for {0} is {1}".format(var_IV_sortet_2[i], vif)

print 'Step4:挑选出变量IV>0.02结束'
#############################################################################################################
# Step 5: build the logistic regression using selected variables after single analysis and mulitple analysis#
#############################################################################################################
print 'Step5:开始'
### (1) put all the features after single & multiple analysis into logisitic regression
var_WOE_list = [i + '_WOE' for i in var_IV_sortet_2]
y = trainData['target']
X = trainData[var_WOE_list]
X['intercept'] = [1] * X.shape[0]

LR = sm.Logit(y, X).fit()
summary = LR.summary()
pvals = LR.pvalues
pvals = pvals.to_dict()

### Some features are not significant, so we need to delete feature one by one.
varLargeP = {k: v for k, v in pvals.items() if v >= 0.1}
varLargeP = sorted(varLargeP.iteritems(), key=lambda d: d[1], reverse=True)
while (len(varLargeP) > 0 and len(var_WOE_list) > 0):
    # In each iteration, we remove the most insignificant feature and build the regression again, until
    # (1) all the features are significant or
    # (2) no feature to be selected
    varMaxP = varLargeP[0][0]
    if varMaxP == 'intercept':
        print 'the intercept is not significant!'
        break
    var_WOE_list.remove(varMaxP)
    y = trainData['target']
    X = trainData[var_WOE_list]
    X['intercept'] = [1] * X.shape[0]

    LR = sm.Logit(y, X).fit()
    summary = LR.summary()
    pvals = LR.pvalues
    pvals = pvals.to_dict()
    varLargeP = {k: v for k, v in pvals.items() if v >= 0.1}
    varLargeP = sorted(varLargeP.iteritems(), key=lambda d: d[1], reverse=True)

'''
Now all the features are significant and the sign of coefficients are negative
现在所有的特征是重要的，系数的符号是​​负的
var_WOE_list = ['UserInfo_15_encoding_WOE', 'UserInfo_14_encoding_WOE', u'ThirdParty_Info_Period6_10_WOE', u'ThirdParty_Info_Period5_2_WOE',
                'UserInfo_16_encoding_WOE', 'WeblogInfo_20_encoding_WOE', u'WeblogInfo_6_WOE', 'UserInfo_19_encoding_WOE', u'UserInfo_17_WOE',
                u'ThirdParty_Info_Period3_10_WOE', u'ThirdParty_Info_Period1_10_WOE', 'WeblogInfo_2_encoding_WOE', 'UserInfo_1_encoding_WOE']
'''

saveModel = open(folderOfData + 'LR_Model_Normal.pkl', 'w')
pickle.dump(LR, saveModel)
saveModel.close()

print 'Step5:结束'
######################################################################################################
# Step 6(a): build the logistic regression using lasso and weights based on variables given in Step 5#
######################################################################################################
print 'Step6a:开始'
### use cross validation to select the best regularization parameter
X = trainData[var_WOE_list]  # by default  LogisticRegressionCV() fill fit the intercept
X = np.matrix(X)
y = trainData['target']
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
X_train.shape, y_train.shape

model_parameter = {}
for C_penalty in np.arange(0.005, 0.2, 0.005):
    for bad_weight in range(2, 101, 2):
        LR_model_2 = LogisticRegressionCV(Cs=[C_penalty], penalty='l1', solver='liblinear',
                                          class_weight={1: bad_weight, 0: 1})
        LR_model_2_fit = LR_model_2.fit(X_train, y_train)
        y_pred = LR_model_2_fit.predict_proba(X_test)[:, 1]
        scorecard_result = pd.DataFrame({'prob': y_pred, 'target': y_test})
        performance = KS_AR(scorecard_result, 'prob', 'target')
        KS = performance['KS']
        model_parameter[(C_penalty, bad_weight)] = KS
print 'Step6a:结束'
####################################################################################
# Step 6(b): build the logistic regression using according to RF feature importance#
####################################################################################
### build random forest model to estimate the importance of each feature
### In this case we use the original feautures with WOE encoding before single analysis
print 'Step6b:开始'
X = trainData[var_WOE_list]
X = np.matrix(X)
y = trainData['target']
y = np.array(y)

RFC = RandomForestClassifier()
RFC_Model = RFC.fit(X, y)
features_rfc = trainData[var_WOE_list].columns
featureImportance = {features_rfc[i]: RFC_Model.feature_importances_[i] for i in range(len(features_rfc))}
featureImportanceSorted = sorted(featureImportance.iteritems(), key=lambda x: x[1], reverse=True)
# we selecte the top 10 features
features_selection = [k[0] for k in featureImportanceSorted[:10]]

y = trainData['target']
X = trainData[features_selection]
X['intercept'] = [1] * X.shape[0]

LR = sm.Logit(y, X).fit()
summary = LR.summary()
print '执行结束'
"""
                           Logit Regression Results
==============================================================================
Dep. Variable:                 target   No. Observations:                30000
Model:                          Logit   Df Residuals:                    29989
Method:                           MLE   Df Model:                           10
Date:                Wed, 03 May 2017   Pseudo R-squ.:                 0.05182
Time:                        23:09:14   Log-Likelihood:                -7452.9
converged:                       True   LL-Null:                       -7860.2
                                        LLR p-value:                1.424e-168
==================================================================================================
                                     coef    std err          z      P>|z|      [0.025      0.975]
--------------------------------------------------------------------------------------------------
UserInfo_19_encoding_WOE          -0.9937      0.100     -9.905      0.000      -1.190      -0.797
ThirdParty_Info_Period1_10_WOE    -0.7825      0.136     -5.735      0.000      -1.050      -0.515
UserInfo_1_encoding_WOE           -0.8724      0.132     -6.589      0.000      -1.132      -0.613
ThirdParty_Info_Period3_10_WOE    -0.2880      0.118     -2.438      0.015      -0.520      -0.056
UserInfo_16_encoding_WOE          -0.9349      0.091    -10.254      0.000      -1.114      -0.756
WeblogInfo_20_encoding_WOE        -0.8038      0.105     -7.633      0.000      -1.010      -0.597
ThirdParty_Info_Period6_10_WOE    -0.6269      0.084     -7.451      0.000      -0.792      -0.462
WeblogInfo_2_encoding_WOE         -0.1810      0.033     -5.508      0.000      -0.245      -0.117
WeblogInfo_6_WOE                  -0.7409      0.091     -8.155      0.000      -0.919      -0.563
ThirdParty_Info_Period5_2_WOE     -0.5969      0.082     -7.258      0.000      -0.758      -0.436
intercept                         -2.6398      0.026   -101.145      0.000      -2.691      -2.589
==================================================================================================
"""
