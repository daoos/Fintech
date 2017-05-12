# coding=utf8
import pandas as pd
from pandas import DataFrame
import datetime
import collections
import numpy as np
import numbers
import random
from pandas.tools.plotting import scatter_matrix
from itertools import combinations
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import sys
import pickle
import sys

reload(sys)
sys.setdefaultencoding("utf-8")
# sys.path.append(path+"/Notes/07 申请评分卡中的数据预处理和特征衍生/")
# 引入评分卡处理函数
from scorecard_functions import *

# -*- coding: utf-8 -*-

###############################################################################################
# Step 0: Reading the raw testing data, which are in the same structure with training datasets#
# 第0步：读取与培训数据集相同结构的原始测试数据
###############################################################################################
print '##########################################'
print '第0步：读取与培训数据集相同结构的原始测试数据开始'
folderOfData = '../lastdata/'
data1b = pd.read_csv(folderOfData + 'LogInfo_9w_2.csv', header=0)
data2b = pd.read_csv(folderOfData + 'Kesci_Master_9w_gbk_2.csv', header=0, encoding='utf8')
data3b = pd.read_csv(folderOfData + 'Userupdate_Info_9w_2.csv', header=0)
print '第0步：读取与培训数据集相同结构的原始测试数据结束'
print '##########################################'
#################################################################################
# Step 1: Derivate the features using in the same with as it in training dataset
# 第一步：使用与训练数据集相同的功能来实现#
#################################################################################
print '##########################################'
print '第一步：使用与训练数据集相同的功能来实现开始'
### Extract the applying date of each applicant
data1b['logInfo'] = data1b['LogInfo3'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
data1b['Listinginfo'] = data1b['Listinginfo1'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
data1b['ListingGap'] = data1b[['logInfo', 'Listinginfo']].apply(lambda x: (x[1] - x[0]).days, axis=1)

'''
We use 180 as the maximum time window to work out some features in data1b.
The used time windows can be set as 7 days, 30 days, 60 days, 90 days, 120 days, 150 days and 180 days.
We calculate the count of total and count of distinct of each raw field within selected time window.
我们使用180作为最大时间窗口来计算data1b中的一些功能。
使用时间窗口可以设置为7天，30天，60天，90天，120天，150天和180天。
我们计算所选时间窗口内每个原始字段的不同总计数和计数。
'''
print "我们使用180作为最大时间窗口来计算data1b中的一些功能。使用时间窗口可以设置为7天，30天，60天，90天，120天，150天和180天。我们计算所选时间窗口内每个原始字段的不同总计数和计数。"
time_window = [7, 30, 60, 90, 120, 150, 180]
var_list = ['LogInfo1', 'LogInfo2']
data1bGroupbyIdx = pd.DataFrame({'Idx': data1b['Idx'].drop_duplicates()})

for tw in time_window:
    data1b['TruncatedLogInfo'] = data1b['Listinginfo'].map(lambda x: x + datetime.timedelta(-tw))
    temp = data1b.loc[data1b['logInfo'] >= data1b['TruncatedLogInfo']]
    for var in var_list:
        # count the frequences of LogInfo1 and LogInfo2
        print '计数LogInfo1和LogInfo2的频率'+str(var)
        count_stats = temp.groupby(['Idx'])[var].count().to_dict()
        data1bGroupbyIdx[str(var) + '_' + str(tw) + '_count'] = data1bGroupbyIdx['Idx'].map(
            lambda x: count_stats.get(x, 0))

        # count the distinct value of LogInfo1 and LogInfo2
        # 计算LogInfo1和LogInfo2的不同值
        print '计算LogInfo1和LogInfo2的不同值'+str(var)
        Idx_UserupdateInfo1 = temp[['Idx', var]].drop_duplicates()
        uniq_stats = Idx_UserupdateInfo1.groupby(['Idx'])[var].count().to_dict()
        data1bGroupbyIdx[str(var) + '_' + str(tw) + '_unique'] = data1bGroupbyIdx['Idx'].map(
            lambda x: uniq_stats.get(x, 0))

        # calculate the average count of each value in LogInfo1 and LogInfo2
        # 计算LogInfo1和LogInfo2中每个值的平均计数
        print '计算LogInfo1和LogInfo2中每个值的平均计数'+str(var)
        data1bGroupbyIdx[str(var) + '_' + str(tw) + '_avg_count'] = data1bGroupbyIdx[
            [str(var) + '_' + str(tw) + '_count', str(var) + '_' + str(tw) + '_unique']]. \
            apply(lambda x: x[0] * 1.0 / x[1], axis=1)

data3b['ListingInfo'] = data3b['ListingInfo1'].map(lambda x: datetime.datetime.strptime(x, '%Y/%m/%d'))
data3b['UserupdateInfo'] = data3b['UserupdateInfo2'].map(lambda x: datetime.datetime.strptime(x, '%Y/%m/%d'))
data3b['ListingGap'] = data3b[['UserupdateInfo', 'ListingInfo']].apply(lambda x: (x[1] - x[0]).days, axis=1)
data3b['UserupdateInfo1'] = data3b['UserupdateInfo1'].map(ChangeContent)
data3bGroupbyIdx = pd.DataFrame({'Idx': data3b['Idx'].drop_duplicates()})

time_window = [7, 30, 60, 90, 120, 150, 180]
for tw in time_window:
    data3b['TruncatedLogInfo'] = data3b['ListingInfo'].map(lambda x: x + datetime.timedelta(-tw))
    temp = data3b.loc[data3b['UserupdateInfo'] >= data3b['TruncatedLogInfo']]

    # frequency of updating
    # 更新频率
    print '更新频率'+str(tw)
    freq_stats = temp.groupby(['Idx'])['UserupdateInfo1'].count().to_dict()
    data3bGroupbyIdx['UserupdateInfo_' + str(tw) + '_freq'] = data3bGroupbyIdx['Idx'].map(
        lambda x: freq_stats.get(x, 0))

    # number of updated types
    # 更新类型的数量
    print '更新类型的数量'+str(tw)
    Idx_UserupdateInfo1 = temp[['Idx', 'UserupdateInfo1']].drop_duplicates()
    uniq_stats = Idx_UserupdateInfo1.groupby(['Idx'])['UserupdateInfo1'].count().to_dict()
    data3bGroupbyIdx['UserupdateInfo_' + str(tw) + '_unique'] = data3bGroupbyIdx['Idx'].map(
        lambda x: uniq_stats.get(x, x))

    # average count of each type
    # 每种类型的平均计数
    print '每种类型的平均计数'+str(tw)
    data3bGroupbyIdx['UserupdateInfo_' + str(tw) + '_avg_count'] = data3bGroupbyIdx[
        ['UserupdateInfo_' + str(tw) + '_freq', 'UserupdateInfo_' + str(tw) + '_unique']]. \
        apply(lambda x: x[0] * 1.0 / x[1], axis=1)

    # whether the applicant changed items like IDNUMBER,HASBUYCAR, MARRIAGESTATUSID, PHONE
    # 申请人是否更改了IDNUMBER，HASBUYCAR，MARRIAGESTATUSID，PHONE等项目
    print '申请人是否更改了IDNUMBER，HASBUYCAR，MARRIAGESTATUSID，PHONE等项目'+str(tw)
    Idx_UserupdateInfo1['UserupdateInfo1'] = Idx_UserupdateInfo1['UserupdateInfo1'].map(lambda x: [x])
    Idx_UserupdateInfo1_V2 = Idx_UserupdateInfo1.groupby(['Idx'])['UserupdateInfo1'].sum()
    for item in ['_IDNUMBER', '_HASBUYCAR', '_MARRIAGESTATUSID', '_PHONE']:
        item_dict = Idx_UserupdateInfo1_V2.map(lambda x: int(item in x)).to_dict()
        data3bGroupbyIdx['UserupdateInfo_' + str(tw) + str(item)] = data3bGroupbyIdx['Idx'].map(
            lambda x: item_dict.get(x, x))

# Combine the above features with raw features in PPD_Training_Master_GBK_3_1_Training_Set
# 将上述功能与PPD_Training_Master_GBK_3_1_Training_Set中的原始功能相结合
print '将上述功能与PPD_Training_Master_GBK_3_1_Training_Set中的原始功能相结合'
allData = pd.concat([data2b.set_index('Idx'), data3bGroupbyIdx.set_index('Idx'), data1bGroupbyIdx.set_index('Idx')],
                    axis=1)
allData.to_csv(folderOfData + 'allData_0_Test.csv', encoding='gbk')
print '第一步：使用与训练数据集相同的功能来实现结束'
print '##########################################'
####################################################
# Step 2: Makeup missing value continuous variables
# 第二步：补全缺失值连续变量#
####################################################
print '##########################################'
print '第二步：补全缺失值连续变量开始'
# make some change to the string type varaiable, espeically converting nan to NAN so as it could be read in the mapping dictionary
# 对字符串类型进行一些更改，特别是将nan转换为NAN，以便可以在映射字典中读取
print '对字符串类型进行一些更改，特别是将nan转换为NAN，以便可以在映射字典中读取'
fread = open(folderOfData + 'numerical_var.pkl', 'r')
numerical_var = pickle.load(fread)
fread.close()

fread = open(folderOfData + 'categorical_var.pkl', 'r')
categorical_var = pickle.load(fread)
fread.close()


var_WOE_model = ['UserInfo_15_encoding_WOE', 'UserInfo_14_encoding_WOE', u'ThirdParty_Info_Period6_10_WOE',
                 u'ThirdParty_Info_Period5_2_WOE',
                 'UserInfo_16_encoding_WOE', 'WeblogInfo_20_encoding_WOE', u'WeblogInfo_6_WOE',
                 'UserInfo_19_encoding_WOE', u'UserInfo_17_WOE',
                 u'ThirdParty_Info_Period3_10_WOE', u'ThirdParty_Info_Period1_10_WOE', 'WeblogInfo_2_encoding_WOE',
                 'UserInfo_1_encoding_WOE']
raw_var = [i.replace('_WOE', '').replace('_encoding', '').replace('_mergeByBadRate', '') for i in var_WOE_model]
all_var = raw_var + ['Idx', 'target']

testData = pd.read_csv(folderOfData + 'allData_0_Test.csv', header=0, encoding='gbk')
testData = testData[all_var]

numerical_var_test = [i for i in raw_var if i in numerical_var]
categorical_var_test = [i for i in raw_var if i in categorical_var]

# makeup the missing values in the categorical variables, by using NAN to replace nan
# 通过使用NAN替代nan来分类分类变量中的缺失值
print '通过使用NAN替代nan来分类分类变量中的缺失值'
for col in categorical_var_test:
    if col != 'UserInfo_19':
        testData[col] = testData[col].map(lambda x: str(x).upper())

# makeup the missing values in the numerical variables, by using random sampling
# 通过使用随机抽样来弥补数值变量中的缺失值
print '通过使用随机抽样来弥补数值变量中的缺失值'
for col in numerical_var_test:
    missingRate = MissingContinuous(testData, col)
    if missingRate > 0:
        not_missing = testData.loc[testData[col] == testData[col]][col]
        # makeuped = allData[col].map(lambda x: MakeupRandom(x, list(not_missing)))
        missing_position = testData.loc[testData[col] != testData[col]][col].index
        not_missing_sample = random.sample(not_missing, len(missing_position))
        testData.loc[missing_position, col] = not_missing_sample
        missingRate2 = MissingContinuous(testData, col)
        print 'missing rate in {} after making up is:{}'.format(col, str(missingRate2))
print '##########################################'
############################################################
# Step 3: Map the raw values into bins and use WOE encoding
# 第三步：将原始值映射到bin中并使用WOE编码#
############################################################
print '##########################################'
print '第三步：将原始值映射到bin中并使用WOE编码开始'
### read the saved WOE encoding dictionary ###
# 读取保存的WOE编码字典
print '读取保存的WOE编码字典'
fread = open(folderOfData + 'var_WOE.pkl', 'r')
WOE_dict = pickle.load(fread)
fread.close()

fread = open(folderOfData + 'var_cutoff.pkl', 'r')
var_cutoff = pickle.load(fread)
fread.close()

fread = open(folderOfData + 'encoded_features.pkl', 'r')
encoded_features = pickle.load(fread)
fread.close()

### the below features are selected into the scorecard model in Step 5
# 在步骤5中将以下特征选入记分卡模型
print '在步骤5中将以下特征选入记分卡模型'

# some features are catgorical type and we need to encode them
# 一些功能是catgorical类型，我们需要对它们进行编码
print '一些功能是catgorical类型，我们需要对它们进行编码'
var_encoding = [i.replace('_WOE', '').replace('_encoding', '') for i in var_WOE_model if i.find('_encoding') >= 0]
for col in var_encoding:
    print col
    [col1, encode_dict] = encoded_features[col]
    if col == 'UserInfo_19':
        testData[col1] = testData[col].map(lambda x: encode_dict.get(x, -99999))
    else:
        testData[col1] = testData[col].map(lambda x: encode_dict.get(str(x), -99999))
    if -99999 in set(testData[col1]):
        print "some attributes in {} cannot be found in encoding dictionary".format(col)
    # testData[col1] = testData[col].map(lambda x: encode_dict[str(x)])
    col2 = str(col1) + "_WOE"
    cutOffPoints = var_cutoff[col1]
    special_attribute = []
    if - 1 in cutOffPoints:
        special_attribute = [-1]
    binValue = testData[col1].map(lambda x: AssignBin(x, cutOffPoints, special_attribute=special_attribute))
    testData[col2] = binValue.map(lambda x: WOE_dict[col1][x])

### WeblogInfo_20 has some exception and we have to remove it
# WeblogInfo_20有一些例外，我们必须删除它
print 'WeblogInfo_20有一些例外，我们必须删除它'
testData = testData.loc[(testData['WeblogInfo_20_encoding'] != -99999)]

### map the merged features into WOE
# fileread = open(folderOfData + 'merged_features.pkl', 'r')
# merged_features = pickle.load(fileread)
# fileread.close()
#
# var_merged = [i.replace('_WOE','').replace('_mergeByBadRate','') for i in var_WOE_model if i.find('_mergeByBadRate')>=0]
# for var in var_merged:
#     merging_map = merged_features[var][1]
#     testData[var+"_mergeByBadRate"] = testData[var].map(lambda x: merging_map.get(str(x),-99999))


# other features can be mapped to WOE directly
# 其他功能可以直接映射到WOE
print '其他功能可以直接映射到WOE'
var_others = [i.replace('_WOE', '').replace('_encoding', '') for i in var_WOE_model if i.find('_encoding') < 0]
for col in var_others:
    print col
    col2 = str(col) + "_WOE"
    if col in var_cutoff.keys():
        cutOffPoints = var_cutoff[col]
        special_attribute = []
        if - 1 in cutOffPoints:
            special_attribute = [-1]
        binValue = testData[col].map(lambda x: AssignBin(x, cutOffPoints, special_attribute=special_attribute))
        testData[col2] = binValue.map(lambda x: WOE_dict[col][x])
    else:
        testData[col2] = testData[col].map(lambda x: WOE_dict[col][x])
print '##########################################'
#######################################################
# Step 4: Use the trained model to make the prediction#
# 步骤4：使用训练有素的模型进行预测
#######################################################
print '##########################################'
print '步骤4：使用训练有素的模型进行预测开始'
### make the design matrix
# 做出设计矩阵
print '做出设计矩阵'
X = testData[var_WOE_model]
X['intercept'] = [1] * X.shape[0]
y = testData['target']
# very strange that some 0 and 1 are recgonized as '0' and '1'
# 非常奇怪的是，一些0和1被重新表示为'0'和'1'
print '非常奇怪的是，一些0和1被重新表示为0和1'
y = [int(i) for i in y]

#### load the training model
# 加载训练模型
print '加载训练模型'
saveModel = open(folderOfData + 'LR_Model_Normal.pkl', 'r')
LR = pickle.load(saveModel)
saveModel.close()
y_pred = LR.predict(X)

scorecard_result = pd.DataFrame({'prob': y_pred, 'target': y})
# we check the performance of the model using KS and AR
# 我们使用KS和AR检查模型的性能
print '我们使用KS和AR检查模型的性能'
# both indices should be above 30%
# 两个指标均应在30％以上
print '两个指标均应在30％以上'
performance = KS_AR(scorecard_result, 'prob', 'target')
print "KS and AR for the scorecard in the test dataset are %.0f%% and %.0f%%" % (
performance['KS'] * 100, performance['AR'] * 100)
print '##########################################'
##################################################
# Step 5: Convert the probability into the scores#
# 步骤5：将概率转换成分数
##################################################
print '##########################################'
print '步骤5：将概率转换成分数开始'
base_point = 500
PDO = 200
score = [int(base_point + PDO / np.log(2) * (-np.log(i / (1 - i)))) for i in y_pred]
print '最终结束'
print '##########################################'
