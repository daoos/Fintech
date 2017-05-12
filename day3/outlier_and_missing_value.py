# coding=utf8
import pandas as pd
# from matplotlib import pyplot as plt
import random
from numpy import *
import operator
import numbers
import datetime
import time


### The function making up missing values in Continuous or Categorical variable
# 在“连续”或“分类”变量中组成缺失值的函数
def MakeupMissing(df, col, type, method):
    '''
    :param df: 数据集包含缺少值的列dataset containing columns with missing value
    :param col: 缺少值的列columns with missing value
    :param type: 列的类型应为连续或分类the type of the column, should be Continuous or Categorical
    :return: 组成的列the made up columns
    '''
    # Take the sample with non-missing value in col
    # 以col中的非缺失值取样本
    # validDf = df.loc[df[col] == df[col]][[col]]
    if type == 'Continuous':
        validDf = df.loc[~isnan(df[col])][[col]]
    elif type == 'Categorical':
        validDf = df.loc[df[col] == df[col]][[col]]
    if validDf.shape[0] == df.shape[0]:
        print 'There is no missing value in {}'.format(col)
        return df[col]
    # copy the original value from col to protect the original dataframe
    # 从col复制原始值以保护原始数据帧
    missingList = [i for i in df[col]]
    if type == 'Continuous':
        if method not in ['Mean', 'Random']:
            print 'Please specify the correct treatment method for missing continuous variable!'
            return df[col]
        # get the descriptive statistics of col
        # 获取col的描述性统计信息
        descStats = validDf[col].describe()
        print "获取col的描述性统计信息", descStats
        mu = descStats['mean']
        std = descStats['std']
        maxVal = descStats['max']
        # detect the extreme value using 3-sigma method
        # 使用3-sigma方法检测极值
        if maxVal > mu + 3 * std:
            for i in list(validDf.index):
                if validDf.loc[i][col] > mu + 3 * std:
                    # decrease the extreme value to normal level
                    # 将极值降至正常水平
                    validDf.loc[i][col] = mu + 3 * std
            # re-calculate the mean based on cleaned data
            # 基于清理数据重新计算平均值
            mu = validDf[col].describe()['mean']
        for i in range(df.shape[0]):
            if isnan(df.loc[i][var]):
                # use the mean or sampled data to replace the missing value
                # 使用平均值或采样数据替换缺失值
                if method == 'Mean':
                    missingList[i] = mu
                elif method == 'Random':
                    missingList[i] = random.sample(validDf[col], 1)[0]
    elif type == 'Categorical':
        if method not in ['Mode', 'Random']:
            return 'Please specify the correct treatment method for missing categorical variable!'
        # calculate the probability of each type of the categorical variable
        # 计算每种类型的分类变量的概率
        freqDict = {}
        recdNum = validDf.shape[0]
        for v in set(validDf[col]):
            vDf = validDf.loc[validDf[col] == v]
            freqDict[v] = vDf.shape[0] * 1.0 / recdNum
        # find the category with highest probability
        # 找到概率最高的类别
        modeVal = max(freqDict.items(), key=lambda x: x[1])[0]
        freqTuple = freqDict.items()
        # cumulative sum of each category
        # 每个类别的累积和
        freqList = [0] + [i[1] for i in freqTuple]
        freqCumsum = cumsum(freqList)
        for i in range(df.shape[0]):
            if df.loc[i][col] != df.loc[i][col]:
                if method == 'Mode':
                    missingList[i] = modeVal
                if method == 'Random':
                    # determine the sampled category using unifor distributed random variable
                    # 使用unifor分布随机变量确定抽样类别
                    a = random.random(1)
                    position = [k + 1 for k in range(len(freqCumsum) - 1) if freqCumsum[k] < a <= freqCumsum[k + 1]][0]
                    missingList[i] = freqTuple[position - 1][0]
    print 'The missing value in {0} has been made up with the mothod of {1}'.format(col, method)
    return missingList


### Use numerical representative for ategorical variable
# 使用数值代表的代数变量
def Encoder(df, col, target):
    '''
    # :param df: 数据集包含分类变量the dataset containing categorical variable
    :param col: 分类变量的名称the name of categorical variabel
    :param target: class, with value 1 or 0
    :return: 分类变量的数值编码the numerical encoding for categorical variable
    '''
    encoder = {}
    for v in set(df[col]):
        if v == v:
            subDf = df[df[col] == v]
        else:
            xList = list(df[col])
            nanInd = [i for i in range(len(xList)) if xList[i] != xList[i]]
            subDf = df.loc[nanInd]
        encoder[v] = sum(subDf[target]) * 1.0 / subDf.shape[0]
    newCol = [encoder[i] for i in df[col]]
    return newCol


### convert the date variable into the days
# 将日期变量转换为日期
def Date2Days(df, dateCol, base):
    '''
    :param df: 该数据集包含日期变量的格式为2017/1/1the dataset containing date variable in the format of 2017/1/1
    :param date: 日期列the column of date
    :param base: 用于计算日差的基准日期the base date used in calculating day gap
    :return: 天差距the days gap
    '''
    base2 = time.strptime(base, '%Y/%m/%d')
    base3 = datetime.datetime(base2[0], base2[1], base2[2])
    date1 = [time.strptime(i, '%Y/%m/%d') for i in df[dateCol]]
    date2 = [datetime.datetime(i[0], i[1], i[2]) for i in date1]
    daysGap = [(date2[i] - base3).days for i in range(len(date2))]
    return daysGap


### Calculate the ratio between two variables
# 计算两个变量之间的比率
def ColumnDivide(df, colNumerator, colDenominator):
    '''
    :param df: 包含变量x＆y的数据集the dataframe containing variable x & y
    :param colNumerator: 分子变量x the numerator variable x
    :param colDenominator: 分母变量ythe denominator variable y
    :return: x/y
    '''
    N = df.shape[0]
    rate = [0] * N
    xNum = list(df[colNumerator])
    xDenom = list(df[colDenominator])
    for i in range(N):
        # if the denominator is non-zero, work out the ratio
        if xDenom[i] > 0:
            rate[i] = xNum[i] * 1.0 / xDenom[i]
        # if the denominator is zero, assign 0 to the ratio
        else:
            rate[i] = 0
    return rate


print '================================'
path = '/home/daoos/Documents/xiaoxiang_fintech/day3/';
bankChurn = pd.read_csv(path + '/data/bankChurn.csv', header=0)
externalData = pd.read_csv(path + '/data/ExternalData.csv', header=0)
# merge two dataframes
AllData = pd.merge(bankChurn, externalData, on='CUST_ID')
print '合并数据集'

modelData = AllData.copy()
# convert date to days, using minimum date 1999/1/1 as the base to calculate the gap
# 将日期转换为天数，使用最小日期1999/1/1作为计算差距的基数
print '将日期转换为天数，使用最小日期1999/1/1作为计算差距的基数'
modelData['days_from_open'] = Date2Days(modelData, 'open_date', '1999/1/1')
del modelData['open_date']
indepCols = list(modelData.columns)
indepCols.remove('CHURN_CUST_IND')
indepCols.remove('CUST_ID')

except_var = []
for var in indepCols:
    try:
        x0 = list(set(modelData[var]))
        # 有什么问题forgntvl，我不知道如何处理它，所以使用这个傻瓜方法
        if var == 'forgntvl':  # something wrong with forgntvl, and I don't know how to deal with it so use this fool method~~~
            a = [i for i in modelData['forgntvl']]
            for i in range(len(a)):
                if a[i] not in (0, 1):
                    a[i] = -1
            del modelData['forgntvl']
            modelData['forgntvl'] = a
            x0 = list(set(modelData[var]))
        if len(x0) == 1:
            print 'Remove the constant column {}'.format(var)
            indepCols.remove(var)
            continue
        x = [i for i in x0 if i == i]
        # 我们需要消除噪音，这是nan型
        print '我们需要消除噪音，这是nan型'
        if isinstance(x[0], numbers.Real) and len(x) > 4:
            print 'nan is found in column {}, so we need to make up the missing value'.format(var)
            modelData[var] = MakeupMissing(modelData, var, 'Continuous', 'Mean')
        else:
            # 对于分类变量，在这个时刻，我们不化妆缺失的值。相反，我们认为失踪是一种特殊的类型
            # for categorical variable, at this moment we do not makeup the missing value. Instead we think the missing as a special type
            # if nan in x0:
            # print 'nan is found in column {}, so we need to make up the missing value'.format(var)
            # modelData[var] = MakeupMissing(modelData, var, 'Categorical', 'Random')
            print '对于分类变量，在这个时刻，我们不化妆缺失的值。相反，我们认为失踪是一种特殊的类型'
            print 'Encode {} using numerical representative'.format(var)
            modelData[var] = Encoder(modelData, var, 'CHURN_CUST_IND')
    except:
        print "something is wrong with {}".format(var)
        except_var.append(var)
        continue

#### 1: creating features : max of all
print '创建最大值的features'
maxValueFeatures0 = ['LOCAL_CUR_SAV_SLOPE', 'LOCAL_BELONEYR_FF_SLOPE', 'LOCAL_OVEONEYR_FF_SLOPE', 'LOCAL_SAV_SLOPE',
                     'SAV_SLOPE']
maxValueFeatures1 = ['avg3mou', 'avg6mou', 'avgmou']
maxValueFeatures2 = ['avg3qty', 'avg6qty', 'avgqty']

modelData['volatilityMax'] = modelData[maxValueFeatures0].apply(max, axis=1)
modelData['avgmouMax'] = modelData[maxValueFeatures1].apply(max, axis=1)
modelData['avgqtyMax'] = modelData[maxValueFeatures2].apply(max, axis=1)

#### 2: deleting features : some features are coupling so we need to delete the redundant
# 删除features：一些features是耦合的，所以我们需要删除冗余
print '删除features：一些features是耦合的，所以我们需要删除冗余'
del modelData['LOCAL_CUR_MON_AVG_BAL_PROP']

#### 3: sum up features: some features can be summed up to work out a total number
# 总结特点：一些功能可以总结出一个总数
print '总结特点：一些功能可以总结出一个总数'
sumupCols0 = ['LOCAL_CUR_MON_AVG_BAL', 'LOCAL_FIX_MON_AVG_BAL']
sumupCols1 = ['LOCAL_CUR_WITHDRAW_TX_NUM', 'LOCAL_FIX_WITHDRAW_TX_NUM']
sumupCols2 = ['LOCAL_CUR_WITHDRAW_TX_AMT', 'LOCAL_FIX_WITHDRAW_TX_AMT']
sumupCols3 = ['COUNTER_NOT_ACCT_TX_NUM', 'COUNTER_ACCT_TX_NUM']
sumupCols4 = ['ATM_ALL_TX_NUM', 'COUNTER_ALL_TX_NUM']
sumupCols5 = ['ATM_ACCT_TX_NUM', 'COUNTER_ACCT_TX_NUM']
sumupCols6 = ['ATM_ACCT_TX_AMT', 'COUNTER_ACCT_TX_AMT']
sumupCols7 = ['ATM_NOT_ACCT_TX_NUM', 'COUNTER_NOT_ACCT_TX_NUM']
sumupCols8 = ['COUNTER_ALL_TX_NUM', 'ATM_ALL_TX_NUM']
sumupCols9 = ['mouiwylisv_Mean', 'mouowylisv_Mean']
sumupCols10 = ['iwylis_vce_Mean', 'owylis_vce_Mean']
sumupCols11 = ['unan_dat_Mean', 'unan_vce_Mean']
sumupCols12 = ['mou_opkv_Mean', 'mou_peav_Mean']
sumupCols13 = ['peak_vce_Mean', 'peak_dat_Mean']

modelData['TOTAL_LOCAL_MON_AVG_BAL'] = modelData[sumupCols0].apply(sum, axis=1)
modelData['TOTAL_WITHDRAW_TX_NUM'] = modelData[sumupCols1].apply(sum, axis=1)
modelData['TOTAL_WITHDRAW_TX_AMT'] = modelData[sumupCols2].apply(sum, axis=1)
modelData['TOTAL_COUNTER_TX_NUM'] = modelData[sumupCols3].apply(sum, axis=1)
modelData['TOTAL_ALL_TX_NUM'] = modelData[sumupCols4].apply(sum, axis=1)
modelData['TOTAL_ACCT_TX_NUM'] = modelData[sumupCols5].apply(sum, axis=1)
modelData['TOTAL_ACCT_TX_AMT'] = modelData[sumupCols6].apply(sum, axis=1)
modelData['TOTAL_NOT_ACCT_TX_NUM'] = modelData[sumupCols7].apply(sum, axis=1)
modelData['TOTAL_TX_NUM'] = modelData[sumupCols8].apply(sum, axis=1)
modelData['TOTAL_moulisv_Mean'] = modelData[sumupCols9].apply(sum, axis=1)
modelData['TOTAL_lis_vce_Mean'] = modelData[sumupCols10].apply(sum, axis=1)
modelData['TOTAL_unan_Mean'] = modelData[sumupCols11].apply(sum, axis=1)
modelData['TOTAL_mou_Mean'] = modelData[sumupCols12].apply(sum, axis=1)
modelData['TOTAL_peak_Mean'] = modelData[sumupCols13].apply(sum, axis=1)

### creating features 3: ratio
print '创建特征3：比例'
numeratorCols = ['LOCAL_SAV_CUR_ALL_BAL', 'SAV_CUR_ALL_BAL', 'ASSET_CUR_ALL_BAL', 'LOCAL_CUR_WITHDRAW_TX_NUM',
                 'LOCAL_CUR_WITHDRAW_TX_AMT', 'COUNTER_NOT_ACCT_TX_NUM',
                 'ATM_ALL_TX_NUM', 'ATM_ACCT_TX_AMT', 'ATM_NOT_ACCT_TX_NUM', 'LOCAL_CUR_MON_AVG_BAL',
                 'ASSET_MON_AVG_BAL', "LOCAL_CUR_TRANS_TX_AMT",
                 "LOCAL_CUR_LASTSAV_TX_AMT", "L6M_INDFINA_ALL_TX_AMT", "LOCAL_CUR_WITHDRAW_TX_AMT",
                 "POS_CONSUME_TX_AMT", "LOCAL_FIX_OPEN_ACC_TX_AMT",
                 "ATM_ACCT_TX_AMT", "LOCAL_FIX_WITHDRAW_TX_AMT", "LOCAL_FIX_CLOSE_ACC_TX_AMT", "COUNTER_ACCT_TX_AMT",
                 "TELEBANK_ALL_TX_NUM",
                 "COUNTER_ALL_TX_NUM", "L6M_INDFINA_ALL_TX_NUM", "L6M_INDFINA_ALL_TX_AMT",
                 'threeway_Mean', 'comp_vce_Mean', 'mou_opkv_Mean', 'vceovr_Range', 'da_Mean', 'mouiwylisv_Mean',
                 'iwylis_vce_Mean', 'unan_dat_Mean',
                 'mou_opkv_Mean', 'peak_vce_Mean']
denominatorCols = ['LOCAL_SAV_MON_AVG_BAL', 'SAV_MON_AVG_BAL', 'ASSET_MON_AVG_BAL', 'TOTAL_WITHDRAW_TX_NUM',
                   'TOTAL_WITHDRAW_TX_AMT', 'TOTAL_COUNTER_TX_NUM',
                   'TOTAL_ACCT_TX_NUM', 'TOTAL_ACCT_TX_AMT', 'TOTAL_NOT_ACCT_TX_NUM', 'LOCAL_CUR_ACCT_NUM',
                   'ASSET_CUR_ALL_BAL', "LOCAL_CUR_TRANS_TX_NUM",
                   "LOCAL_CUR_LASTSAV_TX_NUM", "L6M_INDFINA_ALL_TX_NUM", "LOCAL_CUR_WITHDRAW_TX_NUM",
                   "POS_CONSUME_TX_NUM", "LOCAL_FIX_OPEN_ACC_TX_NUM",
                   "ATM_ACCT_TX_NUM", "LOCAL_FIX_WITHDRAW_TX_NUM", "LOCAL_FIX_CLOSE_ACC_TX_NUM", "COUNTER_ACCT_TX_NUM",
                   "TOTAL_TX_NUM", "TOTAL_TX_NUM",
                   "TOTAL_TX_NUM", "ASSET_CUR_ALL_BAL",
                   'totcalls', 'comp_dat_Mean', 'mou_cvce_Mean', 'rev_Range', 'totcalls', 'TOTAL_moulisv_Mean',
                   'TOTAL_lis_vce_Mean', 'TOTAL_unan_Mean',
                   'TOTAL_mou_Mean', 'TOTAL_peak_Mean']

newColName = ["RATIO_" + str(i) for i in range(len(numeratorCols))]
for i in range(len(numeratorCols)):
    modelData[newColName[i]] = ColumnDivide(modelData, numeratorCols[i], denominatorCols[i])

modelData.to_csv(path + '/data/modelData.csv', index=False)
print '==============End================'
