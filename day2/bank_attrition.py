# coding=utf8
import pandas as pd
import numbers
import numpy as np
import math
from matplotlib import pyplot
from pandas.tools.plotting import scatter_matrix
import random
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from scipy.stats import chisquare

'''
:param df: 数据集包含数值独立变量和因变量the dataset containing numerical independent variable and dependent variable
:param col: 具有数值类型的自变量independent variable with numerical type
:param target: 因变量，0-1类dependent variable, class of 0-1
:param filepath: 我们保存直方图的位置the location where we save the histogram
:param truncation: 指出是否需要对异常值进行截断indication whether we need to do some truncation for outliers
:return: 描述统计the descriptive statistics
'''
def NumVarPerf(df, col, target, filepath, truncation=False):
    # extract target variable and specific indepedent variable
    # 提取目标变量和具体的独立变量
    validDf = df.loc[df[col] == df[col]][[col, target]]
    # the percentage of valid elements
    # 有效元素的百分比
    validRcd = validDf.shape[0] * 1.0 / df.shape[0]
    # format the percentage in the form of percent
    # 以百分比的形式格式化百分比
    validRcdFmt = "%.2f%%" % (validRcd * 100)
    # the descriptive statistics of each numerical column
    # 每个数值栏的描述性统计
    descStats = validDf[col].describe()
    mu = "%.2e" % descStats['mean']
    std = "%.2e" % descStats['std']
    maxVal = "%.2e" % descStats['max']
    minVal = "%.2e" % descStats['min']
    # we show the distribution by churn/not churn state
    # 我们通过流失/不流失状态显示分布
    x = validDf.loc[validDf[target] == 1][col]
    y = validDf.loc[validDf[target] == 0][col]
    xweights = 100.0 * np.ones_like(x) / x.size
    yweights = 100.0 * np.ones_like(y) / y.size
    # if need truncation, truncate the numbers in 95th quantile
    # 如果需要截断，则截断第95位数的数字
    if truncation == True:
        pcnt95 = np.percentile(validDf[col], 95)
        x = x.map(lambda x: min(x, pcnt95))
        y = y.map(lambda x: min(x, pcnt95))
    fig, ax = pyplot.subplots()
    ax.hist(x, weights=xweights, alpha=0.5, label='Attrition')
    ax.hist(y, weights=yweights, alpha=0.5, label='Retained')
    titleText = 'Histogram of ' + col + '\n' + 'valid pcnt =' + validRcdFmt + ', Mean =' + mu + ', Std=' + std + '\n max=' + maxVal + ', min=' + minVal
    ax.set(title=titleText, ylabel='% of Dataset in Bin')
    ax.margins(0.05)
    ax.set_ylim(bottom=0)
    pyplot.legend(loc='upper right')
    figSavePath = filepath + str(col) + '.png'
    pyplot.savefig(figSavePath)
    # pyplot.show()
    pyplot.close(1)

def NumVarPerf1(df, col, target, filepath, truncation=False):
    # extract target variable and specific indepedent variable
    # 提取目标变量和具体的独立变量
    validDf = df.loc[df[col] == df[col]][[col, target]]
    # the percentage of valid elements
    # 有效元素的百分比
    validRcd = validDf.shape[0] * 1.0 / df.shape[0]
    # format the percentage in the form of percent
    # 以百分比的形式格式化百分比
    validRcdFmt = "%.2f%%" % (validRcd * 100)
    # the descriptive statistics of each numerical column
    # 每个数值栏的描述性统计
    descStats = validDf[col].describe()
    mu = "%.2e" % descStats['mean']
    std = "%.2e" % descStats['std']
    maxVal = "%.2e" % descStats['max']
    minVal = "%.2e" % descStats['min']
    # we show the distribution by churn/not churn state
    # 我们通过流失/不流失状态显示分布
    x = validDf.loc[validDf[target] == 1][col]
    y = validDf.loc[validDf[target] == 0][col]
    xweights = 100.0 * np.ones_like(x) / x.size
    yweights = 100.0 * np.ones_like(y) / y.size
    # if need truncation, truncate the numbers in 95th quantile
    # 如果需要截断，则截断第95位数的数字
    if truncation == True:
        pcnt95 = np.percentile(validDf[col], 95)
        x = x.map(lambda x: min(x, pcnt95))
        y = y.map(lambda x: min(x, pcnt95))
    fig, ax = pyplot.subplots()
    ax.hist(x, weights=xweights, alpha=0.5, label='Attrition')
    ax.hist(y, weights=yweights, alpha=0.5, label='Retained')
    titleText = 'Histogram of ' + col + '\n' + 'valid pcnt =' + validRcdFmt + ', Mean =' + mu + ', Std=' + std + '\n max=' + maxVal + ', min=' + minVal
    ax.set(title=titleText, ylabel='% of Dataset in Bin')
    ax.margins(0.05)
    ax.set_ylim(bottom=0)
    pyplot.legend(loc='upper right')
    figSavePath = filepath + str(col) + '.png'
    pyplot.savefig(figSavePath)
    # pyplot.show()
    pyplot.close(1)

'''
:param df: the dataset containing numerical independent variable and dependent variable
:param col: independent variable with numerical type
:param target: dependent variable, class of 0-1
:param filepath: the location where we save the histogram
:return: the descriptive statistics
'''
def CharVarPerf(df, col, target, filepath):
    validDf = df.loc[df[col] == df[col]][[col, target]]
    validRcd = validDf.shape[0] * 1.0 / df.shape[0]
    recdNum = validDf.shape[0]
    validRcdFmt = "%.2f%%" % (validRcd * 100)
    freqDict = {}
    churnRateDict = {}
    # for each category in the categorical variable, we count the percentage and churn rate
    # 对于分类变量中的每个类别，我们计算百分比和流失率
    for v in set(validDf[col]):
        vDf = validDf.loc[validDf[col] == v]
        freqDict[v] = vDf.shape[0] * 1.0 / recdNum
        churnRateDict[v] = sum(vDf[target]) * 1.0 / vDf.shape[0]
    descStats = pd.DataFrame({'percent': freqDict, 'churn rate': churnRateDict})
    # 创建matplotlib图
    fig = pyplot.figure()  # Create matplotlib figure
    ax = fig.add_subplot(111)  # 创建matplotlib轴
    ax2 = ax.twinx()  # 创建另一个与ax共享x轴的轴
    pyplot.title('The percentage and churn rate for ' + col + '\n valid pcnt =' + validRcdFmt)
    descStats['churn rate'].plot(kind='line', color='red', ax=ax)
    descStats.percent.plot(kind='bar', color='blue', ax=ax2, width=0.2, position=1)
    ax.set_ylabel('churn rate')
    ax2.set_ylabel('percentage')
    figSavePath = filepath + str(col) + '.png'
    pyplot.savefig(figSavePath)
    # pyplot.show()
    pyplot.close(1)

def CharVarPerf1(df, col, target, filepath):
    validDf = df.loc[df[col] == df[col]][[col, target]]
    validRcd = validDf.shape[0] * 1.0 / df.shape[0]
    recdNum = validDf.shape[0]
    validRcdFmt = "%.2f%%" % (validRcd * 100)
    freqDict = {}
    churnRateDict = {}
    # for each category in the categorical variable, we count the percentage and churn rate
    # 对于分类变量中的每个类别，我们计算百分比和流失率
    for v in set(validDf[col]):
        vDf = validDf.loc[validDf[col] == v]
        freqDict[v] = vDf.shape[0] * 1.0 / recdNum
        churnRateDict[v] = sum(vDf[target]) * 1.0 / vDf.shape[0]
    descStats = pd.DataFrame({'percent': freqDict, 'churn rate': churnRateDict})
    # 创建matplotlib图
    fig = pyplot.figure()  # Create matplotlib figure
    ax = fig.add_subplot(111)  # 创建matplotlib轴
    ax2 = ax.twinx()  # 创建另一个与ax共享x轴的轴
    pyplot.title('The percentage and churn rate for ' + col + '\n valid pcnt =' + validRcdFmt)
    descStats['churn rate'].plot(kind='line', color='red', ax=ax)
    descStats.percent.plot(kind='bar', color='blue', ax=ax2, width=0.2, position=1)
    ax.set_ylabel('churn rate')
    ax2.set_ylabel('percentage')
    figSavePath = filepath + str(col) + '.png'
    pyplot.savefig(figSavePath)
    # pyplot.show()
    pyplot.close(1)

# 读取客户详细信息数据集中的数据：客户详细信息和字段字典
# 读取内部数据和外部数据
# 'path' is the path for students' folder which contains the lectures of xiaoxiang
print '读取客户详细信息数据集中的数据：客户详细信息和字段字典,读取内部数据和外部数据'
path = '/home/daoos/Documents/xiaoxiang_fintech/day2/';
bankChurn = pd.read_csv(path + '/data/bankChurn.csv', header=0)
externalData = pd.read_csv(path + '/data/ExternalData.csv', header=0)
# 合并数据
print '合并数据'
AllData = pd.merge(bankChurn, externalData, on='CUST_ID')

# 步骤1：检查每列的类型，并描述基本配置文件
print '步骤1：检查每列的类型，并描述基本配置文件'

columns = set(list(AllData.columns))
print '获取数据集中的所有字段：',columns
columns.remove('CHURN_CUST_IND')  # 变量不是我们的对象
print '移除指定字段CHURN_CUST_IND：',columns
# we differentiate the numerical type and catigorical type of the columns
# 我们区分列的数值类型和字符类型
numericCols = []
stringCols = []
print '循环处理字符类型和数值类型'
for var in columns:
    x = list(set(AllData[var]))
    x = [i for i in x if i == i]  # 我们需要消除噪音，这是nan型
    if isinstance(x[0], numbers.Real):
        print var, '是数值类型'
        numericCols.append(var)
        print 'numericCols的值：',numericCols
    elif isinstance(x[0], str):
        print var, '是String类型'
        stringCols.append(var)
        print 'stringCols的值：', stringCols
    else:
        print 'The type of ', var, ' cannot be determined'

# Part 1: Single factor analysis for independent variables
# 第1部分：独立变量的单因素分析
# we check the distribution of each numerical variable, separated by churn/not churn
# 我们检查每个数值变量的分布，由流失/不是流失分隔
print '==============================='
print '第1部分：独立变量的单因素分析'
print '我们检查每个数值变量的分布，由流失/不是流失分隔'
filepath = path + '/Notes/Pictures1/'
for var in numericCols:
    NumVarPerf(AllData, var, 'CHURN_CUST_IND', filepath)

# need to do some truncation for outliers
# 需要对异常值进行一些截断
filepath = path + '/Notes/Pictures2/'
for val in numericCols:
    NumVarPerf(AllData, val, 'CHURN_CUST_IND', filepath, True)

# anova test
anova_results = anova_lm(ols('ASSET_MON_AVG_BAL~CHURN_CUST_IND', AllData).fit())

# single factor analysis for categorical analysis
# 单因素分析用于分类分析
filepath = path + '/Notes/Pictures3/'
for val in stringCols:
    print val
    CharVarPerf(AllData, val, 'CHURN_CUST_IND', filepath)

# chisquare test卡方检验
chisqDf = AllData[['GENDER_CD', 'CHURN_CUST_IND']]
grouped = chisqDf['CHURN_CUST_IND'].groupby(chisqDf['GENDER_CD'])
count = list(grouped.count())
churn = list(grouped.sum())
chisqTable = pd.DataFrame({'total': count, 'churn': churn})
chisqTable['expected'] = chisqTable['total'].map(lambda x: round(x * 0.101))
chisqValList = chisqTable[['churn', 'expected']].apply(lambda x: (x[0] - x[1]) ** 2 / x[1], axis=1)
chisqVal = sum(chisqValList)
# the 2-degree of freedom chisquare under 0.05 is 5.99, which is smaller than chisqVal = 32.66, so GENDER is significant
# 小于0.05的2自由度小于5.99，小于chisqVal = 32.66，所以性别显着,或者，我们可以直接使用函数
# Alternatively, we can use function directly
chisquare(chisqTable['churn'], chisqTable['expected'])

# Part 1: Multi factor analysis for independent variables
# 第1部分：自变量的多因素分析
# use short name to replace the raw name, since the raw names are too long to be shown
# 使用短名称替换原始名称，因为原始名称太长，无法显示
col_to_index = {numericCols[i]: 'var' + str(i) for i in range(len(numericCols))}
# sample from the list of columns, since too many columns cannot be displayed in the single plot
# 列的列表中的样本，因为在单个图中无法显示太多的列
corrCols = random.sample(numericCols, 15)
sampleDf = AllData[corrCols]
for col in corrCols:
    sampleDf.rename(columns={col: col_to_index[col]}, inplace=True)
scatter_matrix(sampleDf, alpha=0.2, figsize=(6, 6), diagonal='kde')
# pyplot.scatter(sampleDf, alpha=0.2, figsize=(6, 6), diagonal='kde')
pyplot.show()
print '+++++'
