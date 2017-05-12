import pandas as pd
def Chi2(df, total_col, bad_col, overallRate):
    '''
    :param df: the dataset containing the total count and bad count
    :param total_col: total count of each value in the variable
    :param bad_col: bad count of each value in the variable
    :param overallRate: the overall bad rate of the training set
    :return: the chi-square value
    '''
    df2 = df.copy()
    df2['expected'] = df[total_col].apply(lambda x: x*overallRate)
    combined = zip(df2['expected'], df2[bad_col])
    chi = [(i[0]-i[1])**2/i[0] for i in combined]
    chi2 = sum(chi)
    return chi2


### ChiMerge_MaxInterval: split the continuous variable using Chi-square value by specifying the max number of intervals
def ChiMerge_MaxInterval(df, col, target, max_interval = 5):
    '''
    :param df: the dataframe containing splitted column, and target column with 1-0
    :param col: splitted column
    :param target: target column with 1-0
    :param max_interval: the maximum number of intervals. If the raw column has attributes less than this parameter, the function will not work
    :return: the combined bins
    '''
    colLevels = set(df[col])
    if len(colLevels) <= max_interval:  #If the raw column has attributes less than this parameter, the function will not work
        print "The number of original levels for {} is less than or equal to max intervals".format(col)
        return []
    else:
        #Step 1: group the dataset by col and work out the total count & bad count in each level of the raw column
        total = df.groupby([col])[target].count()
        total = pd.DataFrame({'total':total})
        bad = df.groupby([col])[target].sum()
        bad = pd.DataFrame({'bad':bad})
        regroup =  total.merge(bad,left_index=True,right_index=True, how='left')
        regroup.reset_index(level=0, inplace=True)
        N = sum(regroup['total'])
        B = sum(regroup['bad'])
        #the overall bad rate will be used in calculating expected bad count
        overallRate = B*1.0/N
        # since we always combined the neighbours of intervals, we need to sort the attributes
        colLevels =sorted(list(colLevels))
        # initially, each single attribute forms a single interval
        groupIntervals = [[i] for i in colLevels]
        groupNum = len(groupIntervals)
        while(len(groupIntervals)>max_interval):   #the termination condition: the number of intervals is equal to the pre-specified threshold
            # in each step of iteration, we calcualte the chi-square value of each atttribute
            chisqList = []
            for interval in groupIntervals:
                df2 = regroup.loc[regroup[col].isin(interval)]
                chisq = Chi2(df2, 'total','bad',overallRate)
                chisqList.append(chisq)
            #find the interval corresponding to minimum chi-square, and combine with the neighbore with smaller chi-square
            min_position = chisqList.index(min(chisqList))
            if min_position == 0:
                combinedPosition = 1
            elif min_position == groupNum - 1:
                combinedPosition = min_position -1
            else:
                if chisqList[min_position - 1]<=chisqList[min_position + 1]:
                    combinedPosition = min_position - 1
                else:
                    combinedPosition = min_position + 1
            groupIntervals[min_position] = groupIntervals[min_position]+groupIntervals[combinedPosition]
            # after combining two intervals, we need to remove one of them
            groupIntervals.remove(groupIntervals[combinedPosition])
            groupNum = len(groupIntervals)
        return groupIntervals


### ChiMerge_MaxInterval: split the continuous variable using Chi-square value by specifying the minimum chi-square value
def ChiMerge_MinChisq(df, col, target, confidenceVal = 3.841):
    '''
    :param df: the dataframe containing splitted column, and target column with 1-0
    :param col: splitted column
    :param target: target column with 1-0
    :param confidenceVal: the specified chi-square thresold, by default the degree of freedom is 1 and using confidence level as 0.95
    :return: the splitted bins
    '''
    colLevels = set(df[col])
    total = df.groupby([col])[target].count()
    total = pd.DataFrame({'total':total})
    bad = df.groupby([col])[target].sum()
    bad = pd.DataFrame({'bad':bad})
    regroup =  total.merge(bad,left_index=True,right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    N = sum(regroup['total'])
    B = sum(regroup['bad'])
    overallRate = B*1.0/N
    colLevels =sorted(list(colLevels))
    groupIntervals = [[i] for i in colLevels]
    groupNum  = len(groupIntervals)
    while(1):   #the termination condition: all the attributes form a single interval; or all the chi-square is above the threshould
        if len(groupIntervals) == 1:
            break
        chisqList = []
        for interval in groupIntervals:
            df2 = regroup.loc[regroup[col].isin(interval)]
            chisq = Chi2(df2, 'total','bad',overallRate)
            chisqList.append(chisq)
        min_position = chisqList.index(min(chisqList))
        if min(chisqList) >=confidenceVal:
            break
        if min_position == 0:
            combinedPosition = 1
        elif min_position == groupNum - 1:
            combinedPosition = min_position -1
        else:
            if chisqList[min_position - 1]<=chisqList[min_position + 1]:
                combinedPosition = min_position - 1
            else:
                combinedPosition = min_position + 1
        groupIntervals[min_position] = groupIntervals[min_position]+groupIntervals[combinedPosition]
        groupIntervals.remove(groupIntervals[combinedPosition])
        groupNum = len(groupIntervals)
    return groupIntervals

