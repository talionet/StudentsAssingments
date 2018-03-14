import numpy as np
import pandas as pd
from pandas import DataFrame as df
import sys
from scipy.spatial import distance
from scipy.stats import chi2_contingency


def df_print(data, index=None, columns=None, title='/n***'):
    """ changes data index and columns for printing"""
    print_data=data.copy()
    if index is not None:
        print_data.index=index
    if columns is not None:
        print_data.columns=columns
    print(title)
    print(print_data)

def pivot(raw_data, index_col,columns_col, values_col, is_agg=False, agg_function='mean', dropna=True, fill_value=None, head=None, convert_to_numeric=False):
    """ preprocessing and pivoting of raw_data"""
    #prepares a table.
    if head is not None:
        raw_data=raw_data.head(head)

    if is_agg:
        X= pd.pivot_table(raw_data,values=values_col,index=index_col, columns=columns_col, aggfunc=agg_function)
    else: #for non numeric data
        data_without_duplicates=raw_data.drop_duplicates([index_col,columns_col], keep=agg_function)
        print('%i/%i duplicated rows were removed (keep=%s)' %(len(raw_data)-len(data_without_duplicates), len(raw_data),agg_function))
        X=data_without_duplicates.pivot(index=index_col, columns=columns_col, values=values_col)

    if dropna:
        X=X.dropna(how='all')
    if fill_value is not None:
        X.fillna(fill_value)
    if convert_to_numeric:
        X =X.apply(pd.to_numeric, args=('coerce',))

    return X

def count_data(data, value=None, index='﻿sSchoolName', column='Class', repeats=0):
    """returns pivot table with counts as agg_func"""
    if value is None:
        data=data.dropna(how='all')
    else:
        data=data.loc[data[value].dropna().index]
        data.columns=data.columns.astype(str)
        print(data.head())
    return pd.pivot_table(data[[index,column]], index=index, columns=column,aggfunc=len)

def mask_data(data, type='', makeZero=np.nan, make_plus=1, make_minus=0):
    data=df(data)
    #input - data of one type
    if type=='':
        return data
    if type=='bool':
        return pd.notnull(data)
    elif type=='nan_to_zero':
        data= data.applymap(lambda x: -1 if x==0 else x) #convert 0 to -1
        return data.fillna(0.)

def define_distance_metric(distance_name='sum_joint_questions', min_intersection=10):
    """returns a distance function based on speciied name"""
    def _validate_vector(u, dtype=None):
        # XXX Is order='c' really necessary?
        u = np.asarray(u, dtype=dtype, order='c').squeeze()
        # Ensure values such as u=1 and u=[1] still return 1-D arrays.
        u = np.atleast_1d(u)
        if u.ndim > 1:
            raise ValueError("Input vector should be 1-D.")
        return u

    drop_na = lambda x: pd.Series(x).dropna().index


    if distance_name == 'sum_joint_questions':
        # number of items which are not Nan for BOTH u and v.
        def distfunc(u, v):
            joint_questions=set(drop_na(u)).intersection(set(drop_na(v)))
            return len(joint_questions)

    elif distance_name == 'jaccard_intersection':
        #number of non-similar responses of all responses different from zero (for BOTH v, u)
        def distfunc(u,v):
            u = _validate_vector(u)
            v = _validate_vector(v)
            if np.double(np.bitwise_and(u != 0, v != 0).sum())< min_intersection:
                return np.nan
            else:
                dist = (np.double(np.bitwise_and((u != v),
                                                 np.bitwise_or(u != 0, v != 0)).sum()) /
                        np.double(np.bitwise_and(u != 0, v != 0).sum()))
                return dist


    elif distance_name== 'joint_minus_count_drop0':
        def distfunc(u,v):
            u = _validate_vector(u)
            v = _validate_vector(v)
            if np.double(np.bitwise_and(u != 0, v != 0).sum()) < min_intersection:
                return np.nan
            elif np.bitwise_and(u == -1,v == -1).sum() == 0:
                return 1
            else:
                return  -(np.double(np.bitwise_and(u == -1,v == -1).sum()))


    elif distance_name== 'joint_minus_percent_drop0':
        def distfunc(u,v):
            u = _validate_vector(u)
            v = _validate_vector(v)
            if np.double(np.bitwise_and(u != 0, v != 0).sum())< min_intersection:
                return np.nan
            elif np.bitwise_and(u == -1,v == -1).sum()==0:
                return np.nan
            else:
                return  np.double(np.bitwise_and(u ==-1, v == -1).sum()) / np.double(np.bitwise_and(u == -1, v == -1).sum())

    return distfunc

def normalize_data(data, min_value=0, by='min_max', fillna=None):
    """
    normalize data by specified method (deafult min_max)
    :param data:
    :param min_value: set the minimum value the return table can have
    :param by: normalization method
    :param fillna: select if to fill missing values
    :return: normalized data
    """
    data = df(data)
    is_series= data.shape[1]==1
    if by is None:
        ndata=data
    if by=='min_max':
        ndata = (data - data.mean()) / (data.max() - data.min())
    if by == 'standard':
        ndata = (data - data.mean()) / (data.std())

    if min_value is not None:
        ndata= ndata - ndata.min() + min_value

    if fillna is not None:
        if type(fillna) == float or type(fillna) == int:
            ndata = ndata.fillna(fillna)
        if fillna == 'max_and_std':
            ndata = ndata.fillna(ndata.max() + 2 * ndata.std())
        elif fillna == 'max':
            ndata = ndata.fillna(ndata.max())

    if is_series:
        return ndata[0]
    else:
        return ndata

def pairwise_dist(X,metric):
    """ calcs pairwise distance between X rows based on distance function (metric)"""
    X = np.asarray(X, order='c')
    s = X.shape
    if len(s) == 2:

        m, n = s
        dm = np.zeros((m * (m - 1)) // 2, dtype=np.double)
        k = 0
        for i in range(0, m - 1):
            for j in range(i + 1, m):
                dm[k] = metric(X[i], X[j])
                k = k + 1
            if i%100==0:
                df(dm).to_csv('temp_similarity_matrix_%i.csv' %i)
    else:
        dm = df(columns=X,index=X)
        i = 0
        X_not_calculated=list(X.copy())
        for d in X:
            print(i)
            dist = [None for s in range(i)]+[metric(d, d2) for d2 in X_not_calculated]
            dm.loc[d]=dist
            X_not_calculated.remove(d)
            i+=1
            if i%100==0:
                dm.to_csv('temp_similarity_matrix_%i.csv' %i)
    return dm

def remove_duplicates(data, keep='first'):
    """ remove duplicated indeces from data"""
    duplicated_students=data.index.get_duplicates()
    if len(duplicated_students)>0:
        print('--- %i duplicated indeces found : ---- ' %len(duplicated_students))
        print(data.loc[duplicated_students])
    if keep=='first':
        data_without_duplicates=data.groupby(data.index).first()
    return data_without_duplicates

def chi2_test(data, var1, var2):
    """ compares the observed counts in categories var1 and var2 with the expected counts based on marginal distribution:
     - statistican difference - chi2 contingency test between observed and expected
     - absolute difference - observed - expected diff
    """
    test_vars = pd.Series(
        list(zip(data[var1], data[var2])),
        index=data.index)
    n=len(test_vars)
    observed_counts = test_vars.value_counts()
    observed_counts.index = pd.MultiIndex.from_tuples(observed_counts.index)
    observed_counts = observed_counts.unstack().fillna(0.)

    observed_f = observed_counts / n
    #observed_f.index = pd.MultiIndex.from_tuples(observed_f.index)
    #observed_f = observed_f.unstack().fillna(0.)
    chi2, p, dof, expected_f = chi2_contingency(observed_f)
    g, pg, dofg, expected = chi2_contingency(observed_f, lambda_="log-likelihood")


    if p>0.05:
        conclusion="No significant statistical difference was found\n between the expected and observed %s-%s distributions \n (chi2=%f, p=%f, dof=%i)" % (var1, var2,chi2, p, dof)
    if p < 0.05:
        conclusion = "A SIGNIFICANT statistical difference was found\n between the expected and observed %s-%s distributions \n (chi2=%f, p=%f, dof=%i)" % (
        var1, var2, chi2, p, dof)
    expected_counts=df(expected_f * n, index=observed_counts.index, columns=observed_counts.columns)
    observed_diff=observed_counts - expected_f*n
    return observed_counts, expected_counts, chi2, p, dof , conclusion

def plot_grouped_data(grouped_data, index='ExpGroup', column='pretest_score', n_groups=n_groups):
    fig, axes = plt.subplots(1, n_groups, figsize=(10, 3))
    #num_of_bins=grouped_data.groupby(index).count().min()-2
    for (group_i, group), ax in zip(grouped_data.groupby(index), axes.flatten()):
        group[column].plot(kind='hist',title=group_i,ax=ax,ylim=[0,25], bins=100)
        #ax.title=group_name
        #ax.set_ylim(0, 25)
    plt.tight_layout()
    plt.suptitle=column
    plt.savefig(os.path.join(output_dir, '%s_hist_%s.png' %(now.strftime("%Y%m%d-%H%M%S"),column)))
    plt.close()



    '''data= grouped_data[[index, column]]
    data = pd.pivot_table(grouped_data[[index, column]], index=index, columns=column, aggfunc=len)
    data.T.plot(type='bar',subplots=True)

    for group, data in grouped_data.groupby(index):
        data.index=groups_names
    data.plot()'''