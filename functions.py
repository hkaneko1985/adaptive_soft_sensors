# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""
# サンプルプログラムで使われる関数群

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

def add_nonlinear_terms(x):
    """
    DataFrame型の x に、二乗項と交差項を追加して出力する関数

    Parameters
    ----------
    x: pandas.DataFrame

    Returns
    -------
    x: pandas.DataFrame
    
    """

    original_x = x.copy()  # 元の説明変数のデータセット
    x_square = x ** 2  # 二乗項
    # 追加
    print('\n二乗項と交差項の追加')
    for i in range(original_x.shape[1]):
        print(i + 1, '/', original_x.shape[1])
        for j in range(original_x.shape[1]):
            if i == j:  # 二乗項
                x = pd.concat(
                    [x, x_square.rename(columns={x_square.columns[i]: '{0}^2'.format(x_square.columns[i])}).iloc[:, i]],
                    axis=1)
            elif i < j:  # 交差項
                x = pd.concat([x, original_x.iloc[:, i] * original_x.iloc[:, j]], axis=1)
                x = x.rename(columns={0: '{0}*{1}'.format(x_square.columns[i], x_square.columns[j])})
    return x

def add_time_delayed_variable(x, dynamics_max, dynamics_span):
    """
    DataFrame型もしくは array 型の x に、時間遅れ変数を追加して出力する関数

    Parameters
    ----------
    x: pandas.DataFrame or numpy.array

    Returns
    -------
    x_with_dynamics: pandas.DataFrame or numpy.array
    
    """

    x_array = np.array(x)
    if dynamics_max:
        x_with_dynamics = np.empty((x_array.shape[0] - dynamics_max, 0 ))
        x_with_dynamics = np.append(x_with_dynamics, x_array[dynamics_max:, 0:1], axis=1)
        for x_variable_number in range(x_array.shape[1] - 1):
            x_with_dynamics = np.append(x_with_dynamics, x_array[dynamics_max:, x_variable_number+1:x_variable_number+2], axis=1)
            for time_delay_number in range(int(np.floor(dynamics_max / dynamics_span))):
                x_with_dynamics = np.append(x_with_dynamics, x_array[dynamics_max-(time_delay_number+1)*dynamics_span:-(time_delay_number+1)*dynamics_span, x_variable_number+1:x_variable_number+2], axis=1)
    else:
        x_with_dynamics = x_array.copy()
    return x_with_dynamics

def lwpls(x_train, y_train, x_test, max_component_number, lambda_in_similarity):
    """
    Locally-Weighted Partial Least Squares (LWPLS)
    
    Predict y-values of test samples using LWPLS

    Parameters
    ----------
    x_train: numpy.array or pandas.DataFrame
        autoscaled m x n matrix of X-variables of training data,
        m is the number of training sammples and
        n is the number of X-variables
    y_train: numpy.array or pandas.DataFrame
        autoscaled m x 1 vector of a Y-variable of training data
    x_test: numpy.array or pandas.DataFrame
        k x n matrix of X-variables of test data, which is autoscaled with training data,
        and k is the number of test samples
    max_component_number: int
        number of maximum components
    lambda_in_similarity: float
        parameter in similarity matrix

    Returns
    -------
    estimated_y_test : numpy.array
        k x 1 vector of estimated y-values of test data
    """

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    y_train = np.reshape(y_train, (len(y_train), 1))
    x_test = np.array(x_test)

    estimated_y_test = np.zeros((x_test.shape[0], max_component_number))
    distance_matrix = cdist(x_train, x_test, 'euclidean')
    for test_sample_number in range(x_test.shape[0]):
        query_x_test = x_test[test_sample_number, :]
        query_x_test = np.reshape(query_x_test, (1, len(query_x_test)))
        distance = distance_matrix[:, test_sample_number]
        similarity = np.diag(np.exp(-distance / distance.std(ddof=1) / lambda_in_similarity))
        #        similarity_matrix = np.diag(similarity)

        y_w = y_train.T.dot(np.diag(similarity)) / similarity.sum()
        x_w = np.reshape(x_train.T.dot(np.diag(similarity)) / similarity.sum(), (1, x_train.shape[1]))
        centered_y = y_train - y_w
        centered_x = x_train - np.ones((x_train.shape[0], 1)).dot(x_w)
        centered_query_x_test = query_x_test - x_w
        estimated_y_test[test_sample_number, :] += y_w
        for component_number in range(max_component_number):
            w_a = np.reshape(centered_x.T.dot(similarity).dot(centered_y) / np.linalg.norm(
                centered_x.T.dot(similarity).dot(centered_y)), (x_train.shape[1], 1))
            t_a = np.reshape(centered_x.dot(w_a), (x_train.shape[0], 1))
            p_a = np.reshape(centered_x.T.dot(similarity).dot(t_a) / t_a.T.dot(similarity).dot(t_a),
                             (x_train.shape[1], 1))
            q_a = centered_y.T.dot(similarity).dot(t_a) / t_a.T.dot(similarity).dot(t_a)
            t_q_a = centered_query_x_test.dot(w_a)
            estimated_y_test[test_sample_number, component_number:] = estimated_y_test[test_sample_number,
                                                                                       component_number:] + t_q_a * q_a
            if component_number != max_component_number:
                centered_x = centered_x - t_a.dot(p_a.T)
                centered_y = centered_y - t_a * q_a
                centered_query_x_test = centered_query_x_test - t_q_a.dot(p_a.T)

    return estimated_y_test
