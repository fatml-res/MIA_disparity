import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import os

from Tools import attr_map, find_all_results


def one_hot_transform(data, cate_index, s_ind=-2, y_ind=-1, with_group=True):
    # Input [X, s, y]
    group = data[:, 0].reshape(-1,1)
    X = data[:, 1:s_ind]
    s = data[:, s_ind].reshape(-1, 1)
    y = data[:, y_ind].reshape(-1, 1)
    tr_te = data[:, -1].reshape(-1, 1)
    result = np.hstack([group, s])
    for i in range(X.shape[1]):
        if i not in cate_index:
            tmp = data[:, i]
            result = np.hstack([result, tmp.reshape(-1, 1)])
        else:
            enc = OneHotEncoder(handle_unknown='ignore')
            enc.fit(data[:, i].reshape(-1, 1))
            tmp = enc.transform(data[:, i].reshape(-1, 1)).toarray()
            result = np.hstack([result, tmp])
    result = np.hstack([result, y, tr_te])
    #[g, s, X, y, tr/te]
    return result


def one_hot_transform2(X, cate_index):
    '''

    :param X: data, y is not included
    :param cate_index: list of categorical indexs
    :return: result: transformed data
             weight: weight list for each column. Used in Fairpick
             col_index: original column index
    '''
    result = []
    weight = []
    col_index = []
    for i in range(X.shape[1]):
        if i not in cate_index:
            tmp = X[:, i]
            result.append(tmp.reshape(-1, 1))
            weight.append(1)
            col_index.append(i)
        else:
            enc = OneHotEncoder(handle_unknown='ignore')
            enc.fit(X[:, i].reshape(-1, 1))
            tmp = enc.transform(X[:, i].reshape(-1, 1)).toarray()
            col_count = tmp.shape[1]
            weight = weight + [1/col_count]*col_count
            col_index = col_index + [i]*col_count
            result.append(tmp)
    result = np.hstack(result)
    return result, np.array(weight), np.array(col_index)



if __name__ == "__main__":
    files = find_all_results("Fairpick-bound", result_type="Fairpick_data_1")
    cat_inds = [[8, 1, 3, 5, 6, 7, 12],
                [0],
                [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]

    for i in range(3):
        for file in files:
            if ['Adult', 'Broward', 'Hospital'][i] in file:
                target_file = file.replace("Fairpick_data_1", "Oh_data_1")
                print("Transforming file {}".format(target_file))
                if os.stat(file).st_size == 0:
                    print("Warning! find a 0 size file:", file)
                    continue
                if os.path.exists(target_file):
                    print("Skipping file {} beacause it is done".format(file))
                    pass
                    #continue
                df = pd.read_csv(file, header=None)
                data = np.array(df)
                data_new = one_hot_transform(data, cat_inds[i], s_ind=-3, y_ind=-2)
                df_new = pd.DataFrame(data_new)
                if not os.path.exists(target_file):
                    df_new.to_csv(target_file, index=None, header=False)
