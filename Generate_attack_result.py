import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from NN_DP import attack_model
from Tools import find_all_results
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == "__main__":
    with_group=True
    tf.config.threading.set_intra_op_parallelism_threads(10)
    tf.config.threading.set_inter_op_parallelism_threads(10)
    files = find_all_results("Fairpick-bound", result_type="target_result_1.csv")
    replace = 0
    for file in files:
        folder = file.split("/target_result_1.csv")[0]
        if "Adult" not in file or float(file.split('/')[1].replace('n=',''))<0.13:
            pass
            print("Skip {} because we are not working on this dataset/setting".format(file))
            continue
        if os.path.exists(folder + "/attack_result_1.csv") and replace == 0:
            print("Skipped " + file)
            continue
        else:
            print("Start working on " + file)
        df = pd.read_csv(file, header=None)
        if len(df) < 2000:
            continue
        data = np.array(df)
        data_train = data[data[:, -1]==1]
        data_test = data[data[:, -1]==0]
        np.random.shuffle(data_test)
        data_final = np.vstack([data_train, data_test[:len(data_train)]])

        X_train, X_test, y_train, y_test = train_test_split(data_final[:, :-1], data_final[:, -1], test_size=0.5)
        tmp_all = []
        group_train = X_train[:, 0].reshape(-1, 1)
        X_train = X_train[:, 1:]
        group_test = X_test[:, 0].reshape(-1,1)
        X_test = X_test[:, 1:]
        for c in [0, 1]:
            inds = X_train[:, 0] == c
            X_c = X_train[inds, 1:3]
            s_c = X_train[inds, 3]
            model = attack_model(num_epoch=15000,
                                 learning_rate=1e-5,  # 5e-5 works for both
                                 batch_size=4000,
                                 verbose=0)
            y_c = y_train[inds]
            model.fit(X_c, y_c)
            # testing
            inds_test = X_test[:, 0] == c
            X_c_test = X_test[inds_test, 1:3]
            s_c_test = X_test[inds_test, 3]
            x_prob = model.predict_proba(X_c_test)
            y_c_test = y_test[inds_test]
            group_c_test = group_test[inds_test]
            All_data_c = np.hstack((group_c_test,
                                    y_c_test.reshape(-1, 1),
                                    x_prob,
                                    s_c_test.reshape(-1, 1),
                                    X_test[inds_test, 0].reshape(-1, 1)))
            tmp_all.append(All_data_c)
            # add attack result with T-train in A-train
            ind_TTAT = inds*(y_train == 1)
            X_TTAT = X_train[ind_TTAT, 1:3]
            s_TTAT = X_train[ind_TTAT, 3]
            x_prob = model.predict_proba(X_TTAT)
            y_TTAT = y_train[ind_TTAT]
            group_TTAT = group_train[ind_TTAT]
            All_data_c = np.hstack((group_TTAT,
                                    y_TTAT.reshape(-1, 1),
                                    x_prob,
                                    s_TTAT.reshape(-1, 1),
                                    X_train[ind_TTAT, 0].reshape(-1, 1)))
            tmp_all.append(All_data_c)
        All_Data = np.vstack(tmp_all)
        final_df = pd.DataFrame(All_Data, index=None)
        final_df.to_csv(folder + "/attack_result_1.csv", header=False, index=False)
