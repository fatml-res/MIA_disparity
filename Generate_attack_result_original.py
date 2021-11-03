import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from NN_DP import attack_model
from Tools import find_all_results, get_sub_group_ind
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from sklearn import metrics


def double_attack(X_train, X_test, y_train, y_test, folder=""):
    tmp_all = []
    for c in [0, 1]:
        print("Working on lable = {}".format(c))
        inds = X_train[:, 0] == c
        X_c = X_train[inds, 1:3]
        g_c = X_train[inds, 3]
        r_c = X_train[inds, 4]
        model = attack_model(num_epoch=15000,
                             learning_rate=1e-5,  # 5e-5 works for both
                             batch_size=4000,
                             verbose=0)
        y_c = y_train[inds]
        model.fit(X_c, y_c)
        # testing
        inds_test = X_test[:, 0] == c
        X_c_test = X_test[inds_test, 1:3]
        g_c_test = X_test[inds_test, 3]
        r_c_test = X_test[inds_test, 4]
        x_prob = model.predict_proba(X_c_test)
        y_c_test = y_test[inds_test]
        All_data_c = np.hstack((y_c_test.reshape(-1, 1),
                                x_prob,
                                g_c_test.reshape(-1, 1),
                                r_c_test.reshape(-1, 1),
                                X_test[inds_test, 0].reshape(-1, 1)))
        tmp_all.append(All_data_c)
    All_Data = np.vstack((tmp_all[0], tmp_all[1]))
    final_df = pd.DataFrame(All_Data, index=None)
    final_df.to_csv(folder + "/attack_result.csv", header=False, index=False)


def sp_attack(X_train, X_test, y_train, y_test):
    tmp_all = []
    for c in [0, 1]:
        print("Working on lable = {}".format(c))
        inds = X_train[:, 0] == c
        X_c = X_train[inds, 1:3]
        g_c = X_train[inds, 3]
        r_c = X_train[inds, 4]
        model = attack_model(num_epoch=15000,
                             learning_rate=1e-5,  # 5e-5 works for both
                             batch_size=4000,
                             verbose=0)
        y_c = y_train[inds]
        model.fit(X_c, y_c)
        # testing
        inds_test = X_test[:, 0] == c
        X_c_test = X_test[inds_test, 1:3]
        g_c_test = X_test[inds_test, 3]
        r_c_test = X_test[inds_test, 4]
        x_prob = model.predict_proba(X_c_test)
        y_c_test = y_test[inds_test]
        All_data_c = np.hstack((y_c_test.reshape(-1, 1),
                                x_prob,
                                g_c_test.reshape(-1, 1),
                                r_c_test.reshape(-1, 1),
                                X_c_test[:, 0].reshape(-1,1),
                                X_test[inds_test, 0].reshape(-1, 1)))
        tmp_all.append(All_data_c)
    All_Data = np.vstack((tmp_all[0], tmp_all[1]))
    final_df = pd.DataFrame(All_Data, index=None)
    final_df.to_csv(folder + "/attack_result_sp.csv", header=False, index=False)
    arr = ori_distribution_ratio(folder + "/attack_result_sp.csv", "Hospital")


def ori_distribution_ratio(mia_file, dataset):
    data = pd.read_csv(mia_file, header=None).to_numpy()
    mem_true = data[:, 0]
    pred = data[:, 1].round()
    csv = data[:, 4].round(1)
    arr = []
    for s_ind in [0, 1]:
        inds = get_sub_group_ind(data, s_ind+2, -1)
        for i in range(4):
            s_name = ["Gender", "Race"][s_ind]
            ind = inds[i]
            sg_name = ["pT", "upT", "pF", "upF"][i]
            acc_list = [0]*10
            pred_sg = pred[ind]
            mem_sg = mem_true[ind]
            csv_sg = csv[ind]
            for n in range(9):
                ind_bin = (csv_sg>=n*0.1) * (csv_sg<(n+1)*0.1)
                acc_list[n] = metrics.accuracy_score(pred_sg[ind_bin], mem_sg[ind_bin])
            final_bin_ind = csv_sg>=0.9
            acc_list[-1] = metrics.accuracy_score(pred_sg[final_bin_ind], mem_sg[final_bin_ind])
            arr.append([dataset, s_name, sg_name] + acc_list)
    arr = np.array(arr)
    arr[arr == "nan"] = 0.5
    np.save("Original/{}_fill.npy".format(dataset), arr)
    return arr




if __name__ == "__main__":
    tf.config.threading.set_intra_op_parallelism_threads(10)
    tf.config.threading.set_inter_op_parallelism_threads(10)
    files = find_all_results("gradients/ori")
    replace = 0
    special = False # Single attack is not working well with acceptable MIA accuracy
    if special:
        attack_type = "attack_result_sp.csv"
    else:
        attack_type = "attack_result.csv"
    for file in files:
        folder = file.split("/target_result.csv")[0]
        if (os.path.exists(folder + "/" + attack_type) and replace == 0):
            print("Skipped " + file)
            continue
        else:
            print("Start working on " + file)
        df = pd.read_csv(file, header=None)
        data = np.array(df)
        X_train, X_test, y_train, y_test = train_test_split(data[:, :-1], data[:, -1], test_size=0.5)
        if special:
            sp_attack(X_train, X_test, y_train, y_test)
        else:
            double_attack(X_train, X_test, y_train, y_test)
