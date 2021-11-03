import os

# import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import pandas as pd


def score_calculate(predict_y, y):
    if len(y) > 1 and sum(predict_y == 1) > 0:
        prec = metrics.precision_score(y, predict_y)
        recall = metrics.recall_score(y, predict_y)
        f1 = metrics.f1_score(y, predict_y)
    elif sum(abs(predict_y - y)) == 0:
        prec = 1
        recall = 1
        f1 = 1
    else:
        prec = 0
        recall = 0
        f1 = 0
    return prec, recall, f1


def pld_calculate(predict_y, y, gender, race):
    # 1 is protected
    TP_race1 = sum((race == 1) * (y == 1) * (predict_y == 1))
    TN_race1 = sum((race == 1) * (y == 0) * (predict_y == 0))
    FP_race1 = sum((race == 1) * (y == 0) * (predict_y == 1))
    FN_race1 = sum((race == 1) * (y == 1) * (predict_y == 0))

    TP_race0 = sum((race == 0) * (y == 1) * (predict_y == 1))
    TN_race0 = sum((race == 0) * (y == 0) * (predict_y == 0))
    FP_race0 = sum((race == 0) * (y == 0) * (predict_y == 1))
    FN_race0 = sum((race == 0) * (y == 1) * (predict_y == 0))
    if TP_race0 == 0 or TP_race1 == 0 or TN_race0 == 0 or TN_race1 == 0:
        print("Pause")

    pld_race_acc = (TP_race1 + TN_race1) / (TP_race1 + FP_race1 + FN_race1 + TN_race1) - \
                   (TP_race0 + TN_race0) / (TP_race0 + FP_race0 + FN_race0 + TN_race0)

    pld_race_prec = TP_race1 / (TP_race1 + FP_race1) - TP_race0 / (TP_race0 + FP_race0)

    pld_race_recall = TP_race1 / (TP_race1 + FN_race1) - TP_race0 / (TP_race0 + FN_race0)

    TP_gender1 = sum((gender == 1) * (y == 1) * (predict_y == 1))
    TN_gender1 = sum((gender == 1) * (y == 0) * (predict_y == 0))
    FP_gender1 = sum((gender == 1) * (y == 0) * (predict_y == 1))
    FN_gender1 = sum((gender == 1) * (y == 1) * (predict_y == 0))

    TP_gender0 = sum((gender == 0) * (y == 1) * (predict_y == 1))
    TN_gender0 = sum((gender == 0) * (y == 0) * (predict_y == 0))
    FP_gender0 = sum((gender == 0) * (y == 0) * (predict_y == 1))
    FN_gender0 = sum((gender == 0) * (y == 1) * (predict_y == 0))
    if TP_gender0 == 0 or TP_gender1 == 0 or TN_gender0 == 0 or TN_gender1 == 0:
        print("Pause")

    pld_gender_acc = (TP_gender1 + TN_gender1) / (TP_gender1 + FP_gender1 + FN_gender1 + TN_gender1) - \
                     (TP_gender0 + TN_gender0) / (TP_gender0 + FP_gender0 + FN_gender0 + TN_gender0)

    pld_gender_prec = TP_gender1 / (TP_gender1 + FP_gender1) - TP_gender0 / (TP_gender0 + FP_gender0)

    pld_gender_recall = TP_gender1 / (TP_gender1 + FN_gender1) - TP_gender0 / (TP_gender0 + FN_gender0)

    return pld_gender_acc, pld_gender_prec, pld_gender_recall, \
           pld_race_acc, pld_race_prec, pld_race_recall


def attr_map(s, major_value):
    result = np.zeros([len(s), 1])
    result[s != major_value] = 1
    return result


def binary_transform_s(data, sen_ind):
    X, y = data[:, 0:-1], data[:, -1]
    s = X[:, sen_ind].reshape(-1, 1)
    s_value, s_count = np.unique(s, return_counts=True)

    s_maj_ind = s_count.argmax()
    s_maj_value = s_value[s_maj_ind]
    s = attr_map(s, s_maj_value)
    X[:, sen_ind] = list(s)
    return X, y

def get_min_max(data):
    min = data.min(axis=0)[2:-1]
    max = data.max(axis=0)[2:-1]
    result = np.stack([min, max])
    return tuple(map(tuple, result))


def find_all_results(root_path, result_type="target_result.csv"):
    '''output: all the result file name and their paths'''
    file_list = []
    for path, subdirs, files in os.walk(root_path):
        for name in files:
            if result_type in name:
                file_list.append(os.path.join(path, name).replace("\\", "/"))
    return file_list


def get_distribution(probs, bin_size=0.05, range_mode=0, min_r=0, max_r=1):
    probs = np.array(probs)
    bin_number = int(1 / bin_size)
    count_all = len(probs)
    pdf = []
    if range_mode == 0:
        # 0, 1
        min_r = 0
        max_r = 1
    elif range_mode == 1:
        # adaptive
        min_r = min(probs)
        max_r = max(probs)
    bin_unit = (max_r - min_r) * bin_size
    for b_ind in range(bin_number - 1):
        min_p = min_r + b_ind*bin_unit
        max_p = min_r + (b_ind+1) * bin_unit
        counti = ((probs >= min_p) * (probs <= max_p)).sum()
        pdfi = (counti + 1) / (count_all + bin_number)
        pdf.append(pdfi)
    countr = ((probs >= max_r - bin_unit)).sum()
    pdfr = (countr + 1) / (count_all + bin_number)
    pdf.append(pdfr)
    return pdf


def text_to_npy(file):
    text_file = open(file, "r")
    lines = text_file.read().split(',')
    lines[0] = lines[0].replace('[', '')
    lines[-1] = lines[-1].replace(']', '')
    ar = np.array(lines).astype('float')
    np.save(file.replace('txt', 'npy'), ar)
    print("pause")


def KLD(p, q):
    p = np.array(p)
    q = np.array(q)
    p = p / p.sum()
    q = q / q.sum()
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def get_accuracy_from_file(file, prob_ind, file_type='target', s_ind=0, label_ind=0):
    df = pd.read_csv(file, header=None)
    result = np.array(df)
    if file_type == 'target':
        # [group, True, prob0, prob1, s, train_test_flag]
        train_ind = result[:, -1] == 1
        test_ind = result[:, -1] == 0
        s_attr = result[:, s_ind]
        ind_up = (s_attr == 0)
        ind_p = (s_attr == 1)

        ## global
        train_pred = result[train_ind, prob_ind + 1].round()
        train_acc = metrics.accuracy_score(result[train_ind, label_ind], train_pred)

        test_pred = result[test_ind, prob_ind + 1].round()
        test_acc = metrics.accuracy_score(result[test_ind, label_ind], test_pred)
        ## up
        train_pred_up = result[ind_up*train_ind, prob_ind + 1].round()
        train_label_up = result[train_ind*ind_up, label_ind]
        train_acc_up = metrics.accuracy_score(train_label_up, train_pred_up)

        test_pred_up = result[test_ind*ind_up, prob_ind + 1].round()
        test_label_up = result[test_ind*ind_up, label_ind]
        test_acc_up = metrics.accuracy_score(test_label_up, test_pred_up)

        ## p
        train_pred_p = result[train_ind * ind_p, prob_ind + 1].round()
        train_label_p = result[train_ind * ind_p, label_ind]
        train_acc_p = metrics.accuracy_score(train_label_p, train_pred_p)

        test_pred_p = result[test_ind * ind_p, prob_ind + 1].round()
        test_label_p = result[test_ind * ind_p, label_ind]
        test_acc_p = metrics.accuracy_score(test_label_p, test_pred_p)

        output = [train_acc.round(3), test_acc.round(3),
                  train_acc_up.round(3), test_acc_up.round(3),
                  train_acc_p.round(3), test_acc_p.round(3)]

    elif file_type == 'MIA':
        MIA_flag=1
        # [group, train_test_flag, mia_prob, s, target_class]
        df = pd.read_csv(file, header=None)
        result = np.array(df)
        labels = result[:, label_ind]
        pred = result[:, prob_ind].round()
        s_attr = result[:, s_ind]
        ind_up = (s_attr == 0)
        ind_p = (s_attr == 1)

        Member_ind = labels == 1
        non_Member_ind = labels == 0
        Member_up_ind = Member_ind*ind_up
        Member_p_ind = Member_ind * ind_p
        non_Member_up_ind = non_Member_ind * ind_up
        non_Member_p_ind = non_Member_ind * ind_p

        mia_acc_label = metrics.accuracy_score(labels[Member_ind], pred[Member_ind])
        acc_up_label = metrics.accuracy_score(labels[Member_up_ind], pred[Member_up_ind])
        acc_p_label = metrics.accuracy_score(labels[Member_p_ind], pred[Member_p_ind])

        mia_acc_test = metrics.accuracy_score(labels[non_Member_ind], pred[non_Member_ind])
        acc_up_test = metrics.accuracy_score(labels[non_Member_up_ind], pred[non_Member_up_ind])
        acc_p_test = metrics.accuracy_score(labels[non_Member_p_ind], pred[non_Member_p_ind])

        mia_acc_1 = (mia_acc_label*Member_ind.sum()/2+mia_acc_test*non_Member_ind.sum())/(non_Member_ind.sum()+Member_ind.sum()/2)
        acc_up_1 = (acc_up_label * Member_up_ind.sum() / 2 + acc_up_test * non_Member_up_ind.sum()) / (
                    non_Member_up_ind.sum() + Member_up_ind.sum() / 2)
        acc_p_1 = (acc_p_label * Member_p_ind.sum() / 2 + acc_p_test * non_Member_p_ind.sum()) / (
                    non_Member_p_ind.sum() + Member_p_ind.sum() / 2)

        acc_up = 0
        acc_p = 0
        mia_acc = 0
        for t_label in [0, 1]:
            sub_t_ind = (result[:, -1]==t_label)*Member_ind
            labels_t = labels[sub_t_ind]
            pred_t = pred[sub_t_ind]
            mia_acc += metrics.accuracy_score(labels_t, pred_t)/2

            acc_up += metrics.accuracy_score(labels[ind_up*sub_t_ind], pred[ind_up*sub_t_ind])/2
            acc_p += metrics.accuracy_score(labels[ind_p*sub_t_ind], pred[ind_p*sub_t_ind])/2
        if MIA_flag:
            output = [round(mia_acc_1, 3), round(acc_up_1, 3), round(acc_p_1, 3),
                      round(mia_acc_label, 3), round(acc_up_label, 3), round(acc_p_label, 3)]
        else:
            output = [round(mia_acc, 3), round(acc_up, 3), round(acc_p, 3)]
    else:
        print("Unknown result type")
        output = 0

    return output


def match_majority(X, sen_ind):
    s = X[:, sen_ind].reshape(-1, 1)
    s_value, s_count = np.unique(s, return_counts=True)

    s_maj_ind = s_count.argmax()
    s_maj_value = s_value[s_maj_ind]
    s = attr_map(s, s_maj_value)
    X[:, sen_ind] = list(s)
    return X, s


def get_three_metrics_from_file(file, prob_ind, file_type='MIA', s_ind=0, label_ind=0):
    df = pd.read_csv(file, header=None)
    result = np.array(df)
    if file_type == 'target':
        # We don't need this function for target for now.
        pass
    elif file_type == 'MIA':
        labels = result[:, label_ind]
        pred = result[:, prob_ind].round()
        s_attr = result[:, s_ind]
        ind_up = (s_attr == 0)
        ind_p = (s_attr == 1)
        if ind_up.sum() < ind_p.sum()*0.9:
            print("Error! Unprotected group is smaller than Protected group!")
            s_attr = 1-s_attr
            ind_up = (s_attr == 0)
            ind_p = (s_attr == 1)

        Member_ind = labels == 1
        non_Member_ind = labels == 0
        Member_up_ind = Member_ind * ind_up
        Member_p_ind = Member_ind * ind_p
        non_Member_up_ind = non_Member_ind * ind_up
        non_Member_p_ind = non_Member_ind * ind_p
        # Global
        MIA_ACC = metrics.accuracy_score(pred, labels)
        MIA_Prec = metrics.precision_score(labels, pred)
        MIA_recall = metrics.recall_score(labels, pred)
        # PLD

        up_ACC = metrics.accuracy_score(pred[ind_up], labels[ind_up])
        p_ACC = metrics.accuracy_score(pred[ind_p], labels[ind_p])
        PLD_ACC = p_ACC - up_ACC

        up_Prec = metrics.precision_score(labels[ind_up], pred[ind_up])
        p_Prec = metrics.precision_score(labels[ind_p], pred[ind_p])
        PLD_Prec = p_Prec - up_Prec

        up_recall = metrics.recall_score(labels[ind_up], pred[ind_up])
        p_recall = metrics.recall_score(labels[ind_p], pred[ind_p])
        PLD_recall = p_recall - up_recall

        # Analyze Only
        ind_T = result[:, -1] == 0
        ind_F = result[:, -1] == 1
        up_T_acc = metrics.accuracy_score(labels[ind_up * ind_T], pred[ind_up * ind_T])
        p_T_acc = metrics.accuracy_score(labels[ind_p * ind_T], pred[ind_p * ind_T])
        PLD_ACC_T = p_T_acc - up_T_acc

        up_F_acc = metrics.accuracy_score(labels[ind_up * ind_F], pred[ind_up * ind_F])
        p_F_acc = metrics.accuracy_score(labels[ind_p * ind_F], pred[ind_p * ind_F])
        PLD_ACC_F = p_F_acc - up_F_acc

        #PLD_ACC = (PLD_ACC_T+PLD_ACC_F)/2

        up_T_prec = metrics.precision_score(labels[ind_up * ind_T], pred[ind_up * ind_T])
        p_T_prec = metrics.precision_score(labels[ind_p * ind_T], pred[ind_p * ind_T])
        PLD_prec_T = p_T_prec - up_T_prec

        up_F_prec = metrics.precision_score(labels[ind_up * ind_F], pred[ind_up * ind_F])
        p_F_prec = metrics.precision_score(labels[ind_p * ind_F], pred[ind_p * ind_F])
        PLD_prec_F = p_F_prec - up_F_prec

        #PLD_Prec = (PLD_prec_T+PLD_prec_F)/2

        up_T_recall = metrics.recall_score(labels[ind_up * ind_T], pred[ind_up * ind_T])
        p_T_recall = metrics.recall_score(labels[ind_p * ind_T], pred[ind_p * ind_T])
        PLD_recall_T = p_T_recall - up_T_recall

        up_F_recall = metrics.recall_score(labels[ind_up * ind_F], pred[ind_up * ind_F])
        p_F_recall = metrics.recall_score(labels[ind_p * ind_F], pred[ind_p * ind_F])
        PLD_recall_F = p_F_recall - up_F_recall

        #PLD_recall = (PLD_recall_T + PLD_recall_F)/2

        r_T = sum(ind_p*ind_T)/sum(ind_up*ind_T)
        r_F = sum(ind_p * ind_F) / sum(ind_up * ind_F)
        if sum(ind_up * ind_F)==0 or sum(ind_p * ind_F)==0 or sum(ind_up * ind_T)==0 or sum(ind_p * ind_T)==0:
            print("value error pause")
            return [], []

        analysis_arr = np.array([[sum(ind_p * ind_T), sum(ind_up * ind_T), sum(ind_p * ind_F), sum(ind_up * ind_F)],
                                 [p_T_acc, up_T_acc, p_F_acc, up_F_acc],
                                 [p_T_prec, up_T_prec, p_F_prec, up_F_prec],
                                 [p_T_recall, up_T_recall, p_F_recall, up_F_recall]])



        return [MIA_ACC, MIA_Prec, MIA_recall, PLD_ACC, PLD_Prec, PLD_recall],analysis_arr


def get_sub_group_ind(data, s_ind, y_ind=-1):
    s = data[:, s_ind]
    y = data[:, y_ind]
    up_ind = s == 1
    p_ind = s == 0
    if up_ind.sum() < p_ind.sum():
        s = 1 - s
        up_ind = s == 1
        p_ind = s == 0

    T_ind = y == 0
    F_ind = y == 1

    return p_ind*T_ind, p_ind*F_ind, up_ind*T_ind, up_ind*F_ind



if __name__ == "__main__":
    text_to_npy("Original/influence_Hospital.txt")
