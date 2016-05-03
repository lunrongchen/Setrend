import numpy as np
from sklearn.decomposition import PCA
from classifier import *
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, roc_curve, auc


def get_pca(data_sets, n_component):
    pca = PCA(n_components=n_component)
    pca.fit(X)
    return pca.explained_variance_ratio_

def KNN_PCA_cross_validation(data_sets, folds, pca_n):
    kf = KFold(30000, n_folds=folds)
    final_error = 0.0
    final_ROC = 0.0
    index = 1
    for train_indices, test_indices in kf:
        X_train = []
        X_test = []
        for i in train_indices:
            X_train.append(data_sets[i])
        for i in test_indices:
            X_test.append(data_sets[i])
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        for row in X_train:
            train_x.append(row[0:22])
            train_y.append(row[23])
        for row in X_test:
            test_x.append(row[0:22])
            test_y.append(row[23])
        train_x = PCA(n_components=pca_n).fit_transform(train_x)
        test_x = PCA(n_components=pca_n).fit_transform(test_x)
        KNN_classifer = build_KNN_classifier(train_x, train_y)
        KNN_predicted = predict_test_data(test_x, KNN_classifer)
        KNN_error_rate = error_measure(KNN_predicted, test_y)

        knn_probas = KNN_classifer.predict_proba(test_x)
        knn_fpr, knn_tpr, knn_thresholds = roc_curve(test_y, knn_probas[:, 1])
        knn_roc_auc = auc(knn_fpr, knn_tpr)
        knn_output = ('KNN AUC = %0.2f'% knn_roc_auc)
        final_ROC = float(knn_roc_auc) + final_ROC
        # print knn_output

        final_error += KNN_error_rate
        index = index + 1
    print "KNN final_error: ", final_error / float(folds)
    print "KNN final_ROC: ", final_ROC / float(folds)



def LR_PCA_cross_validation(data_sets, folds, pca_n):
    kf = KFold(30000, n_folds=folds)
    final_error = 0.0
    final_ROC = 0.0
    index = 1
    for train_indices, test_indices in kf:
        X_train = []
        X_test = []
        for i in train_indices:
            X_train.append(data_sets[i])
        for i in test_indices:
            X_test.append(data_sets[i])
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        for row in X_train:
            train_x.append(row[0:22])
            train_y.append(row[23])
        for row in X_test:
            test_x.append(row[0:22])
            test_y.append(row[23])
        train_x = PCA(n_components=pca_n).fit_transform(train_x)
        test_x = PCA(n_components=pca_n).fit_transform(test_x)
        LR_classifer = build_LR_classifier(train_x, train_y)
        LR_predicted = predict_test_data(test_x, LR_classifer)
        LR_error_rate = error_measure(LR_predicted, test_y)

        LR_probas = LR_classifer.predict_proba(test_x)
        LR_fpr, LR_tpr, LR_thresholds = roc_curve(test_y, LR_probas[:, 1])
        LR_roc_auc = auc(LR_fpr, LR_tpr)
        LR_output = ('KNN AUC = %0.2f'% LR_roc_auc)
        final_ROC = float(LR_roc_auc) + final_ROC
        # print knn_output
        
        final_error += LR_error_rate
        index = index + 1
    print "LR final_error: ", final_error / float(folds)
    print "LR final_ROC: ", final_ROC / float(folds)


def DA_PCA_cross_validation(data_sets, folds, pca_n):
    kf = KFold(30000, n_folds=folds)
    final_error = 0.0
    final_ROC = 0.0
    index = 1
    for train_indices, test_indices in kf:
        X_train = []
        X_test = []
        for i in train_indices:
            X_train.append(data_sets[i])
        for i in test_indices:
            X_test.append(data_sets[i])
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        for row in X_train:
            train_x.append(row[0:22])
            train_y.append(row[23])
        for row in X_test:
            test_x.append(row[0:22])
            test_y.append(row[23])
        train_x = PCA(n_components=pca_n).fit_transform(train_x)
        test_x = PCA(n_components=pca_n).fit_transform(test_x)
        DA_classifer = build_DA_classifier(train_x, train_y)
        DA_predicted = predict_test_data(test_x, DA_classifer)
        DA_error_rate = error_measure(DA_predicted, test_y)

        DA_probas = DA_classifer.predict_proba(test_x)
        DA_fpr, DA_tpr, DA_thresholds = roc_curve(test_y, DA_probas[:, 1])
        DA_roc_auc = auc(DA_fpr, DA_tpr)
        DA_output = ('KNN AUC = %0.2f'% DA_roc_auc)
        final_ROC = float(DA_roc_auc) + final_ROC
        # print knn_output
        
        final_error += DA_error_rate
        index = index + 1
    print "DA final_error: ", final_error / float(folds)
    print "DA final_ROC: ", final_ROC / float(folds)


def DT_PCA_cross_validation(data_sets, folds, pca_n):
    kf = KFold(30000, n_folds=folds)
    final_error = 0.0
    final_ROC = 0.0
    index = 1
    for train_indices, test_indices in kf:
        X_train = []
        X_test = []
        for i in train_indices:
            X_train.append(data_sets[i])
        for i in test_indices:
            X_test.append(data_sets[i])
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        for row in X_train:
            train_x.append(row[0:22])
            train_y.append(row[23])
        for row in X_test:
            test_x.append(row[0:22])
            test_y.append(row[23])
        train_x = PCA(n_components=pca_n).fit_transform(train_x)
        test_x = PCA(n_components=pca_n).fit_transform(test_x)
        DT_classifer = build_DT_classifier(train_x, train_y)
        DT_predicted = predict_test_data(test_x, DT_classifer)
        DT_error_rate = error_measure(DT_predicted, test_y)

        DT_probas = DT_classifer.predict_proba(test_x)
        DT_fpr, DT_tpr, DT_thresholds = roc_curve(test_y, DT_probas[:, 1])
        DT_roc_auc = auc(DT_fpr, DT_tpr)
        DT_output = ('KNN AUC = %0.2f'% DT_roc_auc)
        final_ROC = float(DT_roc_auc) + final_ROC
        # print knn_output
        
        final_error += DT_error_rate
        index = index + 1
    print "DT final_error: ", final_error / float(folds)
    print "DT final_ROC: ", final_ROC / float(folds)


def NB_PCA_cross_validation(data_sets, folds, pca_n):
    kf = KFold(30000, n_folds=folds)
    final_error = 0.0
    final_ROC = 0.0
    index = 1
    for train_indices, test_indices in kf:
        X_train = []
        X_test = []
        for i in train_indices:
            X_train.append(data_sets[i])
        for i in test_indices:
            X_test.append(data_sets[i])
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        for row in X_train:
            train_x.append(row[0:22])
            train_y.append(row[23])
        for row in X_test:
            test_x.append(row[0:22])
            test_y.append(row[23])
        train_x = PCA(n_components=pca_n).fit_transform(train_x)
        test_x = PCA(n_components=pca_n).fit_transform(test_x)
        NB_classifer = build_NB_classifier(train_x, train_y)
        NB_predicted = predict_test_data(test_x, NB_classifer)
        NB_error_rate = error_measure(NB_predicted, test_y)

        NB_probas = NB_classifer.predict_proba(test_x)
        NB_fpr, NB_tpr, NB_thresholds = roc_curve(test_y, NB_probas[:, 1])
        NB_roc_auc = auc(NB_fpr, NB_tpr)
        NB_output = ('KNN AUC = %0.2f'% NB_roc_auc)
        final_ROC = float(NB_roc_auc) + final_ROC
        # print knn_output
        
        final_error += NB_error_rate
        index = index + 1
    print "NB final_error: ", final_error / float(folds)
    print "NB final_ROC: ", final_ROC / float(folds)


def SVM_PCA_cross_validation(data_sets, folds, pca_n):
    kf = KFold(30000, n_folds=folds)
    final_error = 0.0
    index = 1
    for train_indices, test_indices in kf:
        X_train = []
        X_test = []
        for i in train_indices:
            X_train.append(data_sets[i])
        for i in test_indices:
            X_test.append(data_sets[i])
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        for row in X_train:
            train_x.append(row[0:22])
            train_y.append(row[23])
        for row in X_test:
            test_x.append(row[0:22])
            test_y.append(row[23])
        train_x = PCA(n_components=pca_n).fit_transform(train_x)
        test_x = PCA(n_components=pca_n).fit_transform(test_x)
        SVM_classifer = build_SVM_classifier(train_x, train_y)
        SVM_predicted = predict_test_data(test_x, SVM_classifer)
        SVM_error_rate = error_measure(SVM_predicted, test_y)
        print index, " fold SVM_error_rate: ",  SVM_error_rate
        final_error += SVM_error_rate
        index = index + 1
    print "final_error: ", final_error / float(folds)
