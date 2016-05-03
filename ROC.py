from __future__ import division
import sys
import csv
import numpy as np
import pandas as pd
from pandas import DataFrame
from classifier import *
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn import tree

def run_analysis(data_sets, labels):
	print "ROC::run_analysis()"
	#print_data(data_sets, labels)	
	
	pre_process = False
	
	if(pre_process):
		#pre-process data, incl. feature selection
		feature_names = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'RATIO_1', 'RATIO_2']
		data_sets = feature_selection(data_sets)
	else:
		feature_names = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

	#finish preprocessing
	Train_data_sets = data_sets.head(15000)
	Train_data_labels = labels.head(15000)
	Test_data_sets = data_sets.tail(15000)
	Test_data_labels = labels.tail(15000)

	print_count = False
	if (print_count):
		s = Test_data_labels["Y"]
		count_default = s.value_counts()
		print count_default

	Train_data_labels = np.ravel(Train_data_labels)
	Test_data_labels = np.ravel(Test_data_labels)

	#DT
	DT_classifier = build_DT_classifier(Train_data_sets, Train_data_labels)
	#print Train_data_sets.head(10)
	DT_predicted = predict_test_data(Test_data_sets, DT_classifier)
	DT_probas = DT_classifier.predict_proba(Test_data_sets)
	
	print_tree = False
	if(print_tree):
		#feature_names = list(data_sets.columns.values)	
		tree.export_graphviz(DT_classifier, class_names = ["No Default", "Yes Default"], feature_names = feature_names, max_depth = 2, out_file='tree.dot')

	#KNN
	KNN_classifier = build_KNN_classifier(Train_data_sets, Train_data_labels)
	KNN_predicted = predict_test_data(Test_data_sets, KNN_classifier)
	knn_probas = KNN_classifier.predict_proba(Test_data_sets)

	#LR
	LR_classifier = build_LR_classifier(Train_data_sets, Train_data_labels)
	LR_predicted = predict_test_data(Test_data_sets, LR_classifier)
	LR_probas = LR_classifier.predict_proba(Test_data_sets)

	#DA
	DA_classifier = build_DA_classifier(Train_data_sets, Train_data_labels)
	DA_predicted = predict_test_data(Test_data_sets, DA_classifier)
	DA_probas = DA_classifier.predict_proba(Test_data_sets)

	#NB
	NB_classifier = build_NB_classifier(Train_data_sets, Train_data_labels)
	NB_predicted = predict_test_data(Test_data_sets, NB_classifier)
	NB_probas = NB_classifier.predict_proba(Test_data_sets)


	print_error_rates = False
	if(print_error_rates):
		print_error_rate("KNN", KNN_predicted, Test_data_labels)
		print_error_rate("LR", LR_predicted, Test_data_labels)
		print_error_rate("DA", DA_predicted, Test_data_labels)
		print_error_rate("DT", DT_predicted, Test_data_labels)
		print_error_rate("NB", NB_predicted, Test_data_labels)

	#ROC analysis
	run_ROC_analysis = False
	if(run_ROC_analysis):
		build_roc_curve(Test_data_labels, knn_probas, LR_probas, DA_probas, DT_probas, NB_probas)


def feature_selection(data_sets):
	print "ROC::feature_selection()"
	data_sets["percent_max_Sept"] = (data_sets["X12"] / data_sets["X1"]) * 100
	data_sets["percent_max_Aug"] = (data_sets["X13"] / data_sets["X1"]) * 100
	
	pd.set_option('display.max_columns', None)
	#print data_sets.head(5)

	return data_sets
	#print data_sets.head(10)

def cardinality(labels):
	yes = 0
	no = 0

	for label in labels:
		if (label == 0):
			no += 1
		else:
			yes += 1

	print "total yes: " + str(yes)
	print "total no: " + str(no)
	print "n: " + str(yes + no)

	percent_default = yes / (yes+no)
	print "percentage defaults: " + str(percent_default)

def build_roc_curve(labels, knn_probas, LR_probas, DA_probas, DT_probas, NB_probas):

	knn_fpr, knn_tpr, knn_thresholds = roc_curve(labels, knn_probas[:, 1])
	knn_roc_auc = auc(knn_fpr, knn_tpr)
	knn_output=('KNN AUC = %0.4f'% knn_roc_auc)
	print knn_output

	LR_fpr, LR_tpr, LR_thresholds = roc_curve(labels, LR_probas[:, 1])
	LR_roc_auc = auc(LR_fpr, LR_tpr)
	LR_output=('LR AUC = %0.4f'% LR_roc_auc)
	print LR_output

	DA_fpr, DA_tpr, DA_thresholds = roc_curve(labels, DA_probas[:, 1])
	DA_roc_auc = auc(DA_fpr, DA_tpr)
	DA_output=('DA AUC = %0.4f'% DA_roc_auc)
	print DA_output

	DT_fpr, DT_tpr, DT_thresholds = roc_curve(labels, DT_probas[:, 1])
	DT_roc_auc = auc(DT_fpr, DT_tpr)
	DT_output=('DT AUC = %0.4f'% DT_roc_auc)
	print DT_output

	NB_fpr, NB_tpr, NB_thresholds = roc_curve(labels, NB_probas[:, 1])
	NB_roc_auc = auc(NB_fpr, NB_tpr)
	NB_output=('NB AUC = %0.4f'% NB_roc_auc)
	print NB_output
	
	plot_on = True
	if(plot_on):
		#setup plot
		plt.plot(NB_fpr, NB_tpr, label='Naive Bayesian')
		plt.plot(DA_fpr, DA_tpr, label='Discriminant Analysis')
		plt.plot(LR_fpr, LR_tpr, label='LogRegression')
		plt.plot(DT_fpr, DT_tpr, label='Classification tree')
		plt.plot(knn_fpr, knn_tpr, label='KNN')
		
		plt.axis([-.1, 1, 0, 1.1])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Receiver Operating Characteristic')
		plt.legend(loc="lower right")
		plt.show()

def calc_confusion_matrix(model, predicted, labels):
	cm = confusion_matrix(labels, predicted, labels = [0,1])

	print model + " confusion_matrix: "
	print cm
	print "---"

def print_data(data_sets, labels):
	pd.set_option('display.max_columns', None)
	data_sets["Y"] = labels
	print data_sets.tail(5)

def print_error_rate(model, predicted, labels):
	error_rate = error_measure(predicted, labels)
	print model + " error rate: ", error_rate
	