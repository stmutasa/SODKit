"""
SOD Tester contains function wrappers for various metric and performance analysis functions including

mean absolute error, mean squared error, DICE score, sensitivity, specificity, AUC

"""

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as skm
from scipy import interp


class SODTester():

    """
    SOD Tester class is a class for testing the performance of our network
    """

    # Linear regression variables
    MAE = 0
    best_MAE = 10
    accuracy = 0

    # Classification variables
    TP, FP, TN, FN = 0, 0, 0, 0
    sensitiviy, specificity = 0, 0
    PPV, NPV = 0, 0
    F1_score, AUC = 0, 0
    roc_auc, fpr, tpr = {}, {}, {}

    # Other variables
    right, total, calls = 0, 0, 0
    best_step, num_classes = 0, 0

    def __init__(self, binary, regression):

        # Define whether this is a binary, multiclass or linear regression model
        self.binary = binary
        self.regression = regression

    """
     Performance Metrics
    """

    def mean_absolute_error(self, prediction, labels, display=True):

        """
        Calculates the MAE between predictions and labels
        :param prediction: network output
        :param labels: the ground truth
        :param display: Whether to print the MAE
        :return: MAE the mae
        """

        # Convert to numpy arrays
        predictions = np.squeeze(prediction.astype(np.float32))
        label = np.squeeze(labels.astype(np.float32))

        # How many to print
        to_print = min(len(label), 15)

        # Calculate MAE
        MAE = np.mean(np.absolute((predictions - label)))

        # Print the summary
        np.set_printoptions(precision=1)  # use numpy to print only the first sig fig
        if display: print('MAE: %s\n Pred: %s\nReal: %s' % (MAE, predictions[to_print], label[to_print]))

        # Append right
        self.right += MAE
        self.total += 1

        return MAE


    def get_accuracy_regression(self, Epoch, display=True):

        """
        Retreive the accuracy so far
        :param Epoch: what epoch we're in
        :param display: Whether to print
        :return: accuracy
        """

        # Calculate final MAE and ACC
        self.accuracy = self.right / self.total

        # Print the final accuracies and MAE if requested
        if display:
            print('-' * 70)
            print('--- EPOCH: %s MAE: %.4f (Old Best: %.1f @ %s) ---'
                    % (Epoch, self.accuracy, self.best_MAE, self.best_step))

        return self.accuracy


    def calculate_metrics(self, logits, labels, positive_class, step, display=True):

        """
        Retreive the Accuracy, Sensitivity, Specificity
        :param logits: network predictions
        :param labels: the ground truth
        :param positive_class: the class we will consider positive for calculating ground truth
        :param step: what step we're on
        :param display: whether to print examples
        :return: acc, sn, sp
        """

        # Retreive and print the labels and logits
        label = np.squeeze(labels.astype(np.int8))
        logit = np.squeeze(np.argmax(logits.astype(np.float), axis=1))

        # First calculate AUC
        self.AUC += skm.roc_auc_score(label, logit)
        self.calls += 1

        # Retreive metrics
        for z in range(len(label)):

            # If we got this right, make it right
            if label[z] == logit[z]: self.right += 1

            # Metrics for when the ground truth is positive
            if label[z] == positive_class:

                # Get metrics
                if label[z] == logit[z]: self.TP += 1
                if label[z] != logit[z]: self.FN += 1

            # Metrics for when the ground truth is negative
            if label[z] != positive_class:

                # Get metrics
                if label[z] == logit[z]: self.TN += 1
                if label[z] != logit[z]: self.FP += 1

        # Increment total
        self.total += len(label)

        # Print Summary if wanted
        if display:

            # How many to print
            to_print = min(len(label), 15)

            # Now print
            print('-' * 70)
            print('Patient %s Class: %s' % (step, label[:to_print]))
            print('Patient %s Preds: %s' % (step, logit[:to_print]))


    def calculate_multiclass_metrics(self, logitz, labelz, step, n_classes, display=True):
        """
        Calculates the metrics for a multiclass problem
        :param logits: raw logits = will be softmaxed
        :param labels: raw labels = will be made one hot
        :param step: the current step
        :param n_classes: number of total classes
        :param display: whether to print a summary
        :return:
        """

        # Retreive the one hot labels and softmax logits
        logits = self.calc_softmax(logitz)
        labels = np.eye(int(n_classes))[labelz.astype(np.int16)]

        # Set classes count since this must be non binary
        self.num_classes = n_classes

        # Get the metrics for each class
        for i in range (n_classes):

            # First retreive the fpr and tpr
            self.fpr[i], self.tpr[i], _ = skm.roc_curve(labels[:, i], logits[:, i])

            # Now get the AUC
            self.roc_auc[i] = skm.auc(self.fpr[i], self.tpr[i])

        # Now get the metrics for the micro ROC
        self.fpr['micro'], self.tpr['micro'], _ = skm.roc_curve(labels.ravel(), logits.ravel())
        self.roc_auc['micro'] = skm.auc(self.fpr['micro'], self.tpr['micro'])

        # Now Macro ROC: First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([self.fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at these points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes): mean_tpr += interp(all_fpr, self.fpr[i], self.tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        self.fpr["macro"] = all_fpr
        self.tpr["macro"] = mean_tpr
        self.roc_auc["macro"] = skm.auc(self.fpr["macro"], self.tpr["macro"])

        # Increment global SOD
        self.AUC += self.roc_auc["macro"]
        self.calls += 1

        # Now for the other metrics
        label = np.squeeze(labelz.astype(np.int8))
        logit = np.squeeze(np.argmax(logitz.astype(np.float), axis=1))

        # Retreive metrics
        for z in range(len(label)):

            # If we got this right, make it right
            if label[z] == logit[z]: self.right += 1

        # Increment total
        self.total += len(label)

        # Print Summary if wanted
        if display:

            # How many to print
            to_print = min(len(label), 15)

            # Now print
            print('-' * 70)
            print('Patient %s Class: %s' % (step, label[:to_print]))
            print('Patient %s Preds: %s' % (step, logit[:to_print]))

            # Display one per class
            for z in range (n_classes): print ('Class %s: %.3f --- '%(z, self.roc_auc[z]), end='')
            print ('Micro AUC: %.3f, Macro AUC: %.3f' %(self.roc_auc['micro'], self.roc_auc["macro"]))


    def retreive_metrics_classification(self, Epoch, display=True):

        """
        Retreives sn, sp, PPV, NPV, ROC
        :param Epoch: What epoch we're in
        :param display: Whether to print the results
        :return: 
        """

        # Calculate the metrics. To prevent division by zero, use error handling

        # Some metrics only make sense for binary
        if self.binary:
            try: self.sensitiviy = self.TP / (self.TP + self.FN)
            except: self.sensitiviy = 0

            try: self.specificity = self.TN / (self.TN + self.FP)
            except: self.specificity = 0

            try: self.PPV = self.TP / (self.TP + self.FP)
            except: self.PPV = 0

            try: self.NPV = self.TN / (self.TN + self.FN)
            except: self.PPV = 0

            # F1 score
            try: self.F1_score = 2/((1/self.sensitiviy)+(1/self.PPV))
            except: self.F1_score = 0

        # AUC
        self.AUC /= self.calls

        # Accuracy
        self.accuracy = 100 * self.right/self.total

        # Print the final accuracies and MAE if requested
        if display:

            print('-' * 70)

            # Print depends on binary or not
            if self.binary:

                print('--- EPOCH: %s, ACC: %.2f, SN: %.3f, SP: %.3f, AUC: %.3f, F1: %.3f ---'
                      % (Epoch, self.accuracy, self.sensitiviy, self.specificity, self.AUC, self.F1_score))
                print('--- True Pos: %s, False Pos: %s, True Neg: %s, False Neg: %s ---'
                      % (self.TP, self.FP, self.TN, self.FN))

            else:

                # Display one per class
                print('--- EPOCH: %s, ACC: %.2f %%, AUC: %.3f, ---' % (Epoch, self.accuracy, self.AUC))
                for z in range(self.num_classes): print('Class %s: %.3f --- ' % (z, self.roc_auc[z]), end='')
                print('Micro AUC: %.3f, Macro AUC: %.3f' % (self.roc_auc['micro'], self.roc_auc["macro"]))


    def calc_softmax(self, X):
        """
        Computes the softmax of a given vector
        :param X: numpy array
        :return:
        """

        # Copy array
        softmax = np.copy(X)

        for z in range (len(X)):
            softmax[z] = np.exp(X[z]) / np.sum(np.exp(X[z]), axis=0)

        return softmax


    def make_one_hot(self, n_classes, labels):
        """
        Makes the input array one HOT encoded
        :param n_classes:
        :param labels:
        :return:
        """

        return np.eye(int(n_classes))[label.astype(np.int16)]


    def display_ROC_graph(self, plot=False):
        """
        Displays a receiver operator graph
        :param plot:
        :return:
        """

        plt.figure()
        lw = 2
        plt.plot(fpr[2], tpr[2], color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()


    def skmetrics_results(self):
        pass

        # print
        # "validation accuracy:", val_accuracy
        # y_true = np.argmax(test_label, 1)
        # print
        # "Precision", sk.metrics.precision_score(y_true, y_pred)
        # print
        # "Recall", sk.metrics.recall_score(y_true, y_pred)
        # print
        # "f1_score", sk.metrics.f1_score(y_true, y_pred)
        # print
        # "confusion_matrix"
        # print
        # sk.metrics.confusion_matrix(y_true, y_pred)
        # fpr, tpr, tresholds = sk.metrics.roc_curve(y_true, y_pred)