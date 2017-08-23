"""
SOD Tester contains function wrappers for various metric and performance analysis functions including

mean absolute error, mean squared error, DICE score, sensitivity, specificity, AUC

"""

import tensorflow as tf
import numpy as np
import sklearn.metrics as skm


class SODTester():

    """
    SOD Tester class is a class for testing the performance of our network
    """

    # Linear regression variables
    MAE = 0
    best_MAE = 10
    accuracy = 0

    # Classification variables
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    sensitiviy = 0      # Also known as recall
    specificity = 0
    PPV = 0             # Also known as precision
    NPV = 0
    F1_score = 0
    AUC = 0

    # Other variables
    right = 0
    total = 0
    best_step = 0
    calls = 0

    def __init__(self):
        pass

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


    def retreive_metrics_classification(self, Epoch, display=True):

        """
        Retreives sn, sp, PPV, NPV, ROC
        :param Epoch: What epoch we're in
        :param display: Whether to print the results
        :return: 
        """

        # Calculate the metrics. To prevent division by zero, use error handling
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
            print('--- EPOCH: %s, ACC: %.2f, SN: %.3f, SP: %.3f, AUC: %.3f, F1: %.3f ---'
                  % (Epoch, self.accuracy, self.sensitiviy, self.specificity, self.AUC, self.F1_score))
            print('--- True Pos: %s, False Pos: %s, True Neg: %s, False Neg: %s ---'
                  % (self.TP, self.FP, self.TN, self.FN))


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