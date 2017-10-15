"""
SOD Tester contains function wrappers for various metric and performance analysis functions including

mean absolute error, mean squared error, DICE score, sensitivity, specificity, AUC

"""

import os, glob

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as skm
import pandas as pd

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

    def calculate_mean_absolute_error(self, prediction, labels, display=True):

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
        try: to_print = min(len(label), 15)
        except: to_print = 1

        # Calculate MAE
        MAE = np.mean(np.absolute((predictions - label)))

        # Print the summary
        np.set_printoptions(precision=1)  # use numpy to print only the first sig fig
        if display: print('MAE: %s\n Pred: %s\nReal: %s' % (MAE, predictions[:to_print], label[:to_print]))

        # Append right
        self.right += MAE
        self.total += 1

        return MAE


    def retreive_accuracy_regression(self, Epoch, display=True):

        """
        Retreive the accuracy so far
        :param Epoch: what epoch we're in
        :param display: Whether to print
        :return: accuracy
        """

        # Calculate final MAE and ACC
        self.MAE = self.right / self.total

        # Print the final accuracies and MAE if requested
        if display:
            print('-' * 70)
            print('--- EPOCH: %s MAE: %.3f (Old Best: %.3f @ %s) ---'
                    % (Epoch, self.MAE, self.best_MAE, self.best_step))

        # Update bests
        if self.MAE <= self.best_MAE: self.best_step, self.best_MAE = Epoch, self.MAE

        return self.MAE


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

        # Try/if statement for cases with one dimensional logit matrix (one example)
        try: logit = np.squeeze(np.argmax(logits.astype(np.float), axis=1))
        except:
            logit = np.expand_dims(logits.astype(np.float), 0)
            logit = np.squeeze(np.argmax(logit, axis=1))

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


    def calc_multiclass_square_metrics(self, logits, labels, Epoch, n_classes):
        """
        Calculates and displays SN and specificity of multiclass labels.
        :param logits:
        :param labels:
        :param Epoch:
        :param n_classes:
        :return:
        """

        # Retreive and print the labels and logits
        label = np.squeeze(labels.astype(np.int8))

        # Try/if statement for cases with one dimensional logit matrix (one example)
        try:
            logit = np.squeeze(np.argmax(logits.astype(np.float), axis=1))
        except:
            logit = np.expand_dims(logits.astype(np.float), 0)
            logit = np.squeeze(np.argmax(logit, axis=1))

        # Iterate over the number of classes
        for positive_class in range(n_classes):

            # Initialize counters
            right, TP, FN, TN, FaP = 0,0,0,0,0
            sensitivity, specificity, PPV, NPV = 0,0,0,0

            #  Retreive metrics
            for z in range(len(label)):

                # Metrics for when the ground truth is positive
                if label[z] == positive_class:

                    # Get metrics
                    if label[z] == logit[z]: TP += 1
                    if label[z] != logit[z]: FN += 1

                # Metrics for when the ground truth is negative
                if label[z] != positive_class:

                    # Get metrics
                    if logit[z] != positive_class: TN += 1
                    if logit[z] == positive_class: FaP += 1

            # Now calculate this class' scores
            try: sensitiviy = TP / (TP + FN)
            except: sensitiviy = 0

            try: specificity = TN / (TN + FaP)
            except: specificity = 0

            try: PPV = TP / (TP + FaP)
            except: PPV = 0

            try: NPV = TN / (TN + FN)
            except: PPV = 0

            # Accuracy
            accuracy = 100 * (TP+TN) / len(label)

            # Print
            print('--- Class %s ACC: %.2f, SN: %.3f, SP: %.3f, PPV: %.3f, NPV: %.3f ---'
                  % (positive_class, accuracy, sensitiviy, specificity, PPV, NPV), end=' | ')
            print('--- True Pos: %s, False Pos: %s, True Neg: %s, False Neg: %s ---'
                  % (TP, FaP, TN, FN))

            # Finally update the class tracker
            self.TP += TP
            self.TN += TN
            self.FP = FaP
            self.FN += FN

        # Overall SN and SP
        try: self.sensitiviy = self.TP / (self.TP + self.FN)
        except: self.sensitiviy = 0

        try: self.specificity = self.TN / (self.TN + self.FP)
        except: self.specificity = 0

        print ('--- Overall Sensitivity: %.3f, Specificity: %.3f' %(self.sensitiviy, self.specificity))


    def calculate_multiclass_metrics(self, logitz, labelz, step, n_classes, display=True, individual=False):
        """
        Calculates the metrics for a multiclass problem
        :param logits: raw logits = will be softmaxed
        :param labels: raw labels = will be made one hot
        :param step: the current step
        :param n_classes: number of total classes
        :param display: whether to print a summary
        :param individual: print individual outputs or not
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
            if int(label[z]) == int(logit[z]): self.right += 1

        # Increment total
        self.total += len(label)

        # calc accuracy
        self.accuracy = float(self.right/self.total)

        # Print Summary if wanted
        if display:

            # How many to print
            to_print = min(len(label), 15)

            # Now print
            print('-' * 70)
            print('Patient %s Class: %s' % (step, label[:to_print]))
            print('Patient %s Preds: %s' % (step, logit[:to_print]))

            # Print individual
            if individual:
                for z in range(len(label)): print(label[z], '=', logit[z], end=' | ')
                print (' ')

            # Display one per class
            for z in range (n_classes): print ('Class %s: %.3f --- '%(z, self.roc_auc[z]), end='')
            print ('Micro AUC: %.3f, Macro AUC: %.3f, ACC: %.2f'
                   %(self.roc_auc['micro'], self.roc_auc["macro"], self.accuracy*100))


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


    def calc_metrics_segmentation(self, logitz, label_in, pt, slice, images=None, step=1, dice_threshold=0.5, batch_size=32):

        # Convert to numpy arrays
        logitz = np.squeeze(logitz.astype(np.float))
        images = np.squeeze(images.astype(np.float))
        labelz = np.squeeze(label_in.astype(np.float))
        slice = np.asarray(slice)
        pt = np.asarray(pt)
        dice_score = 0

        for i in range(0, batch_size):

            # Retreive one image, label and prediction from the batch to save
            prediction = logitz[i, :, :, 1]

            # Manipulations to improve display data
            pred1 = 1 - prediction  # Invert the softmax

            # First create copies
            p1 = np.copy(pred1)  # make an independent copy of logits map
            p2 = np.copy(labelz[i])  # make an independent copy of labels map

            # Now create boolean masks
            p1[p1 > dice_threshold] = True  # Set predictions above threshold value to True
            p1[p1 <= dice_threshold] = False  # Set those below to False
            p2[p2 == 0] = False  # Mark lung and background as False
            p2[p2 > 0] = True  # Mark nodules as True

            # calculate DICE score
            dice_score = self.calc_DICE(p1, p2, 1.0)

            if np.sum(p2) > 10:
                print('Sums: ', np.sum(p1), np.sum(p2))
                self.display_single_image(p1, False, 'Inverse_Predictions')
                self.display_single_image(p2, False, 'Label')
                self.display_single_image(prediction, False, 'Raw_Prediction')
                self.display_single_image(images[0, 2, :, :], True, 'Input')

        # garbage
        del logitz, images, labelz, slice, pt, prediction, pred1, p1, p2

        # Return the DICE score
        return dice_score/batch_size


    def calc_DICE(self, im1, im2, empty_score=1.0):
        """
        Computes the DICE coefficient
        :param im1: binary array
        :param im2: binary array
        :param empty_score:
        :return: DICE score float bet 0-1. If both are empty then score is 1.0
        Notes
        -----
        The order of inputs for `dice` is irrelevant. The result will be
        identical if `im1` and `im2` are switched.
        """

        im1 = np.asarray(im1).astype(np.bool)
        im2 = np.asarray(im2).astype(np.bool)

        if im1.shape != im2.shape:
            raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

        im_sum = im1.sum() + im2.sum()
        if im_sum == 0:
            return empty_score

        # Compute Dice coefficient
        intersection = np.logical_and(im1, im2)

        return 2. * intersection.sum() / im_sum


    def combine_predictions(self, ground_truth, predictions, unique_ID, batch_size):
        """
        Combines multi parametric predictions into one group
        :param ground_truth:
        :param predictions:
        :param unique_ID:
        :param batch_size:
        :return:
        """

        # Convert to numpy arrays
        predictions, label = np.squeeze(predictions.astype(np.float)), np.squeeze(ground_truth.astype(np.float))
        serz = np.squeeze(unique_ID)

        # The dictionary to return
        data = {}

        # add up the predictions
        for z in range(batch_size):

            # If we already have the entry then just append
            if serz[z] in data:
                data[serz[z]]['log1'] = data[serz[z]]['log1'] + predictions[z]
                data[serz[z]]['total'] += 1
            else:
                data[serz[z]] = {'label': label[z], 'log1': predictions[z], 'total': 1, 'avg': None}

        # Initialize new labels and logits
        logga, labba = [], []

        # Combine the data
        for idx, dic in data.items():

            # Calculate the new average
            avg = np.asarray(dic['log1']) / dic['total']

            # Append to the new arrays
            labba.append(dic['label'])
            logga.append(np.squeeze(avg))

            # add to the dictionary
            dic['avg'] = np.squeeze(avg)

        return data, labba, logga


    def make_one_hot(self, n_classes, labels):
        """
        Makes the input array one HOT encoded
        :param n_classes:
        :param labels:
        :return:
        """

        return np.eye(int(n_classes))[labels.astype(np.int16)]


    def save_dic_csv(self, dictionary, filename='Submission', append=False, transpose=True):
        """
        Saves a ready made dictionary to CSV
        :param dictionary: the input dictionary
        :param filename: filename to save
        :param append: whether to append an existing csv or make a new one
        :param transpose: whether to transpose the dictionary
        :return:
        """

        # Now create the data frame and save the csv
        df = pd.DataFrame(dictionary)

        # Transpose the data frame
        if transpose: df.transpose()

        # Append if the flag is determined
        if append:
            with open(filename, 'a') as f:  df.to_csv(f, index=True, index_label='Batch_Num', header=False)

        # Otherwise make a new CSV
        else: df.to_csv(filename, index=True, index_label='Lymph_Node', )


    def save_to_csv(self, patients, predictions, step, error, filename='submission'):
        """
        Saves the patient dictionaries to a CSV
        :param patients: the dictionary of patient indexes
        :param predictions: the predicted outputs of the network
        :param step: the current step. If not 0 then we will be appending the csv not making a new one
        :param error: the column with the error
        :param filename: the filename to save
        :return:
        """

        # Make dummy dictionary
        dictionary = {}

        # Loop through the dictionary
        for idx, array in patients.items():

            # no need to save the image data
            if idx == 'image': continue

            # append new dictionary
            dictionary[idx] = np.squeeze(np.array(array))

        # Append the predictions
        dictionary['predictions'] = predictions
        dictionary['error'] = error

        # Now create the data frame and save the csv
        df = pd.DataFrame(dictionary)

        # Append if this is not the first step
        if step != 0:
            with open(filename, 'a') as f: df.to_csv(f, index=True, index_label='Batch_Num', header=False)

        # Otherwise make a new CSV
        else: df.to_csv(filename, index=True, index_label='Batch_Num')


    def display_ROC_graph(self, plot=False):
        """
        Displays a receiver operator graph
        :param plot:
        :return:
        """

        # Plot all ROC curves
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        plt.show()


    def calculate_boneage_errors(self, predictions, label):
        """
        This function retreives the labels and predictions and then outputs the accuracy based on the actual
        standard deviations from the atlas of bone ages. The prediction is considered "right" if it's within
        two standard deviations
        :param predictions:
        :param labels:
        :param girls: Whether we're using the female or male standard deviations
        :return: Accurace : calculated as % of right/total
        """

        # First define our variables:
        right = 0.0  # Number of correct predictions
        total = predictions.size  # Number of total predictions
        std_dev = np.zeros_like(predictions, dtype='float32')  # The array that will hold our STD Deviations
        tot_err = 0.0

        # No apply the standard deviations
        for i in range(0, total):

            # Bunch of if statements assigning the STD for the patient's true age
            if FLAGS.model < 3:  # Girls
                if label[i] <= (3 / 12):
                    std_dev[i] = 0.72 / 12
                elif label[i] <= (6 / 12):
                    std_dev[i] = 1.16 / 12
                elif label[i] <= (9 / 12):
                    std_dev[i] = 1.36 / 12
                elif label[i] <= (12 / 12):
                    std_dev[i] = 1.77 / 12
                elif label[i] <= (18 / 12):
                    std_dev[i] = 3.49 / 12
                elif label[i] <= (24 / 12):
                    std_dev[i] = 4.64 / 12
                elif label[i] <= (30 / 12):
                    std_dev[i] = 5.37 / 12
                elif label[i] <= 3:
                    std_dev[i] = 5.97 / 12
                elif label[i] <= 3.5:
                    std_dev[i] = 7.48 / 12
                elif label[i] <= 4:
                    std_dev[i] = 8.98 / 12
                elif label[i] <= 4.5:
                    std_dev[i] = 10.73 / 12
                elif label[i] <= 5:
                    std_dev[i] = 11.65 / 12
                elif label[i] <= 6:
                    std_dev[i] = 10.23 / 12
                elif label[i] <= 7:
                    std_dev[i] = 9.64 / 12
                elif label[i] <= 8:
                    std_dev[i] = 10.23 / 12
                elif label[i] <= 9:
                    std_dev[i] = 10.74 / 12
                elif label[i] <= 10:
                    std_dev[i] = 11.73 / 12
                elif label[i] <= 11:
                    std_dev[i] = 11.94 / 12
                elif label[i] <= 12:
                    std_dev[i] = 10.24 / 12
                elif label[i] <= 13:
                    std_dev[i] = 10.67 / 12
                elif label[i] <= 14:
                    std_dev[i] = 11.3 / 12
                elif label[i] <= 15:
                    std_dev[i] = 9.23 / 12
                else:
                    std_dev[i] = 7.31 / 12

            else:  # Boys
                if label[i] <= (3 / 12):
                    std_dev[i] = 0.72 / 12
                elif label[i] <= (6 / 12):
                    std_dev[i] = 1.13 / 12
                elif label[i] <= (9 / 12):
                    std_dev[i] = 1.43 / 12
                elif label[i] <= (12 / 12):
                    std_dev[i] = 1.97 / 12
                elif label[i] <= (18 / 12):
                    std_dev[i] = 3.52 / 12
                elif label[i] <= (24 / 12):
                    std_dev[i] = 3.92 / 12
                elif label[i] <= (30 / 12):
                    std_dev[i] = 4.52 / 12
                elif label[i] <= 3:
                    std_dev[i] = 5.08 / 12
                elif label[i] <= 3.5:
                    std_dev[i] = 5.40 / 12
                elif label[i] <= 4:
                    std_dev[i] = 6.66 / 12
                elif label[i] <= 4.5:
                    std_dev[i] = 8.36 / 12
                elif label[i] <= 5:
                    std_dev[i] = 8.79 / 12
                elif label[i] <= 6:
                    std_dev[i] = 9.17 / 12
                elif label[i] <= 7:
                    std_dev[i] = 8.91 / 12
                elif label[i] <= 8:
                    std_dev[i] = 9.10 / 12
                elif label[i] <= 9:
                    std_dev[i] = 9.0 / 12
                elif label[i] <= 10:
                    std_dev[i] = 9.79 / 12
                elif label[i] <= 11:
                    std_dev[i] = 10.09 / 12
                elif label[i] <= 12:
                    std_dev[i] = 10.38 / 12
                elif label[i] <= 13:
                    std_dev[i] = 10.44 / 12
                elif label[i] <= 14:
                    std_dev[i] = 10.72 / 12
                elif label[i] <= 15:
                    std_dev[i] = 11.32 / 12
                elif label[i] <= 16:
                    std_dev[i] = 12.86 / 12
                else:
                    std_dev[i] = 13.05 / 12

            # Calculate the MAE
            if predictions[i] < 0: predictions[i] = 0
            if predictions[i] > 18: predictions[i] = 18
            abs_err = abs(predictions[i] - label[i])
            tot_err += abs_err

            # Mark it right if we are within 2 std_devs
            if abs_err <= (std_dev[i] * 2):  # If difference is less than 2 stddev
                right += 1

        accuracy = (right / total) * 100  # Calculate the percent correct
        mae = (tot_err / total)

        return accuracy, mae


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


    def display_overlay(self, img, mask):
        """
        Method to superimpose masks on 2D image
        :params
        (np.array) img : 2D image of format H x W or H x W x C
          if C is empty (grayscale), image will be converted to 3-channel grayscale
          if C == 1 (grayscale), image will be squeezed then converted to 3-channel grayscale
          if C == 3 (rgb), image will not be converted
        (np.array) mask : 2D mask(s) of format H x W or N x H x W
        """

        # Adjust shapes of img and mask
        if len(img.shape) == 3 and img.shape[-1] == 1:
            img = np.squeeze(img)
        if len(img.shape) == 2:
            img = self.gray2rgb(img)
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, 0)
        mask = mask.astype('bool')

        # Overlay mask(s)
        if np.shape(img)[2] == 3:
            rgb = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0]]
            overlay = []
            for channel in range(3):
                layer = img[:,:,channel]
                for i in range(mask.shape[0]):
                    layer[mask[i, :,:]] = rgb[i % 6][channel]
                layer = np.expand_dims(layer, 2)
                overlay.append(layer)
            return np.concatenate(tuple(overlay), axis=2)


    def display_single_image(self, nda, plot=True, title=None, cmap='gray', margin=0.05):
        """ Helper function to display a numpy array using matplotlib
        Args:
            nda: The source image as a numpy array
            title: what to title the picture drawn
            margin: how wide a margin to use
            plot: plot or not
        Returns:
            none"""

        # Set up the figure object
        fig = plt.figure()
        ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])

        # The rest is standard matplotlib fare
        plt.set_cmap(cmap)  # Print in greyscale
        ax.imshow(nda)

        if title: plt.title(title)
        if plot: plt.show()