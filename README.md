SODKit Base classes:

SODLoader contains functions for loading and preprocessing files. Everything from loading a 3d or 2d DICOM to normalizing MRI images to saving a numpy dictionary in a binary format.

SODNetwork contains wrappers for the base neural network functions in Tensorflow, such as convolutions, residual convolutions, and various popular loss functions.

SODTester contains functions for evaluating inference. Like calculating AUC or DICE scores

SODKit Derived Classes:

SOD_ResNet contains a wrapper for setting up a residual style network in 2D, 2.5D or a residual style Unet in 2D or 2.5D

SOD_DenseNet contains a wrapper for setting up a Dense network in 2D, 2.5D or a Dense style Unet in 2D or 2.5D

SOD_ObjectDetection contains wrappers for The popular object detection networks Faster-RCNN and MASK-RCNN

SOD_Display contains wrappers for displaying data in various ways from 3D volumes, mosaics to histograms and bar plots.
