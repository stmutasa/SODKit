"""
SOD_Recurrent contains the class utilized in implementing recurrent convolutional neural networks:
Conv-GRU
Conv-LSTM
"""


from SODNetwork import tf
from SODNetwork import np
from SODNetwork import SODMatrix
from SOD_ResNet import ResNet
from SOD_DenseNet import DenseNet
import tensorflow.contrib.slim as slim


class MRCNN(SODMatrix):

    """
    The multiple inheritence class to perform all functions of making mask RCNNs!
    """

    # Shared class variables here

    def __init__(self):
        pass