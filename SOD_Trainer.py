"""
This class provides a wrapper for generating the training loop,
it's expected that most of these functions will be overwritten
"""

from SODNetwork import SODLoss, SODMatrix
from SODLoader import SODLoader
from SOD_Display import SOD_Display
from SODTester import SODTester

class SODTrain():

    def __init__(self):

        """
        Initializes the class handler object
        :param data_root: The source directory with the data files
        """

        # Data is all the data, everything else is instance based
        self.data = {}

        # Initialize some volumes
        self.images = None

        # Labels
        self.segmentation_labels = None
        self.classification_labels = None
        self.registration_labels = None

        # Logits
        self.segmentation_logits = None
        self.classification_logits = None
        self.registration_logits = None

        # Losses
        self.total_loss = None
        self.loss_type = 'CROSS_ENTROPY'
        self.L2_loss = None
        self.segmentation_loss = None
        self.classification_loss = None
        self.registration_loss = None

        # Training stuff
        self.phase_train = None
        self.global_step = None
        self.train_op = None


    def forward_pass(self):
        pass


    def calculate_loss(self):
        pass


    def backward_pass(self):
        pass


    def generate_inputs(self):
        pass
