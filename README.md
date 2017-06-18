# SODKit
Helper classes
   """
     Data Loading Functions. Support for DICOM, Nifty, CSV
    """


    def load_DICOM_3D(self, path, dtype=np.int16, overwrite_dims=513):
        """
        This function loads a DICOM folder and stores it into a numpy array. From Kaggle
        :param: path: The path of the DICOM folder
        :param: overwrite_dims = In case slice dimensions can't be retreived, define overwrite dimensions here
        :param: dtype = what data type to save the image as
        :return: image = A 3D numpy array of the image
        :return: numpyorigin, the real world coordinates of the origin
        :return: numpyspacing: An array of the spacing of the CT scanner used
        :return: spacing: The spacing of the pixels in millimeters
        """


    def load_DICOM_2D(self, path, dtype=np.int16):

        """
        This function loads a 2D DICOM file and stores it into a numpy array. From Bone Age
        :param path: The path of the DICOM files
        :param: dtype = what data type to save the image as
        :return: image = A numpy array of the image
        :return: accno = the accession number
        :return: dims = the dimensions of the image
        :return: window = the window level of the file
        """

    def load_CSV_Dict(self, indexname, path):
        """
        This function loads the annotations into a dictionary of dictionary with the columns as keys
        :param indexname: what column name to assign as the index for each dictionary
        :param path: file name
        :return: return_dict: a dict of dicts with indexname as the pointer and each entry based on the title row
        """

    def load_NIFTY(self, path):
        """
        This function loads a .nii.gz file into a numpy array with dimensions Z, Y, X, C
        :param filename:
        :return:
        """


    def load_MHA(self, path):
        """
            Loads the .mhd image and stores it into a numpy array
            :param filename: The name of the file
            :return: ndimage = A 3D numpy array of the image
            :return: numpyorigin, the real world coordinates of the origin
            :return spacing: An array of the spacing of the CT scanner used
        """


    def load_image(self, path, grayscale=True):
        """
        Loads an image from a jpeg or other type of image file into a numpy array
        :param path:
        :param grayscale:
        :return:
        """


    def load_HA_labels(self, filename):

        """
        To retreive the labels and patient ID using Dr. Ha's naming convention
        :param filename: the file name
        :return: label, id : self explanatory
        """



    """
             Pre processing functions.
    """

    def world_to_voxel(self, worldCoord, origin, spacing):
        """
        A function to convert world coordinates to voxel coordinates
        :param worldCoord: The pixel location in world coordinates as [x, y, z]
        :param origin: The world coordinate origin of the scan
        :param spacing: the spacing dimensions of the scan
        :return: voxelCoord: The pixels coordinates in voxels
        """


    def create_lung_mask(self, image, radius_erode=2):
        """
        Method to create lung mask. Returns lungs as 1, everything else as 0
        """

    def screate_bone_mask(self, image, seed):
        """
        Creates a bone mask
        :param image: input image
        :param seed: seedpoints for non bone areas to start the threshold
        :return: image: the bone mask with areas of bone = 0
        """
        

    def create_MIP_2D(self, vol, slice, thickness=5.0, slice_spacing=1.0):
        """
        This function creates a MIP of one given input slice of the given thickness
        :param vol: The input image (i.e. the entire Ct scan)
        :param slice: the slice we want to MIP
        :param thickness: the desired thickness (in real world dimensions)
        :param slice_spacing: the spacing of the slices in the real world
        :return MIP the maximum projection intensity of the given image with the thickness
        """


    def create_volume_MIP(self, volume, slice_spacing=1.0, thickness=5.0):

        """
        This function creates a MIP of an entire volume
        :param volume: the whole volume of data
        :param slice_spacing: real world spacing of the slices
        :param thickness: MIP thickness in real world mm
        :return:
        """


    def create_2D_label(self, input, coords, diameter):
        """
        This function creates a 2D label Mask for the given slice of the input image
        :param input: the input mask (this is a binary operation!)
        :param coords: the location of the label
        :param diamter: diameter of the nodule
        :return label_mask: a mask of the entire CT with labels
        """


    def create_3D_label(self, lung_mask, coords, diameter, spacing):
        """
        This function creates a 3D sphere shaped label
        :param lung_mask: The binary lung mask array
        :param coords: Location of the center of the label
        :param diameter: Diameter desired in real world mm
        :param spacing: pixel spacing of the scan
        :return:
        """


    def resample(self, image, spacing, new_spacing=[1, 1, 1]):
        """
        This function resamples the input volume into an isotropic resolution
        :param image: Input image
        :param spacing: the spacing of the scan
        :param new_spacing: self explanatory
        :return: image, new spacing: the resampled image
        """


    def affine_transform_data(self, data, tform, data_key=1):
        """
        Method to augment data by affine transform and random offset.

        :params

        (np.array) data : 2D image (H x W x C) or 3D volume (Z x H x W x C)
          Note, all channels are automatically transformed at one time.
        (str) data_key : to determine interpolation method
        (np.array) tform : affine transformation matrix

        """


    """
         Utility functions: Random tools for help
    """

    def random_3daffine(self, angle=45):
        """
        Method to generate a random 4 x 4 affine transformation matrix.
        :param angle: the range of angles to generate

        """

    def display_single_image(self, nda, plot=True, title=None, cmap='gray', margin=0.05):
        """ Helper function to display a numpy array using matplotlib
        Args:
            nda: The source image as a numpy array
            title: what to title the picture drawn
            margin: how wide a margin to use
            plot: plot or not
        Returns:
            none"""
        
    def reshape_NHWC(self, vol, NHWC):
        """
        Method to reshape 2D or 3D tensor into Tensorflow's NHWC format
        vol: The image data
        NHWC whether the input has a channel dimension
        """

    def gray2rgb(self, img, maximum_val=1, percentile=0):
        """
        Method to convert H x W grayscale tensor to H x W x 3 RGB grayscale
        :params
        (np.array) img : input H x W tensor
        (int) maximum_val : maximum value in output
          if maximum_val == 1, output is assumed to be float32
          if maximum_val == 255, output is assumed to be uint8 (standard 256 x 256 x 256 RGB image)
        (int) percentile : lower bound to set to 0
        """


    def imoverlay(self, img, mask):
        """
        Method to superimpose masks on 2D image
        :params
        (np.array) img : 2D image of format H x W or H x W x C
          if C is empty (grayscale), image will be converted to 3-channel grayscale
          if C == 1 (grayscale), image will be squeezed then converted to 3-channel grayscale
          if C == 3 (rgb), image will not be converted
        (np.array) mask : 2D mask(s) of format H x W or N x H x W
        """


    def retreive_filelist(self, extension, include_subfolders=False, path=None):
        """
        Returns a list with all the files of a certain type in path
        :param extension: what extension to search for
        :param include_subfolders: whether to include subfolders
        :param path: specified path. otherwise use data root
        :return:
        """


    def display_mosaic(self, vol, fig=None, title=None, size=[10, 10], vmin=None, vmax=None,
               return_mosaic=False, cbar=True, return_cbar=False, **kwargs):
        """
        Display a 3-d volume of data as a 2-d mosaic
        :param vol: The 3D numpy array of the data
        :param fig: matplotlib figure, optional If this should appear in an already existing figure instance
        :param title: str, the title
        :param size: the height of each slice
        :param vmin: upper and lower clip-limits on the color-map
        :param vmax:
        :param return_mosaic:
        :param cbar:
        :param return_cbar:
        :param kwargs: **kwargs: additional arguments to matplotlib.pyplot.matshow
        :return: fig: the figure handle
        """


    """
         Tool functions: Most of these are hidden
    """

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


    def create_feature_dict(self, data_to_write={}, id=1):
        """
        Create the features of each image:label pair we want to save to our TFRecord protobuf here instead of inline
        :param data_to_write: The data we will be writing into a dict
        :param id: The ID of this entry
        :return:
        """


    def imfill(self, img, connectivity=4):
        """
        Method to fill holes (binary).
        """


    def find_z_range(self, mask, min_size=0.01):
        """
        Method to find range of z-slices containing a mask surface area > min_size.
        """
