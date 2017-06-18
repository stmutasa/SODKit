"""
SOD Loader is the class for loading various file types including:
JPegs
Nifty
DICOM

It then stores the file as a numpy array and has functions to create the protocol buffer

"""

import glob, os, dicom, csv, random, cv2, math
import scipy.ndimage

import numpy as np
import nibabel as nib
import tensorflow as tf
import SimpleITK as sitk

from matplotlib import pyplot as plt
from skimage import morphology
import matplotlib.image as mpimg

class SODLoader():

    """
    SOD Loader class is a class for loading all types of data into protocol buffers
    """

    def __init__(self, data_root):

        """
        Initializes the class handler object
        :param data_root: The source directory with the data files
        """

        self.data_root = data_root
        self.files_in_root = glob.glob('**/*', recursive=True)

        # Data is all the data, everything else is instance based
        self.data = {}

        # Stuff for the dictionary
        self.label = None
        self.image = None
        self.origin = None
        self.spacing = None
        self.dims = None
        self.patient = None     # Usually the accession number


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

        # First try for files without the DICM marker
        try:

            # Populate an array with the dicom slices
            ndimage = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]

        # Otherwise they have the marker
        except:

            # Find all the .dcm files
            filenames = glob.glob(path + '/' + '*.dcm')

            # Populate an array with the dicom slices
            ndimage = [dicom.read_file(s) for s in filenames]

        # Sort the slices
        ndimage.sort(key=lambda x: int(x.ImagePositionPatient[2]))

        # Retreive slice thickness by subtracting the real world location of the first two slices
        try:
            slice_thickness = np.abs(ndimage[0].ImagePositionPatient[2] - ndimage[1].ImagePositionPatient[2])
        except:
            slice_thickness = np.abs(ndimage[0].SliceLocation - ndimage[1].SliceLocation)

        # Retreive the dimensions of the scan
        try:
            dims = np.array([int(ndimage[0].Columns), int(ndimage[0].Rows)])
        except:
            dims = np.array([overwrite_dims, overwrite_dims])

        # Retreive the spacing of the pixels in the XY dimensions
        pixel_spacing = ndimage[0].PixelSpacing

        # Create spacing matrix
        numpySpacing = np.array([slice_thickness, float(pixel_spacing[0]), float(pixel_spacing[1])])

        # Retreive the origin of the scan
        orig = ndimage[0].ImagePositionPatient

        # Make a numpy array of the origin
        numpyOrigin = np.array([float(orig[2]), float(orig[0]), float(orig[1])])

        # Finally, make the image actually equal to the pixel data and not the header
        image = np.stack([s.pixel_array for s in ndimage])

        # Set image data type to the type specified
        image = image.astype(dtype)

        # Convert to Houndsfield units
        try:
            for sl in range(len(ndimage)):

                # retreive the slope and intercept of this slice
                slope = ndimage[sl].RescaleSlope
                intercept = ndimage[sl].RescaleIntercept

                # If the slope isn't 1, rescale the images using the slope
                if slope != 1:
                    image[sl] = slope * image[sl].astype(np.float64)
                    image[sl] = image[sl].astype(dtype)

                # Reset the Intercept
                image[sl] += dtype(intercept)

        except: pass

        return image, numpyOrigin, numpySpacing, dims


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

        # Load the Dicom
        try:
            ndimage = dicom.read_file(path)
        except:
            print('For some reason, cant load: %s' % path)
            return

        # Retreive the dimensions of the scan
        dims = np.array([int(ndimage.Columns), int(ndimage.Rows)])

        # Retreive the window level
        window = [int(ndimage.WindowCenter), int(ndimage.WindowWidth)]

        # Retreive the dummy accession number
        accno = int(ndimage.AccessionNumber)

        # Finally, make the image actually equal to the pixel data and not the header
        image = np.asarray(ndimage.pixel_array, dtype)

        # Convert to Houndsfield units if slope and intercept is available:
        try:
            # retreive the slope and intercept of this slice
            slope = ndimage.RescaleSlope
            intercept = ndimage.RescaleIntercept

            # If the slope isn't 1, rescale the images using the slope
            if slope != 1:
                image = slope * image.astype(np.float64)
                image = image.astype(dtype)

            # Reset the Intercept
            image += dtype(intercept)
        except: pass

        return image, accno, dims, window


    def load_CSV_Dict(self, indexname, path):
        """
        This function loads the annotations into a dictionary of dictionary with the columns as keys
        :param indexname: what column name to assign as the index for each dictionary
        :param path: file name
        :return: return_dict: a dict of dicts with indexname as the pointer and each entry based on the title row
        """

        # Create the reader object to load
        reader = csv.DictReader(open(path))

        # Initialize the return dictionary
        return_dict = {}

        # Iterate and append the dictionary
        for row in reader:

            # Name the key as the indexname
            key = row.pop(indexname)

            # What to do if there is a duplicate
            if key in return_dict:
                print ("Duplicate entry found in dict %s" %key)
                continue

            # Make the entire row (as a dict) the index
            return_dict[key] = row

        return return_dict


    def load_NIFTY(self, path):
        """
        This function loads a .nii.gz file into a numpy array with dimensions Z, Y, X, C
        :param filename:
        :return:
        """

        # Load the data from the nifty file
        raw_data = nib.load(path)

        # Reshape the image data from NiB's XYZ to numpy's ZYXC
        data = self.reshape_NHWC(raw_data.get_data(), False)

        # Return the data
        return data


    def load_MHA(self, path):
        """
            Loads the .mhd image and stores it into a numpy array
            :param filename: The name of the file
            :return: ndimage = A 3D numpy array of the image
            :return: numpyorigin, the real world coordinates of the origin
            :return spacing: An array of the spacing of the CT scanner used
            """

        # Load the image.
        itkimage = sitk.ReadImage(path)
        ndimage = sitk.GetArrayFromImage(itkimage)

        # Retreive the original origin and spacing of the image
        numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
        numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

        return ndimage, numpyOrigin, numpySpacing


    def load_image(self, path, grayscale=True):
        """
        Loads an image from a jpeg or other type of image file into a numpy array
        :param path:
        :param grayscale:
        :return:
        """

        # Make the image into a greyscale by default
        if grayscale:
            image = mpimg.imread(path)

        # Else keep all the color channels
        else:
            image = mpimg.imread(path)

        return image


    def load_HA_labels(self, filename):

        """
        To retreive the labels and patient ID using Dr. Ha's naming convention
        :param filename: the file name
        :return: label, id : self explanatory
        """

        # os.path.basename returns : 20110921_103117ROUTINEBREAST721115s10402a1001-label.nii.gz
        basename = os.path.basename(filename)

        # Retreive the ID from the basename
        id = basename[:-13]

        # os.path.dirname returns : source/Type1/108-15-p1
        dirname = os.path.dirname(filename)

        # Retreive the label from the dirname
        label = os.path.split(dirname)[-2].split('/')[-1]

        # Retreive the patient name
        patient = os.path.split(dirname)[-1]

        return label, id, patient



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

        # Shift the position based on the origin
        stretchedVoxelCoord = np.absolute(worldCoord - origin)

        # Move to another
        voxelCoord = stretchedVoxelCoord / spacing

        return voxelCoord


    def create_lung_mask(self, image, radius_erode=2):
        """
        Method to create lung mask.
        """
        height = image.shape[1]  # Holder for the variable
        radius_close = np.round(height / 12).astype('int16')
        radius_dilate = np.round(height / 12).astype('int16')

        kernel_erode = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(radius_erode, radius_erode))
        kernel_close = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(radius_close, radius_close))
        kernel_dilate = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(radius_dilate, radius_dilate))
        shape = image.shape[1:3]

        edge_mask = np.zeros(shape=shape, dtype='bool')
        edge_mask[0:2, :] = True
        edge_mask[:, 0:2] = True
        edge_mask[-2:, :] = True
        edge_mask[:, -2:] = True

        mask = np.squeeze(image < -500)

        # Remove background air
        for z in range(mask.shape[0]):
            foreground = cv2.erode(src=mask[z, :].astype('uint8'), kernel=kernel_erode, iterations=1)
            background = cv2.erode(src=(~mask[z, :]).astype('uint8'), kernel=kernel_erode, iterations=1)
            N, markers = cv2.connectedComponents(image=foreground, connectivity=4, ltype=cv2.CV_32S)
            markers[background.astype('bool')] = N
            labels = cv2.watershed(image=np.zeros(shape=[shape[0], shape[1], 3], dtype='uint8'),
                                   markers=markers.astype('int32'))

            edges = np.unique(labels[edge_mask])
            for edge in edges[1:]:
                mask[z, :] = mask[z, :] & ~(labels == edge)
                mask[z, :] = mask[z, :] & ~edge_mask

        # Create seed_mask with horizontal line to force connection between lungs
        z = np.sum(mask, axis=(1, 2)).argmax()
        qh = np.round(height / 4).astype('int16')
        seed_mask = np.zeros_like(mask[z], dtype='bool')
        seed_mask[qh * 2, qh:qh * 3] = 1
        seed_mask = seed_mask & ~mask[z]
        mask[z] = mask[z] | seed_mask

        # Keep blob with seed_mask
        labels = morphology.label(mask)
        N = np.unique(labels[z][seed_mask])
        labels[z][seed_mask] = 0
        mask = labels == N

        # Apply morphologic operations (close > dilate > fill)
        z_range = self.find_z_range(mask=mask, min_size=0.0)
        for z in range(z_range[0], z_range[1]):
            mask[z, :] = cv2.morphologyEx(src=mask[z, :].astype('uint8'), op=cv2.MORPH_CLOSE, kernel=kernel_close)
            mask[z, :] = cv2.dilate(src=mask[z, :].astype('uint8'), kernel=kernel_dilate, iterations=1)
            mask[z, :] = self.imfill(mask[z, :])

        return mask


    def screate_bone_mask(self, image, seed):
        """
        Creates a bone mask
        :param image: input image
        :param seed: seedpoints for non bone areas to start the threshold
        :return: image: the bone mask with areas of bone = 0
        """
        """
        Method to create bone mask.
        """

        # Holder for the height variable
        slices = image.shape[0]

        # Apply the curve flow image filter, first convert to sitk image
        image = sitk.GetImageFromArray(image)

        # Now apply the filter
        image = sitk.CurvatureFlow(image, 0.125, 5)

        # Edge detection. Essentially plots the magnitude of the difference in contrast gradients of two pixels
        image = sitk.GradientMagnitude(image)

        # Connected threshold labels all pixels connected to the source pixels as long as they're within the threshold
        image = sitk.ConnectedThreshold(image1=image, seedList=seed, lower=0, upper=55, replaceValue=1)

        # Return to numpy array
        image = sitk.GetArrayFromImage(image)

        # Define the CV2 structuring element
        radius_close = np.round(image.shape[1] / 90).astype('int16')
        kernel_close = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(radius_close, radius_close))

        # Close the connected threshold bone mask
        for z in range(image.shape[0]):
            # Just use morphological closing
            image[z] = cv2.morphologyEx(image[z], cv2.MORPH_CLOSE, kernel_close)

        # Return the inverted masks
        return image


    def create_MIP_2D(self, vol, slice, thickness=5.0, slice_spacing=1.0):
        """
        This function creates a MIP of one given input slice of the given thickness
        :param vol: The input image (i.e. the entire Ct scan)
        :param slice: the slice we want to MIP
        :param thickness: the desired thickness (in real world dimensions)
        :param slice_spacing: the spacing of the slices in the real world
        :return MIP the maximum projection intensity of the given image with the thickness
        """

        # First retreive the thickness in slice coordinates
        slice_thickness = np.round_(thickness / slice_spacing).astype('int16')

        # Initialize the MIP
        # MIP = np.zeros_like(image)

        # Create a MIP of the given slice
        MIP = np.amax(vol[slice - slice_thickness:slice + slice_thickness], axis=0)

        return MIP


    def create_volume_MIP(self, volume, slice_spacing=1.0, thickness=5.0):

        """
        This function creates a MIP of an entire volume
        :param volume: the whole volume of data
        :param slice_spacing: real world spacing of the slices
        :param thickness: MIP thickness in real world mm
        :return:
        """

        # define radius
        r = np.round(thickness / slice_spacing).astype('int16')

        mip = np.zeros_like(volume)
        for z in range(r, volume.shape[0] - r - 1):
            mip[z] = np.amax(volume[z - r:z + r], axis=0)

        return mip


    def create_2D_label(self, input, coords, diameter):
        """
        This function creates a 2D label Mask for the given slice of the input image
        :param input: the input mask (this is a binary operation!)
        :param coords: the location of the label
        :param diamter: diameter of the nodule
        :return label_mask: a mask of the entire CT with labels
        """

        # Create the blank label matrix
        label_3d = np.zeros_like(input, dtype='bool')

        # Initialize the coordinates
        z = coords[0]
        y = coords[1]
        x = coords[2]

        # Set the nodule center pixel to 0
        label_3d[z, y, x] = 1

        diam = int(diameter[0] * 2)

        # Dilate by the calculated diameter
        kernel_dilate = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(diam, diam))
        label_2d = cv2.dilate(src=label_3d[z, :, :].astype('uint8'), kernel=kernel_dilate, iterations=1)

        return label_2d


    def create_3D_label(self, lung_mask, coords, diameter, spacing):
        """
        This function creates a 3D sphere shaped label
        :param lung_mask: The binary lung mask array
        :param coords: Location of the center of the label
        :param diameter: Diameter desired in real world mm
        :param spacing: pixel spacing of the scan
        :return:
        """

        # Append the mask with this nodule
        try:
            lung_mask[coords[0]] += self.create_2D_label(lung_mask, coords, [diameter, diameter, diameter])
        except:
            print("Unable to label")
            return

        # First set the radius we will iterate over in real world coordinates
        try:
            slices = int(diameter / spacing[0])
        except:
            print("Unable to label")
            return

        # Now retreive the slices above and below until (radius) away
        for z in range(1, slices + 1):
            # Formula for the radius of a dome at a certain level. It will give a value of 0 eventually
            diam2 = math.sqrt(max(1, (diameter ** 2 - (diameter - (diameter - z) + 1) ** 2)))
            diameter = np.asarray([diam2, diam2, diam2])

            # Append the mask with this nodule above and below
            lung_mask[coords[0] - z] += self.create_2D_label(lung_mask, coords, [diameter, diameter, diameter])
            lung_mask[coords[0] + z] += self.create_2D_label(lung_mask, coords, [diameter, diameter, diameter])

        return lung_mask


    def resample(self, image, spacing, new_spacing=[1, 1, 1]):
        """
        This function resamples the input volume into an isotropic resolution
        :param image: Input image
        :param spacing: the spacing of the scan
        :param new_spacing: self explanatory
        :return: image, new spacing: the resampled image
        """

        resize_factor = spacing / new_spacing
        new_real_shape = image.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / image.shape
        new_spacing = spacing / real_resize_factor

        image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

        return image, new_spacing


    def affine_transform_data(self, data, tform, data_key=1):
        """
        Method to augment data by affine transform and random offset.

        :params

        (np.array) data : 2D image (H x W x C) or 3D volume (Z x H x W x C)
          Note, all channels are automatically transformed at one time.
        (str) data_key : to determine interpolation method
        (np.array) tform : affine transformation matrix

        """
        # First retreive the center of rotation, i.e. the center of the image (really the neck) - SITK uses XYZ not ZYX
        center = [data.shape[2] / 2, data.shape[1] / 2, data.shape[0] / 2]

        # Interpolation method
        interpolation = sitk.sitkLinear if data_key == 1 else sitk.sitkNearestNeighbor

        # Change the numpy array into an SITK image
        data = sitk.GetImageFromArray(data.astype('float32'))

        # Define the holder for the affine transform variable
        T = sitk.AffineTransform(3)

        # Set the variable numbers based on the given transform matrix
        T.SetMatrix(sitk.VectorDouble(tform.astype('double').flatten()))

        # Define the center
        T.SetCenter(center)

        # Resamples the image using the affine transformation
        data = sitk.Resample(data, T, interpolation, sitk.sitkFloat32)

        # Return the data type to a numpy array
        data = sitk.GetArrayFromImage(data)

        # For some reason it sets background pixels to 8, set them back to air HU
        data[data == 8] = -100

        # Return the data
        return data


    """
         Utility functions: Random tools for help
    """

    def random_3daffine(self, angle=45):
        """
        Method to generate a random 4 x 4 affine transformation matrix.
        :param angle: the range of angles to generate

        """
        # Define the affine scaling
        sx = 1
        sy = 1
        sz = 1

        # Define the affine angles of rotation
        anglex = random.randrange(-angle, angle)
        anglex = math.radians(anglex)

        angley = random.randrange(-angle, angle)
        angley = math.radians(angley)

        anglez = random.randrange(-angle, angle)
        anglez = math.radians(anglez)

        # Define the first affine transform with scaling and translations and rotation about x
        tx = np.float32([[sx, 0, 0],
                         [0, sy * math.cos(anglex), -1 * math.sin(anglex)],
                         [0, math.sin(anglex), sz * math.cos(anglex)]])

        # Another for y (really z once MIPED)
        ty = np.float32([[math.cos(angley), 0, math.sin(angley)],
                         [0, 1, 0],
                         [-1 * math.sin(angley), 0, math.cos(angley)]])

        # Another for Z (really y once MIPED)
        tz = np.float32([[math.cos(anglez), -1 * math.sin(anglez), 0],
                         [math.sin(anglez), math.cos(anglez), 0],
                         [0, 0, 1]])

        return tx, ty, tz, anglex, angley, anglez


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


    def reshape_NHWC(self, vol, NHWC):
        """
        Method to reshape 2D or 3D tensor into Tensorflow's NHWC format
        vol: The image data
        NHWC whether the input has a channel dimension
        """

        # If this is a 2D image
        if len(vol.shape) == 2:

            # Create an extra channel at the beginning
            vol = np.expand_dims(vol, axis=0)

        # If there are 3 dimensions to the shape (2D with channels or 3D)
        if len(vol.shape) == 3:

            # If there is no channel dimension (i.e. grayscale)
            if not NHWC:

                # Move the last axis (Z) to the first axis
                vol = np.moveaxis(vol, -1, 0)

            # Create another axis at the end for channel
            vol = np.expand_dims(vol, axis=3)


        return vol


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
        img_min, img_max = np.percentile(img, percentile), np.percentile(img, 100 - percentile)
        img = (img - img_min) / (img_max - img_min)
        img[img > 1] = 1
        img[img < 0] = 0
        img = img * maximum_val
        img = np.expand_dims(img, 2)
        img = np.tile(img, [1, 1, 3])

        dtype = 'float32' if maximum_val == 1 else 'uint8'
        return img.astype(dtype)


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


    def retreive_filelist(self, extension, include_subfolders=False, path=None):
        """
        Returns a list with all the files of a certain type in path
        :param extension: what extension to search for
        :param include_subfolders: whether to include subfolders
        :param path: specified path. otherwise use data root
        :return:
        """

        # If no path specified use the default data root
        if not path: path = self.data_root

        # If we're including subfolders
        if include_subfolders: extension = ('**/*.%s' % extension)

        # Otherwise just search this folder
        else: extension = ('*.%s' %extension)

        # Join the pathnames
        path = os.path.join(path, extension)

        # Return the list of filenames
        return glob.glob(path, recursive=include_subfolders)


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

        if vmin is None:
            vmin = np.nanmin(vol)
        if vmax is None:
            vmax = np.nanmax(vol)

        sq = int(np.ceil(np.sqrt(len(vol))))

        # Take the first one, so that you can assess what shape the rest should be:
        im = np.hstack(vol[0:sq])
        height = im.shape[0]
        width = im.shape[1]

        # If this is a 4D thing and it has 3 as the last dimension
        if len(im.shape) > 2:
            if im.shape[2] == 3 or im.shape[2] == 4:
                mode = 'rgb'
            else:
                e_s = "This array has too many dimensions for this"
                raise ValueError(e_s)
        else:
            mode = 'standard'

        for i in range(1, sq):
            this_im = np.hstack(vol[(len(vol) / sq) * i:(len(vol) / sq) * (i + 1)])
            wid_margin = width - this_im.shape[1]
            if wid_margin:
                if mode == 'standard':
                    this_im = np.hstack([this_im,
                                         np.nan * np.ones((height, wid_margin))])
                else:
                    this_im = np.hstack([this_im,
                                         np.nan * np.ones((im.shape[2],
                                                           height,
                                                           wid_margin))])
            im = np.concatenate([im, this_im], 0)

        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, aspect='equal')
        else:
            # This assumes that the figure was originally created with this
            # function:
            ax = fig.axes[0]

        if mode == 'standard':
            imax = ax.matshow(im.T, vmin=vmin, vmax=vmax, **kwargs)
        else:
            imax = plt.imshow(np.rot90(im), interpolation='nearest')
            cbar = False
        ax.get_axes().get_xaxis().set_visible(False)
        ax.get_axes().get_yaxis().set_visible(False)
        returns = [fig]
        if cbar:
            # The colorbar will refer to the last thing plotted in this figure
            cbar = fig.colorbar(imax, ticks=[np.nanmin([0, vmin]),
                                             vmax - (vmax - vmin) / 2,
                                             np.nanmin([vmax, np.nanmax(im)])],
                                format='%1.2f')
            if return_cbar:
                returns.append(cbar)

        if title is not None:
            ax.set_title(title)
        if size is not None:
            fig.set_size_inches(size)

        if return_mosaic:
            returns.append(im)

        # If you are just returning the fig handle, unpack it:
        if len(returns) == 1:
            returns = returns[0]

        return returns


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

        # initialize an empty dictionary
        feature_dict_write = {}

        # id is the unique identifier of the image, make it an integer
        feature_dict_write['id'] = self._int64_feature(int(id))

        # Loop over the dictionary and append the feature list for each id
        for key, feature in data_to_write.items():

            # If this is our Data array, use the tostring() method.
            if key == 'data':
                feature_dict_write[key] = self._bytes_feature(feature.tobytes())  #

            else:  # Otherwise convert to a string and encode as bytes to pass on
                features = str(feature)
                feature_dict_write[key] = self._bytes_feature(features.encode())

        return feature_dict_write


    def imfill(self, img, connectivity=4):
        """
        Method to fill holes (binary).
        """
        edge_mask = np.zeros(shape=img.shape, dtype='bool')
        edge_mask[0, :] = True
        edge_mask[:, 0] = True
        edge_mask[-1:, :] = True
        edge_mask[:, -1:] = True

        _, labels = cv2.connectedComponents(
            image=(img == 0).astype('uint8'),
            connectivity=connectivity,
            ltype=cv2.CV_32S)

        filled = np.zeros(shape=img.shape, dtype='bool')
        edges = np.unique(labels[edge_mask & (img == 0)])
        for edge in edges:
            filled = filled | (labels == edge)

        return ~filled  # The ~ operator flips bits (turns 1's into 0's and visa versa)


    def find_z_range(self, mask, min_size=0.01):
        """
        Method to find range of z-slices containing a mask surface area > min_size.
        """
        z = np.sum(mask, axis=tuple(np.arange(1, len(mask.shape))))
        z = np.nonzero(z > (np.max(z) * min_size))
        return [z[0][0], z[0][-1]]



