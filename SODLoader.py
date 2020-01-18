"""
SOD Loader is the class for loading and preprocessing various file types including: JPegs Nifty and DICOM into numpy arrays.

There are also functions to preprocess the data including: segmenting lungs, generating cubes, and creating MIPs

It then contains functions to store the file as a protocol buffer

"""

import glob, os, csv, random, cv2, math, pickle, time
#import mudicom, astra

import numpy as np
import pydicom as dicom
import nibabel as nib
import tensorflow as tf
import SimpleITK as sitk
import matplotlib.image as mpimg
import pandas as pd

from scipy.io import loadmat
import scipy.ndimage as scipy
from scipy.ndimage import binary_fill_holes

import imageio
from medpy.io import save as mpsave
import nrrd
import mclahe as mc

import skimage.exposure as hist
from skimage import morphology
from skimage.filters.rank import median
from skimage.measure import regionprops
from skimage.morphology import disk
from skimage.segmentation import felzenszwalb
from skimage.transform import rescale

from moviepy.editor import ImageSequenceClip

# Encryption imports
import secrets
from base64 import urlsafe_b64encode as b64e, urlsafe_b64decode as b64d

from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

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


    def deID_DICOMs(self, path, password, delete=True):

        """
        This function deidentifies every DICOM file within every child directory provided in path
        :param path: The root directory to start working with
        :param password: A password to create an encrypted key of the pt data
        :param delete: Whether to delete the original file. Really should always be True unless debugging
        :return:
        """

        # Load all the files in all of the subfolders here
        path = os.path.join(path, '**/*.*')
        filenames = glob.glob(path, recursive=True)
        print('De-Identifying %s files... from root: ' % len(filenames), path)

        # remove nifti and nrrd files
        filenames = [x for x in filenames if 'nii' not in x or 'nrrd' not in x]

        index = 0
        for file in filenames:

            # PyDicom will only read dicom files, this is a nice checkpoint for non dicom files
            try:
                dataset = dicom.dcmread(file)
            except:
                print(file, ' is not a DICOM')
                continue

            # Patient sex is always there, if not, use this as a proxy that this patient has already been done
            try:
                _ = dataset.PatientSex
            except:
                print('Already deidentified %s' % file)
                continue

            ###############################################################################
            # pydicom allows to remove private tags using ``remove_private_tags`` method

            try:
                dataset.remove_private_tags()
            except:
                print('Private tag error with ', file)
                continue

            ###############################################################################
            # Data elements of type 3 (optional) can be easily deleted using ``del`` or ``delattr``

            # Someteimes age at exam is not listed
            try:
                age = dataset.PatientAge
            except:
                age = str(int(dataset.StudyDate[:4]) - int(dataset.PatientBirthDate[:4]))

            # Get the ACCno
            accno = dataset.AccessionNumber
            keystring = accno + dataset.PatientSex + age + '_' + dataset.PatientID

            # Create an encrypted Accno
            encrypted_accno = self.password_encrypt(keystring, password, return_string=True)

            # Delete Patient tags
            if 'OtherPatientIDs' in dataset: delattr(dataset, 'OtherPatientIDs')
            if 'PatientAddress' in dataset: delattr(dataset, 'PatientAddress')
            if 'PatientAge' in dataset: delattr(dataset, 'PatientAge')
            if 'PatientSex' in dataset: delattr(dataset, 'PatientSex')
            if 'PatientID' in dataset: delattr(dataset, 'PatientID')

            # Institution tags
            if 'ReferringPhysicianName' in dataset: delattr(dataset, 'ReferringPhysicianName')
            if 'InstitutionAddress' in dataset: delattr(dataset, 'InstitutionAddress')
            if 'InstitutionName' in dataset: delattr(dataset, 'InstitutionName')
            if 'IssuerOfPatientID' in dataset: delattr(dataset, 'IssuerOfPatientID')

            # Delete the year of acquisition, keep time
            if 'AcquisitionDate' in dataset: delattr(dataset, 'AcquisitionDate')
            if 'ContentDate' in dataset: delattr(dataset, 'ContentDate')
            if 'StudyDate' in dataset: delattr(dataset, 'StudyDate')
            if 'SeriesDate' in dataset: delattr(dataset, 'SeriesDate')

            # Replace tags that may be used to list patients, use time.time to pick a random number
            strings = time.strftime("%Y,%m,%d")
            t = strings.split(',')
            z = ''
            bdate = accno[:4] + z.join(t)[-4:]
            accno2 = z.join(t)[:4] + accno[-4:]
            if 'PatientBirthDate' in dataset: dataset.data_element('PatientBirthDate').value = bdate
            if 'PatientName' in dataset: dataset.data_element('PatientName').value = 'De Identified_' + str(time.time())[-6:]
            if 'AccessionNumber' in dataset: dataset.data_element('AccessionNumber').value = accno2
            if 'StudyID' in dataset: dataset.data_element('StudyID').value = encrypted_accno

            ##############################################################################
            # Finally, save the file again

            # Delete the original
            if delete: os.remove(file)

            # Save this new one
            dataset.save_as(file)

            if index % (len(filenames) // 10) == 0: print('*' * 20,
                                                          '\nDe-Identified %s of %s files' % (index, len(filenames)))
            index += 1
            del dataset, accno, keystring, encrypted_accno, accno2, bdate


    def ReID_DICOMs(self, password, file):

        """
        Function that returns a patient's accession number, MRN, gender, and date of birth
        This only works on files deidentified by the DeID_DICOMs function
        :param password: The password used to generate the key when the dicom was saved
        :param file: DICOM file to use to id this patient, one file per accession
        :return: {'MRN', 'Accession', 'Sex', 'Age'}
        """

        # PyDicom will only read dicom files, this is a nice checkpoint for non dicom files
        try:
            dataset = dicom.dcmread(file)
        except:
            print(file, ' is not a DICOM')
            return None

        # Return the information
        encrypted_info = dataset.StudyID
        try:
            decrypted_info = self.password_decrypt(encrypted_info, password, return_string=True)
        except:
            print('Password %s is incorrect!' % password)
            return None

        # Decypher
        MRN = decrypted_info.split('_')[-1]
        if 'F' in decrypted_info:
            Accno = decrypted_info.split('F')[0]
            Sex = 'F'
            Age = decrypted_info.split('F')[1].split('_')[0]
        else:
            Sex = 'M'
            Accno = decrypted_info.split('M')[0]
            Age = decrypted_info.split('M')[1].split('_')[0]

        # --- Save first slice for header information
        header = {'MRN': MRN, 'Accession': Accno, 'Sex': Sex, 'Age': Age}

        return header


    def load_DICOM_3D(self, path, dtype=np.int16, sort=False, overwrite_dims=513, display=False, return_header=False):

        """
        This function loads a DICOM folder and stores it into a numpy array. From Kaggle
        :param: path: The path of the DICOM folder
        :param sort: Whether to sort through messy folders for the actual axial acquisition. False, 'Lung' or 'PE'
        :param: overwrite_dims = In case slice dimensions can't be retreived, define overwrite dimensions here
        :param: dtype = what data type to save the image as
        :param: display = Whether to display debug text
        :param return_header = whether to return the header dictionary
        :return: image = A 3D numpy array of the image
        :return: numpyorigin, the real world coordinates of the origin
        :return: numpyspacing: An array of the spacing of the CT scanner used
        :return: spacing: The spacing of the pixels in millimeters
        :return header: a dictionary of the file's header information
        """

        # Some DICOMs end in .dcm, others do not
        fnames = list()
        for (dirpath, dirnames, filenames) in os.walk(path):
            fnames += [os.path.join(dirpath, file) for file in filenames]

        # Load the dicoms
        ndimage = [dicom.read_file(path, force=True) for path in fnames]

        # Calculate how many volumes are interleaved here
        sort_list = np.asarray([x.SliceLocation for x in ndimage], np.int16)
        _, counts = np.unique(sort_list, return_counts=True)
        repeats = np.max(counts)

        # Sort teh slices
        if sort:
            if 'Lung' in sort: ndimage = self.sort_DICOMS_Lung(ndimage, display, path)
            elif 'PE' in sort: ndimage = self.sort_DICOMS_PE(ndimage, display, path)
        ndimage, fnames, orientation, st, shape, four_d = self.sort_dcm(ndimage, fnames)
        ndimage.sort(key=lambda x: int(x.ImagePositionPatient[2]))

        # Retreive the dimensions of the scan
        try: dims = np.array([int(ndimage[0].Columns), int(ndimage[0].Rows)])
        except: dims = np.array([overwrite_dims, overwrite_dims])

        # Retreive the spacing of the pixels in the XY dimensions
        pixel_spacing = ndimage[0].PixelSpacing

        # Create spacing matrix
        numpySpacing = np.array([st, float(pixel_spacing[0]), float(pixel_spacing[1])])

        # Retreive the origin of the scan
        orig = ndimage[0].ImagePositionPatient

        # Make a numpy array of the origin
        numpyOrigin = np.array([float(orig[2]), float(orig[0]), float(orig[1])])

        # --- Save first slice for header information
        header = {'orientation': orientation, 'slices': shape[1], 'channels': shape[0], 'num_Interleaved': repeats,
                  'fnames': fnames, 'tags': ndimage[0], '4d': four_d, 'spacing': numpySpacing, 'origin': numpyOrigin}

        # Finally, load pixel data. You can use Imageio here
        try: image = np.stack([self.read_dcm_uncompressed(s) for s in ndimage])
        except: image = imageio.volread(path, 'DICOM')

        image = self.compress_bits(image)

        # Set image data type to the type specified
        image = image.astype(dtype)

        # Convert to Houndsfield units
        if hasattr(ndimage[0], 'RescaleIntercept') and hasattr(ndimage[0], 'RescaleSlope'):
            for slice_number in range(len(ndimage)):
                intercept = ndimage[slice_number].RescaleIntercept
                slope = ndimage[slice_number].RescaleSlope

                image[slice_number] = slope * image[slice_number].astype(np.float64)
                image[slice_number] = image[slice_number].astype('int16')
                image[slice_number] += np.int16(intercept)

        if return_header:
            return image, header
        else:
            return image, numpyOrigin, numpySpacing, dims


    def load_DICOM_Header(self, path, multiple=True):

        """
        This function loads a DICOM folder and stores it into a numpy array. From Kaggle
        :param: path: The path of the DICOM folder
        :param multiple: Whether the path contains single or multiple files of the same accno
        :return header: a dictionary of the file's header information
        """

        # For now just return accession number of the first file
        if multiple:

            # Some DICOMs end in .dcm, others do not
            fnames = list()
            for (dirpath, dirnames, filenames) in os.walk(path):
                fnames += [os.path.join(dirpath, file) for file in filenames]

            # Sort the slices
            try:
                ndimage = dicom.read_file(fnames[0])
            except:
                ndimage = dicom.read_file(fnames[1])

            # --- Save first slice for header information
            header = {'fname': fnames[0], 'tags': ndimage, 'path': path}

        else:

            # Load the Dicom
            ndimage = dicom.read_file(path)

            # --- Save first slice for header information
            header = {'fname': path, 'tags': ndimage}

        return header

    def load_nrrd(self, path, dtype=np.int16, pad_shape=None, dim3d=True):

        """
        Loads an NRRD with pynrrd and returns a numpy array
        :param path: file to load
        :param dtype: filetype to return
        :param pad_shape: NRRD sometimes doesnt save the whole volume (just the segments). And we have to pad it
        If you leave it at none we wont pad, otherwise we will
        :param dim3d: 3D or 2D
        :return: A numpy array
        """

        # Read file and return a tuple and header: z x,y
        data, header = nrrd.read(path, index_order='C')

        # Pad the segmentations to fill the volume shape desired, start with 3D mode
        if pad_shape and dim3d:

            # initiate the new array
            pad_shape = np.asarray(pad_shape)
            data_pad = np.zeros([pad_shape[0], pad_shape[1], pad_shape[2], data.shape[3]])

            # Get the offset origin  and extent from 3D slicer. This is in XYZ format so reverse
            origins = header['Segmentation_ReferenceImageExtentOffset'].split(' ')
            orig = [int(x) for x in origins] # XYZ
            orig = orig[::-1]
            end = [orig[0]+data.shape[0], orig[1]+data.shape[1], orig[2]+data.shape[2]]

            # Generate the array
            data_pad[orig[0]:end[0], orig[1]:end[1], orig[2]:end[2], :] = data

            # Return values to data
            data, data_pad = data_pad, data

            # Slicer swaps the y and x axes, so return to native numpy form
            data = np.swapaxes(data, 1, 2)

        # The 2D version
        elif pad_shape and not dim3d:

            # initiate the new array
            pad_shape = np.asarray(pad_shape)
            data_pad = np.zeros([pad_shape[0], pad_shape[1], data.shape[2]])

            # Get the offset origin  and extent from 3D slicer. This is in XYZ format so reverse
            origins = header['Segmentation_ReferenceImageExtentOffset'].split(' ')
            orig = [int(x) for x in origins]  # XY
            orig = orig[::-1]
            end = [orig[0] + data.shape[0], orig[1] + data.shape[1]]

            # Generate the array
            data_pad[orig[0]:end[0], orig[1]:end[1], :] = data

            # Return values to data
            data, data_pad = data_pad, data

            # Slicer swaps the y and x axes, so return to native numpy form
            data = np.swapaxes(data, 0, 1)

        return np.squeeze(data.astype(dtype)), header


    def save_nrrd(self, path, data):

        """
        Saves a numpy array as an nrrd
        :param path:
        :param data:
        :return:
        """

        nrrd.write(path, data)


    def load_MAT(self, path):
        """
        Loads a matlab .mat file
        :param path: 
        :return: 
        """

        return loadmat(path)


    def load_DICOM_2D(self, path, dtype=np.int16):

        """
        This function loads a 2D DICOM file and stores it into a numpy array. From Bone Age
        :param path: The path of the DICOM files
        :param: dtype = what data type to save the image as
        :return: image = A numpy array of the image
        :return: accno = the accession number
        :return: dims = the dimensions of the image
        :return: window = the window level of the file
        :return: photometric = the photometric interpretation. 1 = min values white, 2 = min values black
        """

        # Load the Dicom
        try:
            ndimage = dicom.read_file(path)
        except:
            print ('Unable to Load DICOM file: %s' % path)
            return -1

        # Retreive the dimensions of the scan
        try: dims = np.array([int(ndimage.Columns), int(ndimage.Rows)])
        except: dims = 0

        # Retreive window level if available
        try: window = [int(ndimage.WindowCenter), int(ndimage.WindowWidth)]
        except: window = 0

        # Retreive photometric interpretation (1 = negative XRay) if available
        try: photometric = int(ndimage.PhotometricInterpretation[-1])
        except: photometric = None

        # Retreive the accession number
        try: accno = ndimage.AccessionNumber
        except: accno = 0

        # Finally, make the image actually equal to the pixel data and not the header
        try: image = np.asarray(ndimage.pixel_array, dtype)
        except:
            try: image = np.asarray(ndimage.pixel_array)
            except:
                try: # Try using imageio
                    dirname = os.path.dirname(path)
                    image = imageio.imread(path, 'DICOM')
                    print ('Loaded DICOM with imagio ', path)
                except:
                    try: # Try using Simple ITK
                        image = np.squeeze(self._load_DICOM_ITK(path))
                        image = np.swapaxes(image, -1, 0)
                        print('Loaded DICOM with ITK ', path)
                    except:
                        print ('Unable to retreive Image Pixel Array: ', path)
                        return -1

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

        # --- Save first slice for header information
        header = {'fname': path, 'tags': ndimage}

        return image, accno, window, photometric, header


    def _load_DICOM_ITK(self, path):

        """
        When all else fails
        :param path:
        :return:
        """

        # Load the image.
        itkimage = sitk.ReadImage(path)
        ndimage = sitk.GetArrayFromImage(itkimage)

        return ndimage



    def Retrieve_DICOM_Header(self, path):

        # Load the Dicom
        try:
            ndimage = dicom.read_file(path)
        except:
            print('For some reason, cant load: %s' % path)
            return

        # Retreive the dimensions of the scan
        dims = np.array([int(ndimage.Columns), int(ndimage.Rows)])

        # Retreive window level if available
        try:
            window = [int(ndimage.WindowCenter), int(ndimage.WindowWidth)]
        except:
            window = None

        # Retreive photometric interpretation (1 = negative XRay) if available
        try:
            photometric = int(ndimage.PhotometricInterpretation[-1])
        except:
            photometric = None

        # Retreive Modality
        try:
            modality = ndimage.Modality
        except:
            modality = None

        # Retreive Spacing
        try:
            spacing = ndimage.PixelSpacing
        except:
            spacing = None

        # Retreive the dummy accession number
        try: accno = ndimage.AccessionNumber
        except: accno = None

        # Retreive gender
        try: sex = ndimage.PatientSex
        except: sex = None

        # Retreive Age
        try:
            age = ndimage.PatientAge
        except:
            age = None

        # Now retreive the date of the study
        try: study_date = ndimage.StudyDate
        except:
            try: study_date = ndimage.SeriesDate
            except:
                try: study_date = ndimage.AcquisitionDate
                except:
                    try: study_date = ndimage.ContentDate
                    except: study_date = None


        try: study_time = ndimage.StudyTime
        except:
            try: study_time = ndimage.SeriesTime
            except:
                try: study_time = ndimage.AcquisitionTime
                except:
                    try: study_time = ndimage.ContentTime
                    except: study_time = None


        return_dict = { 'dimensions': dims, 'window_level': window, 'photometric': photometric, 'accession': accno, 'sex': sex,
                        'modality': modality, 'spacing': spacing, 'age': age, 'study_date':study_date, 'study_time': study_time}

        return return_dict


    def load_BoneAge(self, path, dtype=np.int16):

        """
        This function loads a 2D DICOM file and stores it into a numpy array. From Bone Age
        :param path: The path of the DICOM files
        :param: dtype = what data type to save the image as
        :return: image = A numpy array of the image
        :return: accno = the accession number
        :return: dims = the dimensions of the image
        :return: window = the window level of the file
        :return: photometric = the photometric interpretation. 1 = min values white, 2 = min values black
        """

        # Load the Dicom
        try:
            ndimage = dicom.read_file(path)
        except:
            print('For some reason, cant load: %s' % path)
            return

        # # If this is not a hand radiograph, just delete it
        # NOT ACCURATE - Some hands don't have this flag
        # try:
        #     if ('HAND' in ndimage.BodyPartExamined): return
        # except:
        #     print('No Body Part listed: ', path)
        #     return

        # Retreive the dimensions of the scan
        dims = np.array([int(ndimage.Columns), int(ndimage.Rows)])

        # Retreive window level if available
        try: window = [int(ndimage.WindowCenter), int(ndimage.WindowWidth)]
        except: window = None

        # Retreive photometric interpretation (1 = negative XRay) if available
        try: photometric = int(ndimage.PhotometricInterpretation[-1])
        except: photometric = None

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

        return image, accno, dims, window, photometric


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

            # What to do if there is a duplicate: rename with 1 at the end,
            if key in return_dict:
                key = key + '1'

                # For another duplicate, do the same
                if key in return_dict:
                    key = key + '2'

            # Make the entire row (as a dict) the index
            return_dict[key] = row

        return return_dict


    def save_Dict_CSV(self, dict, path, orient='index'):
        """
        Saves a dictionary as a .CSV file
        :param dict: the input dictionary
        :param path: path to save
        :param orient: Key Orientation, index or columns
        :return:
        """

        # Define data frame and create a CSV from it
        df = pd.DataFrame.from_dict(dict, orient=orient)
        df.to_csv(path)


    def save_dict_pickle(self, dictionary, data_root='data/filetypes'):

        """
        Saves a dictionary to pickle. Good for saving data types
        :param dictionary: the dictionary to save
        :param data_root: the name of the file. "pickle.p" will be added
        :return:
        """

        filename = data_root + '_pickle.p'
        pickle._dump(dictionary, open(filename, 'wb'))


    def load_dict_pickle(self, filename='data/filetypes_pickle.p'):

        """
        This loads the dictionary from pickle
        :param filename: the save file
        :return: the loaded dictionary
        """
        return pickle.load(open(filename, 'rb'))


    def save_dict_filetypes(self, dict_index_0, data_root='data/filetypes'):

        """
        Function to save the loaded dictionary filetypes into a pickle file
        :param dict_index_0: the first entry of the data dictionary
        :param data_root: the first part of the filename
        :return:
        """

        pickle_dic = {}
        for key, val in dict_index_0.items():

            # Generate the type dictionary
            pickle_dic[key] = str(type(val))[8:-2]

            # Save the dictionary
            self.save_dict_pickle(pickle_dic, data_root)


    def load_tfrecords(self, dataset, data_dims=[], image_dtype=tf.float32, segments='label_data',
                       segments_dtype=tf.float32, segments_shape = []):

        """
        Function to load a tfrecord protobuf. numpy arrays (volumes) should have 'data' in them.
        Currently supports strings, floats, ints, and arrays
        :param dataset: the tf.dataset object
        :param data_dims: the dimensions of the image saved, i.e. ZxYxXxC or YxXxC
        :param image_dtype: the data type of the image. i.e. tf.float32
        :param segments: if labels exist as segments, define the name here if the z-dimension is different from images
        :param segments_dtype: the data type of the segments
        :param segments_shape: Shape of segments, i.e. ZxYxXxC or YxXxC
        :return: data: dictionary with all the loaded tensors
        """

        # Pickle load
        loaded_dict = self.load_dict_pickle()

        # Populate the feature dict
        feature_dict = {'id': tf.FixedLenFeature([], tf.int64)}
        for key, value in loaded_dict.items(): feature_dict[key] = tf.FixedLenFeature([], tf.string)

        # Parses one protocol buffer file into the features dictionary which maps keys to tensors with the data: 'key': parse_single_eg
        features = tf.parse_single_example(dataset, features=feature_dict)

        # Make a data dictionary and cast it to floats
        data = {'id': tf.cast(features['id'], tf.float32)}
        for key, value in loaded_dict.items():

            # Depending on the type key or entry value, use a different cast function on the feature
            if 'bbox' in key:
                data[key] = tf.decode_raw(features[key], tf.float32)
                data[key] = tf.reshape(data[key], shape=[-1, 5])
                #data[key] = tf.cast(data[key], tf.float32)

            elif segments in key:
                data[key] = tf.decode_raw(features[key], segments_dtype)
                data[key] = tf.reshape(data[key], shape=segments_shape)
                #data[key] = tf.cast(data[key], tf.float32)

            elif 'data' in key:
                data[key] = tf.decode_raw(features[key], image_dtype)
                data[key] = tf.reshape(data[key], shape=data_dims)
                # data[key] = tf.cast(data[key], tf.float32)

            elif 'str' in str(value):
                data[key] = tf.cast(features[key], tf.string)
            else:
                data[key] = tf.string_to_number(features[key], tf.float32)

        return data

    def load_tfrecord_labels(self, dataset, segments='label_data'):

        """
        Function to load a tfrecord labels only, not the images. Useful when we want to split this up
        """

        # Pickle load
        loaded_dict = self.load_dict_pickle()

        # Populate the feature dict
        feature_dict = {'id': tf.FixedLenFeature([], tf.int64)}
        for key, value in loaded_dict.items(): feature_dict[key] = tf.FixedLenFeature([], tf.string)

        # Parses one protocol buffer file into the features dictionary which maps keys to tensors with the data: 'key': parse_single_eg
        features = tf.parse_single_example(dataset, features=feature_dict)

        # Make a data dictionary and cast it to floats
        data = {'id': tf.cast(features['id'], tf.float32)}
        for key, value in loaded_dict.items():

            # Depending on the type key or entry value, use a different cast function on the feature
            if 'bbox' in key:
                data[key] = tf.decode_raw(features[key], tf.float32)
                data[key] = tf.reshape(data[key], shape=[-1, 5])

            elif segments in key:
                data[key] = features[key]

            elif 'data' in key:
                data[key] = features[key]

            elif 'str' in str(value):
                data[key] = tf.cast(features[key], tf.string)
            else:
                data[key] = tf.string_to_number(features[key], tf.float32)

        return data

    def load_tfrecord_images(self, dataset, data_dims=[], image_dtype=tf.float32, segments='label_data',
                             segments_dtype=tf.float32, segments_shape=[]):

        """
        Function to load a tfrecords images only. Useful when we want to split this up
        """

        for key, value in dataset.items():

            # Depending on the type key or entry value, use a different cast function on the feature
            if segments in key:
                dataset[key] = tf.decode_raw(value, segments_dtype)
                dataset[key] = tf.reshape(dataset[key], shape=segments_shape)

            elif 'data' in key:
                dataset[key] = tf.decode_raw(value, image_dtype)
                dataset[key] = tf.reshape(dataset[key], shape=data_dims)

            else:
                continue

        return dataset

    def load_tfrecords_dataset(self, filenames):

        # now load the remaining files
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=None)

        reader = tf.TFRecordReader()  # Instantializes a TFRecordReader which outputs records from a TFRecords file
        _, serialized_example = reader.read(
            filename_queue)  # Returns the next record (key:value) produced by the reader

        # Pickle load
        loaded_dict = self.load_dict_pickle()

        # Populate the feature dict
        feature_dict = {'id': tf.FixedLenFeature([], tf.int64)}
        for key, value in loaded_dict.items(): feature_dict[key] = tf.FixedLenFeature([], tf.string)

        # Parses one protocol buffer file into the features dictionary which maps keys to tensors with the data: 'key': parse_single_eg
        features = tf.parse_single_example(serialized_example, features=feature_dict)

        return features

        # # Make a data dictionary and cast it to floats
        # data = {'id': tf.cast(features['id'], tf.float32)}
        # for key, value in loaded_dict.items():
        #
        #     # Depending on the type key or entry value, use a different cast function on the feature
        #     if 'data' in key and segments not in key:
        #         data[key] = tf.decode_raw(features[key], image_dtype)
        #         data[key] = tf.reshape(data[key], shape=data_dims)
        #         data[key] = tf.cast(data[key], tf.float32)
        #
        #     if segments in key:
        #         data[key] = tf.decode_raw(features[key], segments_dtype)
        #         data[key] = tf.reshape(data[key], shape=segments_shape)
        #         data[key] = tf.cast(data[key], tf.float32)
        #
        #     if 'bbox' in key:
        #         data[key] = tf.decode_raw(features[key], tf.float32)
        #         data[key] = tf.reshape(data[key], shape=[-1, 5])
        #         data[key] = tf.cast(data[key], tf.float32)
        #
        #     elif 'str' in value: data[key] = tf.cast(features[key], tf.string)
        #     else: data[key] = tf.string_to_number(features[key], tf.float32)
        #
        # return data

    def oversample_class(self, example_class, actual_dists=[0.5, 0.5], desired_dists=[], oversampling_coef=0.9):

        """
            Calculates how many copies to make of each class in order to oversample some classes
            class_dist is an array with the actual distribution of each class
            class_target_dist is an array with the target distribution of each class, defaults to equal
            oversampling_coef smooths the calculation if equal to 0 then oversample_classes() always returns 1
            returns an int64
        """

        # Get this classes distributions
        if not desired_dists: desired_dists = [1 / len(actual_dists)] * len(actual_dists)
        desired_dists = tf.cast(desired_dists, dtype=tf.float32)
        actual_dists = tf.cast(actual_dists, dtype=tf.float32)

        class_target_dist = tf.squeeze(desired_dists[tf.cast(example_class, tf.int32)])
        class_dist = tf.squeeze(actual_dists[tf.cast(example_class, tf.int32)])

        # Get a ratio between these two. If overrepresented, this returns <1 else >1
        prob_ratio = tf.cast(class_target_dist / class_dist, dtype=tf.float32)

        # Factor to soften the chance of generating repeats, if oversampling_coef==0 we recover original distribution
        prob_ratio = prob_ratio ** oversampling_coef

        # For oversampled classes (low ratio) we # want to return 1. Otherwise save as int with math.floor
        prob_ratio = tf.math.maximum(prob_ratio, 1)
        repeat_count = tf.math.floor(prob_ratio)

        # To handle floats, there is a chance to return a higher integer than the floor above.
        # i.e. prob_ratio 1.9 returns 2 instead of 1 90% of the time
        repeat_residual = prob_ratio - repeat_count
        residual_acceptance = tf.math.less_equal(tf.random_uniform([], dtype=tf.float32), repeat_residual)

        # Convert to integers and return
        residual_acceptance = tf.cast(residual_acceptance, tf.int64)
        repeat_count = tf.cast(repeat_count, dtype=tf.int64)
        return repeat_count + residual_acceptance

    def undersample_filter(self, example_class, actual_dists=[0.5, 0.5], desired_dists=[], undersampling_coef=0.5):

        """
        Lets you know if you should reject this sample or not
        Must have a 'class_dist' index indicating the actual class frequency 0-1
        Must have 'class_target_dist' index indicating the desired frequency 0-1
        undersampling_coef if equal to 0 then undersampling_filter() always returns True
        returns true or false
        """

        # Get this classes distributions
        if not desired_dists: desired_dists = [1 / len(actual_dists)] * len(actual_dists)
        desired_dists = tf.cast(desired_dists, dtype=tf.float32)
        actual_dists = tf.cast(actual_dists, dtype=tf.float32)

        class_target_dist = tf.squeeze(desired_dists[tf.cast(example_class, tf.int32)])
        class_dist = tf.squeeze(actual_dists[tf.cast(example_class, tf.int32)])

        prob_ratio = tf.cast(class_target_dist / class_dist, dtype=tf.float32)
        prob_ratio = prob_ratio ** undersampling_coef
        prob_ratio = tf.minimum(prob_ratio, 1.0)

        acceptance = tf.less_equal(tf.random_uniform([], dtype=tf.float32), prob_ratio)
        # predicate must return a scalar boolean tensor
        return acceptance

    def load_NIFTY(self, path, reshape=True):
        """
        This function loads a .nii.gz file into a numpy array with dimensions Z, Y, X, C
        :param filename: path to the file
        :param reshape: whether to reshape the axis from/to ZYX
        :return:
        """

        #try:

        # Load the data from the nifty file
        raw_data = nib.load(path)

        # Reshape the image data from NiB's XYZ to numpy's ZYXC
        if reshape: data = self.reshape_NHWC(raw_data.get_data(), False)
        else: data = raw_data.get_data()

        # Return the data
        return np.squeeze(data)


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


    def load_sachin_labels(self, filename):

        """
        To retreive the labels and patient ID using sachin's naming convention
        :param filename: the file name. something like data/P_028_G_2_id_104152_label.nii.gz
        :return: label, id : self explanatory
        """

        # os.path.basename returns : P_001_G_2_id_100041_label.nii.gz
        basename = os.path.basename(filename)

        # Retreive the ID from the basename
        id = os.path.split(basename)[-1].split('id_')[-1][:6]

        # Retreive the label from the basename
        label = os.path.split(basename)[-1].split('G_')[-1][:1]

        # Retreive the patient name
        patient = os.path.split(basename)[-1].split('P_')[-1][:3]

        # retreive the phase name if available
        if 'D' in filename:
            phase = os.path.split(basename)[-1].split('D_')[-1][:2]
        else:
            phase = 'label'

        return patient, label, id, phase


    def load_HIRAM_labels(self, filename):
        """
        To retreive the labels and patient ID using Dr. Ha's naming convention
        :param filename: the file name: data/raw/12.1.1/o___-label.nii.gz
        :return: label, id : self explanatory
        """

        # os.path.dirname returns : data/raw/12.1.1, split and return the last string
        dirname = os.path.dirname(filename).split('/')[-1]

        # Retreive the label from the dirname
        ln = dirname.split('.')[-1]
        pt = dirname.split('.')[0]
        study = dirname.split('.')[1]

        return pt, study, ln


    def randomize_batches(self, image_dict, batch_size, dynamic=False):
        """
        This function takes our full data tensors and creates shuffled batches of data.
        :param image_dict: the dictionary of tensors with the images and labels
        :param batch_size: batch size to shuffle
        :return: 
        """

        min_dq = 16  # Min elements to queue after a dequeue to ensure good mixing
        capacity = min_dq + 3 * batch_size  # max number of elements in the queue
        keys, tensors = zip(*image_dict.items())  # Create zip object

        # This function creates batches by randomly shuffling the input tensors. returns a dict of shuffled tensors
        if dynamic: shuffled = tf.train.batch(tensors, batch_size=batch_size, capacity=capacity, dynamic_pad=True)
        else: shuffled = tf.train.shuffle_batch(tensors, batch_size=batch_size, capacity=capacity, min_after_dequeue=min_dq)

        # Dictionary to store our shuffled examples
        batch_dict = {}

        # Recreate the batched data as a dictionary with the new batch size
        for key, shuffle in zip(keys, shuffled): batch_dict[key] = shuffle

        return batch_dict


    def val_batches(self, image_dict, batch_size):

        """
        Loads a validation set without shuffling it
        :param image_dict: the dictionary of tensors with the images and labels
        :param batch_size: batch size to shuffle
        :return:
        """

        min_dq = 16  # Min elements to queue after a dequeue to ensure good mixing
        capacity = min_dq + 3 * batch_size  # max number of elements in the queue
        keys, tensors = zip(*image_dict.items())  # Create zip object

        # This function creates batches by randomly shuffling the input tensors. returns a dict of shuffled tensors
        shuffled = tf.train.batch(tensors, batch_size=batch_size, capacity=capacity)

        batch_dict = {}  # Dictionary to store our shuffled examples

        # Recreate the batched data as a dictionary with the new batch size
        for key, shuffle in zip(keys, shuffled): batch_dict[key] = shuffle

        return batch_dict


    def save_image(self, image, path, format=None, type=None):

        """
        Saves an image to disc
        :param image: Input tensor: can be image, or volume
        :param path: destination file
        :param format: The format to save in
        :param type: for volumes: either a gif or a volumetric image
        """

        # Way more powerful than this but we will go on a PRN basis
        imageio.imwrite(path, image, format=format)


    def save_gif_volume(self, volume, path, fps=None, scale=0.8, swapaxes=False, norm=True):

        """
        Saves the input volume to a .gif file
        """

        # Swapaxes, used if this is a nifti source
        if swapaxes: volume = np.swapaxes(volume, 1, 2)

        if norm:
            try:
                volume_norm = self.adaptive_normalization(volume, True)
            except:
                volume_norm = volume
            volume_norm = cv2.convertScaleAbs(volume_norm, alpha=(255.0 / volume_norm.max()))
        else:
            volume_norm = cv2.convertScaleAbs(volume, alpha=(255.0))

        # Save the .gif set FPS to volume depended
        if not fps: fps = volume_norm.shape[0] // 5
        self.gif(path, volume_norm, fps=fps, scale=scale)


    def gif(self, filename, array, fps=10, scale=1.0):
        """Creates a gif given a stack of images using moviepy
        Notes
        -----
        works with current Github version of moviepy (not the pip version)
        https://github.com/Zulko/moviepy/commit/d4c9c37bc88261d8ed8b5d9b7c317d13b2cdf62e
        Usage
        -----
        >>> X = randn(100, 64, 64)
        >>> gif('test.gif', X)
        Parameters
        ----------
        filename : string
            The filename of the gif to write to
        array : array_like
            A numpy array that contains a sequence of images
        fps : int
            frames per second (default: 10)
        scale : float
            how much to rescale each image by (default: 1.0)
        """

        # ensure that the file has the .gif extension
        fname, _ = os.path.splitext(filename)
        filename = fname + '.gif'

        # copy into the color dimension if the images are black and white
        if array.ndim == 3:
            array = array[..., np.newaxis] * np.ones(3)

        # make the moviepy clip
        clip = ImageSequenceClip(list(array), fps=fps).resize(scale)
        clip.write_gif(filename, fps=fps)
        return clip


    def save_volume(self, volume, path, header=False, overwrite=True, compress=False):
        """
        Saves a volume into nii. nii.gz, .dcm, .dicom, .nrrd, .mhd and others
        :param volume: 3d numpy array
        :param path: save path
        :param header: The header information with metadata
        :param overwrite: Overwrite existing
        :param compress: compress the image (if supported)
        :return:
        """

        mpsave(volume, path, header, overwrite, compress)


    def RCNN_extract_box_labels(self, mask, dim_3d=False):

        """
        Returns the bounding box labels from each 2D segmentation mask. Can work with multiple label groupings (classes)
        :param mask: Input mask in 2D [height, weight, channels] or 3D [slice, height, weight, channels]
        :param dim_3d: bool, whether input is 3D or 2D
        :return: np list of arrays of bounding box coordinates of the corners [N, (y1, x1, y2, x2)]
        """

        if dim_3d:

            # Expand dims
            if mask.ndim < 4: mask = np.expand_dims(mask, axis=-1)

            # Make dummy array with diff boxes for each channel
            boxes = np.zeros([mask.shape[0], mask.shape[-1], 4], dtype=np.int32)

            for a in range (mask.shape[0]):

                # Loop through all the classes (channels)
                for z in range(mask[a].shape[-1]):

                    # Work on just this class of pixel
                    m = mask[a, :, :, z]

                    # Bounding boxes generated by finding indices with true values
                    x_indices = np.where(np.any(m, axis=0))[0]
                    y_indices = np.where(np.any(m, axis=1))[0]

                    if x_indices.shape[0]:

                        x1, x2 = x_indices[[0, -1]]
                        y1, y2 = y_indices[[0, -1]]

                        # Increment x2 and y2 by 1 since theyre not part of the initial box
                        x2 += 1
                        y2 += 1

                    else:

                        # No mask for this instance, happens a lot. Sets boxes to zero
                        x1, x2, y1, y2 = 0, 0, 0, 0

                    # Append coordinates to boxes
                    boxes[a, z] = np.array([y1, x1, y2, x2])

            return boxes.astype(np.int32)

        else:

            # Expand dims
            if mask.ndim < 3: mask = np.expand_dims(mask, axis=-1)

            # Make dummy array with diff boxes for each channel
            boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)

            # Loop through all the classes (channels)
            for z in range (mask.shape[-1]):

                # Work on just this class of pixel
                m = mask[:, :, z]

                # Bounding boxes generated by finding indices with true values
                x_indices = np.where(np.any(m, axis=0))[0]
                y_indices = np.where(np.any(m, axis=1))[0]

                if x_indices.shape[0]:

                    x1, x2 = x_indices[[0, -1]]
                    y1, y2 = y_indices[[0, -1]]

                    # Increment x2 and y2 by 1 since theyre not part of the initial box
                    x2 += 1
                    y2 += 1

                else:

                    # No mask for this instance, happens a lot. Sets boxes to zero
                    x1, x2, y1, y2 = 0, 0, 0, 0

                # Append coordinates to boxes
                boxes[z] = np.array([y1, x1, y2, x2])

            return boxes.astype(np.int32)


    def resize_3D_by_axis(self, image, dim_1, dim_2, axis_to_hold, is_grayscale):

        """
        Resizes a 3D volume in tensorflow by axis.
        :param image:
        :param dim_1:
        :param dim_2:
        :param ax:
        :param is_grayscale:
        :return:
        """

        ax = axis_to_hold

        resized_list = []

        if is_grayscale:
            unstack_img_depth_list = [tf.expand_dims(x, 2) for x in tf.unstack(image, axis=ax)]
            for i in unstack_img_depth_list:
                resized_list.append(tf.image.resize_images(i, [dim_1, dim_2], method=0))
            stack_img = tf.squeeze(tf.stack(resized_list, axis=ax))
            print(stack_img.get_shape())

        else:
            unstack_img_depth_list = tf.unstack(image, axis=ax)
            for i in unstack_img_depth_list:
                resized_list.append(tf.image.resize_images(i, [dim_1, dim_2], method=0))
            stack_img = tf.stack(resized_list, axis=ax)

        return stack_img


    def tf_resize_3D(self, image, z_dim, x_dim, y_dim, is_grayscale):

        """
        To resize a 3D tensorflor tensor
        :param image:
        :param z_dim:
        :param x_dim:
        :param y_dim:
        :param is_grayscale:
        :return:
        """

        resized_along_depth = self.resize_3D_by_axis(image, y_dim, z_dim, 2, is_grayscale)
        resized_along_width = self.resize_3D_by_axis(resized_along_depth, y_dim, x_dim, 1, is_grayscale)

        return tf.transpose(resized_along_width, perm=[1, 0, 2])


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


    def create_MRI_breast_mask(self, image, threshold=15, size_denominator=45):
        """
        Creates a rough mask of breast tissue returned as 1 = breast 0 = nothing
        :param image: the input volume (3D numpy array)
        :param threshold: what value to threshold the mask
        :param size_denominator: The bigger this is the smaller the structuring element
        :return: mask: the mask volume
        """

        # Create the mask
        mask2 = np.copy(np.squeeze(image))
        mask1 = np.zeros_like(image, np.int16)

        # Loop through the image volume
        for k in range(0, image.shape[0]):

            # Apply gaussian blur to smooth the image
            mask2[k] = cv2.GaussianBlur(mask2[k], (5, 5), 0)
            # mask[k] = cv2.bilateralFilter(mask[k].astype(np.float32),9,75,75)

            # Threshold the image. Change to the bool
            mask1[k] = np.squeeze(mask2[k] < threshold).astype(np.int16)

            # Define the CV2 structuring element
            radius_close = np.round(mask1.shape[1] / size_denominator).astype('int16')
            kernel_close = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(radius_close, radius_close))

            # Apply morph close
            mask1[k] = cv2.morphologyEx(mask1[k], cv2.MORPH_CLOSE, kernel_close)

            # Invert mask
            mask1[k] = ~mask1[k]

            # Add 2
            mask1[k] += 2

        return mask1


    def create_mammo_mask(self, image, threshold=800, check_mask=False):

        """

        :param image: input mammogram
        :param threshold: Pixel value to use for threshold
        :param check_mask: Check mask to make sure it's not fraudulent, i.e. covers > 80% or <10% of the image
            if it is, then use the other mask function
        :return:
        """

        # Create the mask
        mask = np.copy(image)

        # Apply gaussian blur to smooth the image
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        # Threshold the image
        mask = np.squeeze(mask < threshold)

        # Invert mask
        mask = ~mask

        # Morph Dilate to close in bad segs

        # Define the CV2 structuring element
        radius_close = np.round(mask.shape[1] / 45).astype('int16')
        kernel_close = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(radius_close, radius_close))

        # Just use morphological closing
        mask = cv2.morphologyEx(mask.astype(np.int16), cv2.MORPH_CLOSE, kernel_close)

        if check_mask:

            # Check if mask is too much or too little
            mask_idx = np.sum(mask) / (image.shape[0] * image.shape[1])

            # If it is, try again
            if mask_idx > 0.8 or mask_idx < 0.1:

                print('Mask Failed... using method 2')
                del mask
                mask, _ = self.create_breast_mask2(image)


        return mask


    def create_breast_mask2(self, im, scale_factor=0.25, threshold=3900, felzenzwalb_scale=0.15):

        """
        Fully automated breast segmentation in mammographies.
        https://github.com/olieidel/breast_segment
        :param im: Image
        :param scale_factor: Scale Factor
        :param threshold: Threshold
        :param felzenzwalb_scale: Felzenzwalb Scale
        :return: (im_mask, bbox) where im_mask is the segmentation mask and
        bbox is the bounding box (rectangular) of the segmentation.
        """

        # set threshold to remove artifacts around edges
        im_thres = im.copy()
        im_thres[im_thres > threshold] = 0

        # determine breast side
        col_sums_split = np.array_split(np.sum(im_thres, axis=0), 2)
        left_col_sum = np.sum(col_sums_split[0])
        right_col_sum = np.sum(col_sums_split[1])

        if left_col_sum > right_col_sum:
            breast_side = 'l'
        else:
            breast_side = 'r'

        # rescale and filter aggressively, normalize
        im_small = rescale(im_thres, scale_factor)
        im_small_filt = median(im_small, disk(50))
        # this might not be helping, actually sometimes it is
        im_small_filt = hist.equalize_hist(im_small_filt)

        # run mr. felzenzwalb
        segments = felzenszwalb(im_small_filt, scale=felzenzwalb_scale)
        segments += 1  # otherwise, labels() would ignore segment with segment=0

        props = regionprops(segments)

        # Sort Props by area, descending
        props_sorted = sorted(props, key=lambda x: x.area, reverse=True)

        expected_bg_index = 0
        bg_index = expected_bg_index

        bg_region = props_sorted[bg_index]
        minr, minc, maxr, maxc = bg_region.bbox
        filled_mask = bg_region.filled_image

        im_small_fill = np.zeros((im_small_filt.shape[0] + 2, im_small_filt.shape[1] + 1), dtype=int)

        if breast_side == 'l':
            # breast expected to be on left side,
            # pad on right and bottom side
            im_small_fill[minr + 1:maxr + 1, minc:maxc] = filled_mask
            im_small_fill[0, :] = 1  # top
            im_small_fill[-1, :] = 1  # bottom
            im_small_fill[:, -1] = 1  # right
        elif breast_side == 'r':
            # breast expected to be on right side,
            # pad on left and bottom side
            im_small_fill[minr + 1:maxr + 1, minc + 1:maxc + 1] = filled_mask  # shift mask to right side
            im_small_fill[0, :] = 1  # top
            im_small_fill[-1, :] = 1  # bottom
            im_small_fill[:, 0] = 1  # left

        im_small_fill = binary_fill_holes(im_small_fill)

        im_small_mask = im_small_fill[1:-1, :-1] if breast_side == 'l' \
            else im_small_fill[1:-1, 1:]

        # rescale mask
        im_mask = self.zoom_2D(im_small_mask.astype(np.float32), [im.shape[1], im.shape[0]]).astype(bool)

        # invert!
        im_mask = ~im_mask

        # determine side of breast in mask and compare
        col_sums_split = np.array_split(np.sum(im_mask, axis=0), 2)
        left_col_sum = np.sum(col_sums_split[0])
        right_col_sum = np.sum(col_sums_split[1])

        if left_col_sum > right_col_sum:
            breast_side_mask = 'l'
        else:
            breast_side_mask = 'r'

        if breast_side_mask != breast_side:
            # breast mask is not on expected side
            # we might have segmented bg instead of breast
            # so invert again
            print('breast and mask side mismatch. inverting!')
            im_mask = ~im_mask

        # exclude thresholded area (artifacts) in mask, too
        im_mask[im > threshold] = False

        # fill holes again, just in case there was a high-intensity region
        # in the breast
        im_mask = binary_fill_holes(im_mask)

        # if no region found, abort early and return mask of complete image
        if im_mask.ravel().sum() == 0:
            all_mask = np.ones_like(im).astype(bool)
            bbox = (0, 0, im.shape[0], im.shape[1])
            print('Couldn\'t find any segment')
            return all_mask, bbox

        # get bbox
        minr = np.argwhere(im_mask.any(axis=1)).ravel()[0]
        maxr = np.argwhere(im_mask.any(axis=1)).ravel()[-1]
        minc = np.argwhere(im_mask.any(axis=0)).ravel()[0]
        maxc = np.argwhere(im_mask.any(axis=0)).ravel()[-1]

        bbox = (minr, minc, maxr, maxc)

        return im_mask, bbox


    def create_lung_mask(self, image, radius_erode=2, close=12, dilate=12):
        """
        Creates a binary lung mask, 1 = lung, 0 = not lung
        :param image: input lung CT
        :param radius_erode:
        :param close: a lower number closes more
        :param dilate: a lower number dilates more
        :return:
        """

        # Define the radius of the structuring elements
        height = image.shape[1]  # Holder for the variable
        radius_close = np.round(height / close).astype('int16')
        radius_dilate = np.round(height / dilate).astype('int16')

        # Create the structuring elements
        kernel_erode = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(radius_erode, radius_erode))
        kernel_close = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(radius_close, radius_close))
        kernel_dilate = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(radius_dilate, radius_dilate))
        shape = image.shape[1:3]

        # Start with an edge mask
        edge_mask = np.zeros(shape=shape, dtype='bool')
        edge_mask[0:2, :] = True
        edge_mask[:, 0:2] = True
        edge_mask[-2:, :] = True
        edge_mask[:, -2:] = True

        # STart the mask off with houndsfield units below this number
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


    def create_bone_mask(self, image, seed):
        """
        Creates a bone mask
        :param image: input image
        :param seed: seedpoints for non bone areas to start the threshold
        :return: image: the bone mask with areas of bone = 0
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


    def create_table_mask(self, image, factor=30):
        """
        Creates a mask of the table in scans
        :param image: the 3d volume to mask, numpy format
        :param factor: factor to divide by. the bigger it is, the smaller the radius of closing
        :return: the mask with table and background as 0
        """

        # Define the radius of the structuring element. Make it equal to the width of the scan divided by factor
        radius_close = np.round(image.shape[1] / factor).astype('int16')

        # Create the structuring element
        kernel_close = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(radius_close, radius_close))

        # Start the mask off with houndsfield units below this threshold
        mask = np.squeeze(image < -500)

        # Close the mask
        for z in range(image.shape[0]):
            # Just use morphological closing
            mask[z] = cv2.morphologyEx(mask[z].astype(np.int16), cv2.MORPH_CLOSE, kernel_close)

        # Return the inverted masks
        return ~mask


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

        image = scipy.interpolation.zoom(image, real_resize_factor, mode='nearest')

        return image, new_spacing


    def resize_volume(self, image, dtype, x=256, y=256, z=None, c=None):

        """
        Resize a volume to the new size using open CV
        :param image: input image array
        :param dtype: the data type of the input
        :param x: new x dimensino
        :param y:
        :param z: new z dimension
        :param c: new c dimension
        :return:
        """

        # Resize the array
        if not z: z=image.shape[0]

        if not c:
            resize = np.zeros((z, x, y), dtype)
            for idx in range(image.shape[0]): resize[idx] = self.zoom_2D(image[idx], [x, y])

        else:
            resize = np.zeros((z, x, y, c), dtype)
            for idx in range(image.shape[0]): resize[idx] = self.zoom_2D(image[idx], [x, y])

        # Return
        return resize


    def zoom_3D(self, volume, factor):
        """
        Uses scipy to zoom a 3D volume to a new shape
        :param volume: The input volume, numpy array
        :param factor: The rescale factor: an array corresponding to each axis to rescale
        :return: 
        """

        # Define the resize matrix
        resize_factor = [factor[0] * volume.shape[0], factor[1] * volume.shape[1], factor[2] * volume.shape[2]]
        resize_factor_depth = [factor[0] * volume.shape[0], factor[1] * volume.shape[1], factor[2] * volume.shape[2], 1]

        # Perform the zoom
        try: return scipy.interpolation.zoom(volume, resize_factor, mode='nearest')
        except: return scipy.interpolation.zoom(volume, resize_factor_depth, mode='nearest')


    def zoom_2D(self, image, new_shape):
        """
        Uses open CV to resize a 2D image
        :param image: The input image, numpy array
        :param new_shape: New shape, tuple or array
        :return: the resized image
        """
        return cv2.resize(image,(new_shape[0], new_shape[1]), interpolation = cv2.INTER_CUBIC)


    def fast_3d_affine(self, image, center, angle_range, shear_range=None):
        """
        Performs a 3D affine rotation and/or shear using OpenCV
        :param image: The image volume
        :param center: array: the center of rotation in z,y,x
        :param angle_range: array: range of angles about x, y and z
        :param shear_range: float array: the range of values to shear if you want to shear
        :return:
        """

        # The image is sent in Z,Y,X format
        C = None
        try: Z, Y, X = image.shape
        except: Z, Y, X, C = image.shape

        # OpenCV makes interpolated pixels equal 0. Add the minumum value to subtract it later
        img_min = abs(image.min())
        image = np.add(image, img_min)

        # Define the affine angles of rotation
        anglex = random.randrange(-angle_range[2], angle_range[2])
        angley = random.randrange(-angle_range[1], angle_range[1])
        anglez = random.randrange(-angle_range[0], angle_range[0])

        # Matrix to rotate along Coronal plane (Y columns, Z rows)
        M = cv2.getRotationMatrix2D((center[1], center[0]), anglex, 1)

        # Apply the Coronal transform slice by slice along X
        for i in range(0, X): image[:, :, i] = cv2.warpAffine(image[:, :, i], M, (Y, Z))

        # Matrix to rotate along saggital plane (Z and X) and apply
        M = cv2.getRotationMatrix2D((center[2], center[0]), angley, 1)
        for i in range(0, Y): image[:, i, :] = cv2.warpAffine(image[:, i, :], M, (X, Z))

        # Matrix to rotate along Axial plane (X and Y)
        M = cv2.getRotationMatrix2D((center[1], center[2]), anglez, 1)
        for i in range(0, Z): image[i, :, :] = cv2.warpAffine(image[i, :, :], M, (Y, X))

        # Done with rotation, return if shear is not defined
        if shear_range == None: return np.subtract(image, img_min), [anglex, angley, anglez]

        # Shear defined, repeat everything for shearing

        # Define the affine shear positions
        sx = random.uniform(0-shear_range[2], 0+shear_range[2])
        sy = random.uniform(0-shear_range[1], 0+shear_range[1])
        sz = random.uniform(0-shear_range[0], 0+shear_range[0])

        # First define 3 random points
        pts1 = np.float32([[X / 2, Z / 2], [X / 2, Z / 3], [X / 3, Z / 2]])

        # Then define a custom 3x3 affine matrix
        M = np.array([[1.0, sx, 0], [sz, 1.0, 0], [0, 0, 1.0]], dtype=np.float32)

        # Now retreive the affine matrix that OpenCV uses
        M = cv2.getAffineTransform(pts1, np.dot(M, pts1))

        # Apply the transformation slice by slice
        for i in range(0, Y): image[:, i, :] = cv2.warpAffine(image[:, i, :], M, (X, Z))

        # Garbage collections
        del pts1

        # Repeat and Apply the saggital transform slice by slice along X
        pts1 = np.float32([[Y / 2, Z / 2], [Y / 2, Z / 3], [Y / 3, Z / 2]])
        M = np.array([[1.0, sy, 0], [sz, 1.0, 0], [0, 0, 1.0]], dtype=np.float32)
        M = cv2.getAffineTransform(pts1, np.dot(M, pts1))
        for i in range(0, X): image[:, :, i] = cv2.warpAffine(image[:, :, i], M, (Y, Z))
        del pts1

        # Repeat and Apply the Coronal transform slice by slice along y
        pts1 = np.float32([[Y / 2, X / 2], [Y / 2, X / 3], [Y / 3, X / 2]])
        M = np.array([[1.0, sy, 0], [sx, 1.0, 0], [0, 0, 1.0]], dtype=np.float32)
        M = cv2.getAffineTransform(pts1, np.dot(M, pts1))
        for i in range(0, Z): image[i, :, :] = cv2.warpAffine(image[i, :, :], M, (Y, X))
        del pts1

        return np.subtract(image, img_min), [anglex, angley, anglez], [sx, sy, sz]


    def transform_volume_3d(self, image, angles, shift_range=None, center=None, radians=False):

        """
        Performs a 3D affine rotation and/or shift using OpenCV
        :param image: The image volume
        :param center: array: the center of rotation in z,y,x
        :param angles: array: angle about x, y and z in degrees
        :param shift_range: for translating
        :param radians: if the angle is in radians or not
        :return:
        """

        if radians: angles = [math.degrees(x) for x in angles]

        # Get center if not given
        if not center: center = [x//2 for x in image.shape]

        # The image is sent in Z,Y,X format
        C = None
        try: Z, Y, X = image.shape
        except: Z, Y, X, C = image.shape

        # OpenCV makes interpolated pixels equal 0. Add the minumum value to subtract it later
        img_min = abs(image.min())
        image = np.add(image, img_min)

        # Define the affine angles of rotation
        anglex = angles[2]
        angley = angles[1]
        anglez = angles[0]

        # Matrix to rotate along Coronal plane (Y columns, Z rows)
        M = cv2.getRotationMatrix2D((center[1], center[0]), anglex, 1)

        # Apply the Coronal transform slice by slice along X
        for i in range(0, X): image[:, :, i] = cv2.warpAffine(image[:, :, i], M, (Y, Z))

        # Matrix to rotate along saggital plane (Z and X) and apply
        M = cv2.getRotationMatrix2D((center[2], center[0]), angley, 1)
        for i in range(0, Y): image[:, i, :] = cv2.warpAffine(image[:, i, :], M, (X, Z))

        # Matrix to rotate along Axial plane (X and Y)
        M = cv2.getRotationMatrix2D((center[1], center[2]), anglez, 1)
        for i in range(0, Z): image[i, :, :] = cv2.warpAffine(image[i, :, :], M, (X, Y))

        # Done with rotation, return if shear is not defined
        if shift_range == None: return np.subtract(image, img_min)


    def fast_2d_affine(self, image, center, angle_range, shear_range=None):
        """
        Performs a 3D affine rotation and/or shear using OpenCV
        :param image: The image
        :param center: array: the center of rotation (make this the nodule center) in z,y,x
        :param angle_range: range of angles to rotate
        :param shear_range: float array: the range of values to shear if you want to shear
        :return:
        """

        # The image is sent in Z,Y,X format
        Y, X = image.shape[0], image.shape[1]

        # OpenCV makes interpolated pixels equal 0. Add the minumum value to subtract it later
        img_min = abs(image.min())
        image = np.add(image, img_min)

        # Define the affine angles of rotation
        angle = random.randrange(-angle_range, angle_range)

        # Matrix to rotate the image
        M = cv2.getRotationMatrix2D((center[0], center[1]), angle, 1)

        # Apply the affine transform
        image = cv2.warpAffine(image, M, (Y, X))

        # Done with rotation, return if shear is not defined
        if shear_range == None: return np.subtract(image, img_min), angle

        # Shear defined, repeat everything for shearing

        # Define the affine shear positions
        sx = random.uniform(0-shear_range[1], 0+shear_range[1])
        sy = random.uniform(0-shear_range[0], 0+shear_range[0])

        # First define 3 random points
        pts1 = np.float32([[Y / 2, X / 2], [Y / 2, X / 3], [Y / 3, X / 2]])

        # Then define a custom 3x3 affine matrix
        M = np.array([[1.0, sy, 0], [sx, 1.0, 0], [0, 0, 1.0]], dtype=np.float32)

        # Now retreive the affine matrix that OpenCV uses
        M = cv2.getAffineTransform(pts1, np.dot(M, pts1))

        # Apply the transformation slice by slice
        image = cv2.warpAffine(image, M, (Y, X))

        return np.subtract(image, img_min), angle, [sy, sx]


    def calc_fast_affine(self, center=[], angle_range=[], dim_3d=True):

        """
        This function returns 3 matrices that define affine rotations in 3D
        :param center: The center of the rotation
        :param angle_range: matrix describing range of rotation along z, y, x in degrees
        :param dim_3d: Whether this is 2 dimensional or 3D
        :return: array with the affine matrices
        """

        # First do 2D rotations
        if not dim_3d:
            angle = random.randrange(-angle_range, angle_range)
            M = cv2.getRotationMatrix2D(tuple(center), angle, 1)
            return M

        else:

            # Define the affine angles of rotation
            anglex = random.randrange(-angle_range[2], angle_range[2])
            angley = random.randrange(-angle_range[1], angle_range[1])
            anglez = random.randrange(-angle_range[0], angle_range[0])

            # Matrix to rotate along Coronal plane (Y columns, Z rows)
            Mx = cv2.getRotationMatrix2D((center[1], center[0]), anglex, 1)

            # Matrix to rotate along saggital plane (Z and X) and apply
            My = cv2.getRotationMatrix2D((center[2], center[0]), angley, 1)

            # Matrix to rotate along Axial plane (X and Y)
            Mz = cv2.getRotationMatrix2D((center[1], center[2]), anglez, 1)

            return [Mx, My, Mz]


    def perform_fast_affine(self, image, M=[], dim_3d=True):

        """
        This function applies an affine transform using the given affine matrices
        :param image: input volume
        :param M: Affine matrices along x, y and z
        :param dim_3d: whether this is 3D or 2D
        :return: image after warp
        """

        # The image is sent in Z,Y,X format
        if dim_3d: Z, Y, X = image.shape
        else: Y, X = image.shape

        # OpenCV makes interpolated pixels equal 0. Add the minumum value to subtract it later
        img_min = abs(image.min())
        image = np.add(image, img_min)

        # Apply the Affine transforms slice by slice
        if dim_3d:
            for i in range(0, X): image[:, :, i] = cv2.warpAffine(image[:, :, i], M[0], (Y, Z))
            for i in range(0, Y): image[:, i, :] = cv2.warpAffine(image[:, i, :], M[1], (X, Z))
            for i in range(0, Z): image[i, :, :] = cv2.warpAffine(image[i, :, :], M[2], (Y, X))

        else: image = cv2.warpAffine(image, M, (Y, X))

        # Return the array with normal houndsfield distribution
        return np.subtract(image, img_min)


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


    def generate_box(self, image, origin=[], size=32, display=False, dim3d=True, z_overwrite=None):
        """
        This function returns a cube from the source image
        :param image: The source image
        :param origin: Center of the cube as a matrix of x,y,z [z, y, x]
        :param size: dimensions of the cube in mm
        :param dim3d: Whether this is 3D or 2D
        :param z_overwrite: Use this to overwrite the size of the Z axis, otherwise it defaults to half
        :return: cube: the cube itself
        """

        # Sometimes size is an array
        try:
            sizey = int(size[0])
            sizex = int(size[1])
        except:
            sizey = size
            sizex = size

        # First implement the 2D version
        if not dim3d:

            # Make the starting point = center-size unless near the edge then make it 0
            startx = max(origin[1] - sizex / 2, 0)
            starty = max(origin[0] - sizey / 2, 0)

            # If near the far edge, make it fit inside the image
            if (startx + sizex) > image.shape[1]:
                startx = image.shape[1] - sizex
            if (starty + sizey) > image.shape[0]:
                starty = image.shape[0] - sizey

            # Convert to integers
            startx = int(startx)
            starty = int(starty)

            # Now retreive the box
            box = image[starty:starty + sizey, startx:startx + sizex]

            # If boxes had to be shifted, we have to calculate a new 'center' of the nodule in the box
            new_center = [int(sizey / 2 - ((starty + sizey / 2) - origin[0])),
                          int(sizex / 2 - ((startx + sizex / 2) - origin[1]))]

            return box, new_center

        # first scale the z axis in half
        if z_overwrite: sizez = z_overwrite
        else: sizez = int(size/2)

        # Make the starting point = center-size unless near the edge then make it 0
        startx = max(origin[2] - size/2, 0)
        starty = max(origin[1] - size/2, 0)
        startz = max(origin[0] - sizez/2, 0)

        # If near the far edge, make it fit inside the image
        if (startx + size) > image.shape[2]:
            startx = image.shape[2] - size
        if (starty + size) > image.shape[1]:
            starty = image.shape[1] - size
        if (startz + sizez) > image.shape[0]:
            startz = image.shape[0] - sizez

        # Convert to integers
        startx = int(startx)
        starty = int(starty)
        startz = int(startz)

        # Now retreive the box
        box = image[startz:startz + sizez, starty:starty + size, startx:startx + size]

        # If boxes had to be shifted, we have to calculate a new 'center' of the nodule in the box
        new_center = [int(sizez/ 2 - ((startz + sizez/ 2) - origin[0])),
                      int(size / 2 - ((starty + size / 2) - origin[1])),
                      int(size / 2 - ((startx + size / 2) - origin[2]))]

        # display if wanted
        if display: print(image.shape, startz, starty, startx, box.shape, 'New Center:', new_center)

        return box, new_center


    def generate_DRR(self, volume_data, vert_scale=1.0, horiz_scale=1.0, source_distance=39.37, cone=True):

        """
        Create a radiographic projection of an input volume.
        :param volume_data: Input volume
        :param vert_scale: Row pixel spacing, has effect of vertically scaling input volume
        :param horiz_scale: Row pixel spacing, has effect of horizontally scaling input volume
        :param source_distance: Distance to xray source in inches. Detector placed at 0
        :param cone: whether cone beam or parallel beam
        :return:
        """

        # Retreive shapes
        Z, Y, X = volume_data.shape

        # Create the astra 3D volume geometry
        vol_geom = astra.create_vol_geom(Y, X, Z)

        # Angles is an array with all the angles that will be projected
        source = 25.4 * source_distance
        angles = np.linspace(0, 2 * np.pi, 48, False)  # Don't use angles for now, just affine warps

        # Create a 3D beam geometry
        if cone: proj_geom = astra.create_proj_geom('cone', vert_scale, horiz_scale, Z, X, 0, source, 0)
        else: proj_geom = astra.create_proj_geom('parallel3d', vert_scale, horiz_scale, Z, X, 0)

        # Create the projections
        proj_id, proj_data = astra.create_sino3d_gpu(volume_data, proj_geom, vol_geom)

        # Garbage collection
        proj_data = np.flip(np.moveaxis(proj_data, 0, 1), 1)
        astra.data3d.delete(proj_id)

        # Return the projection
        return np.squeeze(proj_data)


    def crop_data(self, data, origin, boundaries):

        """
        Crops a given array
        :param data: Input volume, numpy array
        :param origin: Center of the crop, tuple
        :param boundaries: z, y, x width of the crop, tuple
        :return: The cropped volume
        """

        # Convert to numpy
        data = np.squeeze(data)
        og = np.asarray(origin)
        bn = np.asarray(boundaries)

        # Is this 3D or 2D
        if data.ndim >= 3:
            is_3d = True
        else:
            is_3d = False

        # Now perform the cut
        if is_3d:
            return data[og[0] - bn[0]:og[0] + bn[0], og[1] - bn[1]:og[1] + bn[1], og[2] - bn[2]:og[2] + bn[2], ...], og, bn
        else:
            return data[og[0] - bn[0]:og[0] + bn[0], og[1] - bn[1]:og[1] + bn[1]], og, bn


    def is_odd(self, num):

        """
        Check if num is odd
        :param num:
        :return:
        """
        return num & 0x1


    def pad_resize(self, input, out_size, pad_value='constant'):

        """
        Pads an input array to the output size with pad_value.
        :param input: The input array
        :param out_size: The desired output size [z, y, x], Must be bigger than input array
        :param pad_mode: string, Pad with this mode, 'constant', edge, max, mean, etc
        :return: the new padded array
        """

        # Convert arrays
        input = np.squeeze(input)
        out_size = np.asarray(out_size)

        # Is this 3D or 2D
        if input.ndim >= 3:
            is_3d = True
        else:
            is_3d = False

        if is_3d:

            # Calculate how much to pad on each axis
            xpad = xpad2 = (out_size[2] - input.shape[2]) // 2
            ypad = ypad2 = (out_size[1] - input.shape[1]) // 2
            zpad = zpad2 = (out_size[0] - input.shape[0]) // 2

            # Handle odd numbers: 5//2 = 2
            if self.is_odd(out_size[0]) != self.is_odd(input.shape[0]):
                zpad2 = 1 + (out_size[0] - input.shape[0]) // 2

            if self.is_odd(out_size[1]) != self.is_odd(input.shape[1]):
                ypad2 = 1 + (out_size[1] - input.shape[1]) // 2

            if self.is_odd(out_size[2]) != self.is_odd(input.shape[2]):
                xpad2 = 1 + (out_size[2] - input.shape[2]) // 2

            # Perform the pad
            return np.pad(input, ((zpad, zpad2), (ypad, ypad2), (xpad, xpad2)), pad_value)

        else:

            # Calculate how much to pad on each axis
            xpad = xpad2 = (out_size[1] - input.shape[1]) // 2
            ypad = ypad2 = (out_size[0] - input.shape[0]) // 2

            # Handle odd numbers: 5//2 = 2
            if self.is_odd(out_size[0]) != self.is_odd(input.shape[0]):
                ypad2 = 1 + (out_size[0] - input.shape[0]) // 2

            if self.is_odd(out_size[1]) != self.is_odd(input.shape[1]):
                xpad2 = 1 + (out_size[1] - input.shape[1]) // 2

            # Perform the pad
            return np.pad(input, ((ypad, ypad2), (xpad, xpad2)), pad_value)

    """
         Utility functions: Random tools for help
    """


    def split_dict_equally(self, input_dict, chunks=2):

        """
        Function that splits a dictionary into equal chunks
        :param input_dict: input dictionary
        :param chunks: number of chunks
        :return: A list of dictionaries
        """

        # Create empty dictionaries list first
        return_list = [dict() for idx in range(chunks)]
        idx = 0

        # Save into each listed dictionary one at a time
        for k, v in input_dict.items():
            return_list[idx][k] = v

            # Loop back to beginning
            if idx < chunks - 1:
                idx += 1
            else:
                idx = 0

        return return_list


    def largest_blob(self, img):
        """
        This finds the biggest blob in a 2D or 3D volume and returns the center of the blob
        :param img: the binary input volume
        :return: img if no labels, labels if there is. and cn: an array with the center locations [z, y, x]
        """

        # Only work if a mask actually exists
        if np.max(img) > 0:

            # Labels all the blobs of connected pixels
            labels = morphology.label(img)

            # Counts the number of ocurences of each value, then returns the 2nd biggest blob (0 occurs the most)
            N = np.bincount(labels.flatten())[1:].argmax() + 1

            # Mark the blob
            labels = (labels == N)

            # Find the center of mass
            cn = scipy.measurements.center_of_mass(labels)

            if labels.ndim == 3: cn = [int(cn[0]), int(cn[1]), int(cn[2])]
            else: cn = [int(cn[0]), int(cn[1])]

            # Return the parts of the label equal to the 2nd biggest blob
            return labels, np.asarray(cn, np.int32)

        else:
            return img, img.shape//2


    def all_blobs(self, img):

        """
        This finds all of the blobs in a 3D volume and returns them along with the center locations
        :param img: the binary input volume
        :return: label binary volume with each blob labeled 1 - n, centers, blob_sizes, blob_count
        """

        # Only work if a mask actually exists
        if np.max(img) > 0:

            # Labels all the blobs of connected pixels
            labels = morphology.label(img)

            # Counts the number of ocurences of each value minus 0 which is always the most
            blob_sizes = np.bincount(labels.flatten())[1:]

            # Gets the amount of blobs
            blob_count = blob_sizes.shape[0]

            # Define array of blobs and centers
            centers = []

            # Loop through and get all the labels
            for z in range(blob_count):

                # Mark the blob for this pixel value (+1 from array index)
                label_temp = (labels == (z+1))

                # Find the center of mass
                cn = scipy.measurements.center_of_mass(label_temp)

                # Generate center based on 2D or 3D
                if labels.ndim == 3: cn = [int(cn[0]), int(cn[1]), int(cn[2])]
                else: cn = [int(cn[0]), int(cn[1])]

                # Append
                centers.append(cn)

                # Delete
                del label_temp, cn

            # Return the array of labels and centers
            return labels, centers, blob_sizes, blob_count

        else:
            return img


    def normalize(self, input, crop=False, crop_val=0.5):
        """
        Normalizes the given np array
        :param input:
        :param crop: whether to crop the values
        :param crop_val: the percentage to crop the image
        :return:
        """

        if crop:

            ## CLIP top and bottom x values and scale rest of slice accordingly
            b, t = np.percentile(input, (crop_val, 100-crop_val))
            slice = np.clip(input, b, t)
            if np.std(slice) == 0:
                return slice
            else:
                return (slice - np.mean(slice)) / np.std(slice)

        return (input - np.mean(input)) / np.std(input)


    def normalize_MRI_histogram(self, image, return_values=False, center_type='mean'):

        """
        Uses histogram normalization to normalize MRI data by removing 0 values
        :param image: input volume numpy array
        :param return_values: Whether to return the mean, std and mode values as well
        :param center_type: What to center the data with, 'mean' or 'mode'
        :return:
        """

        # First calculate the most commonly occuring values in the volume
        occurences, values = np.histogram(image, bins=500)

        # Remove 0 values (AIR) which always win
        occurences, values = occurences[1:], values[1:]

        # The mode is the value array at the index of highest occurence
        mode = values[np.argmax(occurences)]

        # Make dummy no zero image array to calculate STD
        dummy, img_temp = [], np.copy(image).flatten()
        for z in range(len(img_temp)):
            if img_temp[z] > 5: dummy.append(img_temp[z])

        # Mean/std is calculated from nonzero values only
        dummy = np.asarray(dummy, np.float32)
        std, mean = np.std(dummy), np.mean(dummy)

        # Now divide the image by the modified STD
        if center_type=='mode': image = image.astype(np.float32) - mode
        else: image = image.astype(np.float32) - mean
        image /= std

        # Return values or just volume
        if return_values: return image, mean, std, mode
        else: return image


    def adaptive_normalization(self, tensor, dim3d=False):

        """
        Contrast localized adaptive histogram normalization
        :param tensor: ndarray, 2d or 3d
        :param dim3d: 2D or 3d. If 2d use Scikit, if 3D use the MCLAHe implementation
        :return: normalized image
        """

        if dim3d: return mc.mclahe(tensor)
        else: return hist.equalize_adapthist(tensor)


    def normalize_dynamic_range (self, volume, low=0, high=255, mask=None, dtype=np.uint8, dim3d=True):
        """
        Takes an input and stretches the pixel values over the given dynamic range
        :param volume: input volume, numpy array
        :param low: low range
        :param high: high range
        :param mask: If mask defined, only normalize the values in mask
        :param dtype: output data type desired
        :param dim3d: True for 3d volumes
        :return:
        """

        # Normalize the volume into 0-255 range for .gif file
        volume_norm = np.zeros_like(volume, dtype=dtype)

        # For volumes
        if dim3d:
            for z in range(volume.shape[0]):
                volume_norm[z] = cv2.normalize(volume[z], dst=volume_norm[z], alpha=low, beta=high, mask=mask, norm_type=cv2.NORM_MINMAX)

        # For 2d images
        else:
            volume_norm = cv2.normalize(volume, dst=volume_norm, alpha=low, beta=high, mask=mask, norm_type=cv2.NORM_MINMAX)

        return volume_norm


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


    def gray2rgb(self, img, RGB=True):

        """
        Use open CV to convert HxW grayscale image to HxWxC RGB image
        :param img: The input image as numpy array
        :param RGB: True = RGB, False = BGR
        :return: the converted image
        """

        if RGB: return cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        else: return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


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

        # If they want to return the folder list, do that
        if extension == '*': return glob.glob(path + '*')

        # If we're including subfolders
        if include_subfolders: extension = ('**/*.%s' % extension)

        # Otherwise just search this folder
        else: extension = ('*.%s' %extension)

        # Join the pathnames
        path = os.path.join(path, extension)

        # Return the list of filenames
        return glob.glob(path, recursive=include_subfolders)


    def window_image(self, volume, level, width):
        """
        Windows the image. For images that come from CT scans
        :param volume: ndarray, Source image
        :param level: the center of the window
        :param width: how wide the window is +-
        :return: the windowed image
        """

        # Set the min and max HU
        minimum = int(level - width)
        maximum = int(level + width)

        # Fit to max
        volume[volume > maximum] = maximum

        # fit to min
        volume[volume < minimum] = minimum

        # Return the windowed image
        return volume


    def normalize_dictionary(self, data, dims, channels=0, crop=False, range=0.1):
        """
        Crops all the data in a 2D input dictionary, assuming its under the index ['data']
        :param data: input dictionary
        :param dims: dimensions of the image
        :param crop: crop the normalization or not
        :param range: crop range
        :return: 
        """

        # Initialize normalization images array
        if channels>1: normz = np.zeros(shape=(len(data), dims, dims, channels), dtype=np.float32)
        else: normz = np.zeros(shape=(len(data), dims, dims), dtype=np.float32)

        # Normalize all the images. First retreive the images
        for key, dict in data.items(): normz[key, :, :] = dict['data']

        # Now normalize the whole batch
        print('Batch Norm: %s , Batch STD: %s' % (np.mean(normz), np.std(normz)))
        normz = self.normalize(normz, crop, range)

        # Return the normalized images to the dictionary
        for key, dict in data.items(): dict['data'] = normz[key]

        return data


    def save_tfrecords(self, data, xvals, test_size=50, file_root='data/Data'):

        """
        Saves the dictionary given to a protocol buffer in tfecords format
        :param data: Input dictionary
        :param xvals: number of even files to save. If ==2 save 'test_size' test set and a train set
        :param test_size: if xvals == 2 then this is the number of examples to put in the test set
        :param file_root: The first part of the filename to save. This includes the directory
        :return: not a damn thing
        """

        # If only one file, just go head and save
        if xvals ==1:

            # Open the file writer
            writer = tf.python_io.TFRecordWriter((file_root + '.tfrecords'))

            # Loop through each example and append the protobuf with the specified features
            for key, values in data.items():

                # Serialize to string
                example = tf.train.Example(features=tf.train.Features(feature=self.create_feature_dict(values, key)))

                # Save this index as a serialized string in the protobuf
                writer.write(example.SerializeToString())

            # Close the file after writing
            writer.close()

            return

        # generate x number of writers depending on the cross validations
        writer = []

        # Open the file writers
        for z in range(xvals):

            # Define writer name
            if xvals == 2:
                if z == 0: filename = (file_root + '_Test' + '.tfrecords')
                else: filename = (file_root + '_Train' + '.tfrecords')

            # For one file
            elif xvals == 1: filename = (file_root + '.tfrecords')

            else: filename = (file_root + str(z) + '.tfrecords')

            writer.append(tf.python_io.TFRecordWriter(filename))

        # Loop through each example and append the protobuf with the specified features
        if xvals == 2:

            # First for the special case
            tests = 0

            for key, values in data.items():

                # Serialize to string
                example = tf.train.Example(features=tf.train.Features(feature=self.create_feature_dict(values, key)))

                # Save this index as a serialized string in the protobuf
                if tests < test_size: writer[0].write(example.SerializeToString())
                else: writer[1].write(example.SerializeToString())
                tests += 1

        else:

            # Now for every other case
            z = 0

            for key, values in data.items():
                # Serialize to string
                example = tf.train.Example(features=tf.train.Features(feature=self.create_feature_dict(values, key)))

                # Save this index as a serialized string in the protobuf
                writer[(z % xvals)].write(example.SerializeToString())
                z += 1

        # Close the file after writing
        for y in range(xvals): writer[y].close()


    def save_segregated_tfrecords(self, xvals=2, data={}, ID_key='ID', file_root='data/data'):

        """
        Sequestered each patient by id key into unique tfrecords and saves
        :param xvals: Number of different files to save
        :param data: The input dictionary
        :param ID_key: The key of the input dic that segregates the patient (i.e. MRN or Accno)
        :param file_root: The root of the file to save
        :return:
        """

        # Variables to track
        dictonaries = [dict() for _ in range(xvals)]
        ID_track = [list() for _ in range(xvals)]
        index = 0

        for idx, dic in data.items():

            # Tracker to see if already added plus dict to use tracker
            added = 0
            target = index % xvals

            # If this ID already exists in a dictionary, save to that dictionary
            for z in range(xvals):

                if dic[ID_key] in ID_track[z]:
                    # Success, use this one
                    dictonaries[z][index] = dic
                    added = 1
                    break

            # Then continue
            if added == 1:
                index += 1
                continue

            # Failed to find a preexisting, save in the target dictionary
            dictonaries[target][index] = dic
            ID_track[target].append(dic[ID_key])
            index += 1

        # Now save the tfrecords
        for z in range(xvals):
            file = file_root + '_' + str(z)
            print('Saving segregated tfrecords... %s in %s' % (len(dictonaries[z]), file))
            self.save_tfrecords(dictonaries[z], 1, file_root=file)


    def convert_xray_negative(self, image):

        """
        Converts an xray into into it's negative
        :param image: input 2D xray as ndarray
        :return: the negative
        """

        # Get the variables needed for the math
        smallest, largest = np.amin(image), np.amax(image)
        average = (largest - smallest) / 2

        # Do the image transform
        return ((image - average) * -1) + average


    """
         Tool functions: Most of these are hidden
    """

    def _derive_key(self, password: bytes, salt: bytes, iterations: int = 100_000) -> bytes:
        """Derive a secret key from a given password and salt"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(), length=32, salt=salt,
            iterations=iterations, backend=default_backend())
        return b64e(kdf.derive(password))

    def password_encrypt(self, message: bytes, password: str, iterations: int = 100_000, return_string=False) -> bytes:
        salt = secrets.token_bytes(16)
        key = self._derive_key(password.encode(), salt, iterations)
        message = message.encode('utf-8')
        ret = b64e(
            b'%b%b%b' % (
                salt,
                iterations.to_bytes(4, 'big'),
                b64d(Fernet(key).encrypt(message)),
            )
        )
        if return_string:
            return ret.decode('utf-8')
        else:
            return ret

    def password_decrypt(self, token: bytes, password: str, return_string=False) -> bytes:
        decoded = b64d(token)
        salt, iter, token = decoded[:16], decoded[16:20], b64e(decoded[20:])
        iterations = int.from_bytes(iter, 'big')
        key = self._derive_key(password.encode(), salt, iterations)
        ret = Fernet(key).decrypt(token)
        if return_string:
            return ret.decode('utf-8')
        else:
            return ret


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
            if 'data' in key:
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


    def return_nonzero_pixel_ratio(sef, input, depth=3, dim_3d=False):
        """
        Returns the proportion of pixels that are zero
        :param input: numpy array of the image
        :param depth: number of color channes
        :param dim_3d: whether its 3 dimensional
        :return: fraction of pixels that are 0
        """

        # get total pixel count
        if dim_3d: tot_pixels = input.shape[0] * input.shape[1] * input.shape[2]
        else: tot_pixels = input.shape[0] * input.shape[1]

        # Convert image to grayscale if not already
        if depth==3: input = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)

        # Count the fraction of pixels that arent 0
        non_zero = np.sum(input.astype(np.bool)) / tot_pixels

        return non_zero


    def convert_grayscale(self, image):
        """
        Converts an image to grayscale
        :param image: input image numpy array
        :return: gray image
        """

        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    def calculate_image_mean_dict(self, data, index):
        """
        Prints out the mean and STD of an image set in a dictionary
        :param data: the input dictionary
        :param index: The index name of the images you wish to evaluate
        :return:
        """

        # Calculate dict norm
        mean, std, num = 0, 0, 0
        for key, dict in data.items():
            mean += np.mean(dict[index])
            std += np.std(dict[index])
            num += 1

        print('Mean %s, Std: %s' % ((mean / num), (std / num)))


    def sort_DICOMS_PE(self, ndimage, display=False, path=None):

        """
        Sorts through messy DICOM folders and retreives axial original acquisitions of PE studies
        :param ndimage: The loaded DICOM volume
        :param display: Whether to display debugging text
        :param path: Path to the original folder, for debug text
        :return: real: the desired actual images
        """

        # First define some values in the DICOM header that indicate files we want to skip
        desired_ImageTypes = ['PRIMARY', 'AXIAL', 'ORIGINAL']
        skipped_ImageTypes = ['MIP', 'SECONDARY', 'LOCALIZER', 'REFORMATTED']
        skipped_Descriptions = ['BONE', 'LUNG', 'WITHOUT', 'TRACKER', 'SMART PREP', 'MONITORING', 'LOCATOR']
        skipped_Studies = ['ABDOMEN', 'PELVIS']

        # Try saving only the original primary axial series. Use an ID to make sure to save only one series. Skip MIPS
        real, this_ID, this_series, test_ID = [], None, None, None
        for z in range(len(ndimage)):

            # Try statement to skip non DICOM slices
            try:

                # Skip the things we want to skip
                if ('ORIGINAL' not in ndimage[z].ImageType) or ('PRIMARY' not in ndimage[z].ImageType) or ('AXIAL' not in ndimage[z].ImageType): continue
                if any(text in ndimage[z].ImageType for text in skipped_ImageTypes): continue
                if any(text in ndimage[z].SeriesDescription.upper() for text in skipped_Descriptions): continue
                if any(text in ndimage[z].StudyDescription.upper() for text in skipped_Studies): continue
                if len(ndimage[z].SeriesDescription) == 0: continue

                # Make Sure identification matches or is null then add to the volume
                if (this_ID == None or this_ID == ndimage[z].ImageType) and (this_series == None or this_series == ndimage[z].SeriesDescription):
                    this_ID = ndimage[z].ImageType
                    this_series = ndimage[z].SeriesDescription
                    real.append(ndimage[z])
                    if not test_ID:
                        if display: print ('Win on try 1: ', this_series, end = '')
                        test_ID = True

            except: continue

        # No original series must exist if this following try statement fails, load a derived primary axial instead
        try: print (real[0].ImageType, end='')
        except:

            # Save only the original axial, use ID to save one series only
            del real
            real, this_ID, this_series, test_ID = [], None, None, None
            if display: print ('First attempt to load DICOM failed, trying with less strict requirements')
            for z in range(len(ndimage)):

                # Try statement to skip non DICOM slices
                try:

                    # Less strict on image type
                    if ('PRIMARY' not in ndimage[z].ImageType) or ('AXIAL' not in ndimage[z].ImageType): continue
                    if any(text in ndimage[z].ImageType for text in skipped_ImageTypes): continue
                    if any(text in ndimage[z].SeriesDescription.upper() for text in skipped_Descriptions): continue
                    if any(text in ndimage[z].StudyDescription.upper() for text in skipped_Studies): continue
                    if len(ndimage[z].SeriesDescription) == 0: continue

                    # Make Sure identification matches or is null then add to the volume
                    if (this_ID == None or this_ID == ndimage[z].ImageType) and (this_series == None or this_series == ndimage[z].SeriesDescription):
                        this_ID = ndimage[z].ImageType
                        this_series = ndimage[z].SeriesDescription
                        real.append(ndimage[z])
                        if not test_ID:
                            if display: print('Win on try 2: ', this_series, end='')
                            test_ID = True

                except: continue

        # At this point, we're up shit's creek
        try: print (real[0].ImageType)
        except:

            # Save only the original axial, use ID to save one series only
            del real
            real, this_ID, this_series, test_ID = [], None, None, None
            skipped_Descriptions_shit = ['BONE', 'WITHOUT', 'TRACKER', 'SMART PREP', 'MONITORING']
            if display: print('2nd attempt failed... up shits creek now...')
            for z in range(len(ndimage)):

                # Try statement to skip non DICOM slices
                try:

                    # Skip less image types, load lung windows if available, don't skip empty descriptions
                    if ('ORIGINAL' not in ndimage[z].ImageType) or ('PRIMARY' not in ndimage[z].ImageType) or ('AXIAL' not in ndimage[z].ImageType): continue
                    if any(text in ndimage[z].ImageType for text in skipped_ImageTypes): continue
                    if any(text in ndimage[z].SeriesDescription.upper() for text in skipped_Descriptions_shit): continue
                    if any(text in ndimage[z].StudyDescription for text in skipped_Studies): continue

                    # Make Sure identification matches or is null then add to the volume
                    if (this_ID == None or this_ID == ndimage[z].ImageType) and (this_series == None or this_series == ndimage[z].SeriesDescription):
                        this_ID = ndimage[z].ImageType
                        this_series = ndimage[z].SeriesDescription
                        real.append(ndimage[z])
                        if not test_ID:
                            if display: print('Win on try 3: ', this_series, end='')
                            test_ID = True

                except: continue

        return real


    def sort_DICOMS_Lung(self, ndimage, display=False, path=None):

        """
        Sorts through messy DICOM folders and retreives axial original acquisitions of lung CT scans
        :param ndimage: The loaded DICOM volume
        :param display: Whether to display debugging text
        :param path: Path to the original folder, for debug text
        :return: real: the desired actual images
        """

        """
        We want SERIES Description Axial lung thin
        Then bone, thin slice
        Then any thin slice < 1
        Then any thin slice < 2
        Skip derived secondary reformatted
        """

        # Things we want
        desired_Description = ['AXIAL', 'LUNG', 'THIN']

        # First define some values in the DICOM header that indicate files we want to skip
        skipped_ImageTypes = ['MIP', 'SECONDARY', 'LOCALIZER', 'REFORMATTED', 'DERIVED']
        skipped_Descriptions = ['WITH', 'TRACKER', 'SMART PREP', 'MONITORING', 'LOCATOR']
        skipped_Studies = ['ABDOMEN', 'PELVIS']

        # Placeholders
        real, this_ID, this_series, test_ID = [], None, None, None

        # First try finding our ideal study, axial lung thin
        for z in range(len(ndimage)):

            # Try statement to skip non DICOM slices
            try:

                # Skip the things we want to skip
                if ('ORIGINAL' not in ndimage[z].ImageType) or ('PRIMARY' not in ndimage[z].ImageType) or ('AXIAL' not in ndimage[z].ImageType): continue
                if any(text not in ndimage[z].SeriesDescription.upper() for text in desired_Description): continue

                if any(text in ndimage[z].ImageType for text in skipped_ImageTypes): continue
                if any(text in ndimage[z].SeriesDescription.upper() for text in skipped_Descriptions): continue
                if any(text in ndimage[z].StudyDescription.upper() for text in skipped_Studies): continue
                if len(ndimage[z].SeriesDescription) == 0: continue

                # Make Sure identification matches or is null then add to the volume
                if (this_ID == None or this_ID == ndimage[z].ImageType) and (
                        this_series == None or this_series == ndimage[z].SeriesDescription):
                    this_ID = ndimage[z].ImageType
                    this_series = ndimage[z].SeriesDescription
                    real.append(ndimage[z])
                    if not test_ID:
                        if display: print('Win on try 1: ', ndimage[z].SeriesDescription, ndimage[z].SliceThickness, end='')
                        test_ID = True

            except:
                continue

        # No original series must exist if this following try statement fails, load a body thin slice
        try: print (real[0].ImageType, end='')
        except:

            # Save only the original axial, use ID to save one series only
            del real
            real, this_ID, this_series, test_ID = [], None, None, None
            if display: print ('No axial thin lungs, trying any <1.0mm')
            for z in range(len(ndimage)):

                # Try statement to skip non DICOM slices
                try:

                    # Now look for thin bodies
                    if ('ORIGINAL' not in ndimage[z].ImageType) or ('PRIMARY' not in ndimage[z].ImageType) or ('AXIAL' not in ndimage[z].ImageType): continue
                    if (float(ndimage[z].SliceThickness) > 0.99): continue

                    if any(text in ndimage[z].ImageType for text in skipped_ImageTypes): continue
                    if any(text in ndimage[z].SeriesDescription.upper() for text in skipped_Descriptions): continue
                    if any(text in ndimage[z].StudyDescription.upper() for text in skipped_Studies): continue
                    if len(ndimage[z].SeriesDescription) == 0: continue

                    # Make Sure identification matches or is null then add to the volume
                    if (this_ID == None or this_ID == ndimage[z].ImageType) and (this_series == None or this_series == ndimage[z].SeriesDescription):
                        this_ID = ndimage[z].ImageType
                        this_series = ndimage[z].SeriesDescription
                        real.append(ndimage[z])
                        if not test_ID:
                            if display: print('Win on try 2: ', ndimage[z].SeriesDescription, ndimage[z].SliceThickness, end='')
                            test_ID = True

                except: continue

        # Try for thicker slices
        try: print (real[0].ImageType)
        except:

            # Save only the original axial, use ID to save one series only
            del real
            real, this_ID, this_series, test_ID = [], None, None, None
            if display: print('2nd attempt failed... trying any < 4mm')
            for z in range(len(ndimage)):

                # Try statement to skip non DICOM slices
                try:

                    # Now look for Any somewhat thin
                    if ('ORIGINAL' not in ndimage[z].ImageType) or ('PRIMARY' not in ndimage[z].ImageType) or ('AXIAL' not in ndimage[z].ImageType): continue
                    if (float(ndimage[z].SliceThickness) > 4.99): continue

                    if any(text in ndimage[z].ImageType for text in skipped_ImageTypes): continue
                    if any(text in ndimage[z].SeriesDescription.upper() for text in skipped_Descriptions): continue
                    if any(text in ndimage[z].StudyDescription.upper() for text in skipped_Studies): continue
                    if len(ndimage[z].SeriesDescription) == 0: continue

                    # Make Sure identification matches or is null then add to the volume
                    if (this_ID == None or this_ID == ndimage[z].ImageType) and (this_series == None or this_series == ndimage[z].SeriesDescription):
                        this_ID = ndimage[z].ImageType
                        this_series = ndimage[z].SeriesDescription
                        real.append(ndimage[z])
                        if not test_ID:
                            if display: print('Win on try 3: ', ndimage[z].SeriesDescription, ndimage[z].SliceThickness, end='')
                            test_ID = True

                except: continue

        # Now we're just fucked
        try: print(real[0].ImageType)
        except:

            # Save only the original axial, use ID to save one series only
            del real
            real, this_ID, this_series, test_ID = [], None, None, None
            if display: print('Basically We will take anything now')
            for z in range(len(ndimage)):

                # Try statement to skip non DICOM slices
                try:

                    # Now look for Any somewhat thin
                    if ('ORIGINAL' not in ndimage[z].ImageType) or ('PRIMARY' not in ndimage[z].ImageType) or ('AXIAL' not in ndimage[z].ImageType): continue
                    if len(ndimage[z].SeriesDescription) == 0: continue

                    # Make Sure identification matches or is null then add to the volume
                    if (this_ID == None or this_ID == ndimage[z].ImageType) and (
                            this_series == None or this_series == ndimage[z].SeriesDescription):
                        this_ID = ndimage[z].ImageType
                        this_series = ndimage[z].SeriesDescription
                        real.append(ndimage[z])
                        if not test_ID:
                            if display: print('Win on try 4... : ', ndimage[z].SeriesDescription,
                                              ndimage[z].SliceThickness, ndimage[z].ImageType, end='')
                            test_ID = True

                except:
                    continue

        return real


    def read_dcm_uncompressed(self, s):

        """
        Method to load single dicom pixel array

        """
        image = s.pixel_array

        if hasattr(s, 'Rows') and hasattr(s, 'Columns'):
            if image.shape[0] != s.Rows:
                image = image.reshape(s.Rows, s.Columns)

        return image


    def read_dcm_compressed(self, fname):
        """
        Method to load single compressed dicom file with mudicom

        """
        mu = mudicom.load(fname)
        img = mu.image.numpy

        header = mu.read()
        header = dict([(h.name, h.value) for h in header])

        if 'Rows' in header and 'Columns' in header:
            if img.shape[0] != header['Rows']:
                img = img.reshape(int(header['Rows']), int(header['Columns']))

        return img.astype('int16')


    def compress_bits(self, vol):
        """
        Method to ensure image is at most signed 16-bit integer

        """
        m = np.max(np.abs(vol))
        dtype = np.equal(np.mod(vol, 1), 0).all()
        dtype = 'int' if dtype else 'float'

        # --- Convert integers
        if dtype == 'int':
            if m < 255:
                return vol.astype('uint8')
            elif m < 32768:
                return vol.astype('int16')
            else:
                vol = vol.astype('float64') * (32768 / m)
                return vol.astype('int16')

        # --- Convert floats
        if dtype == 'float':
            return vol.astype('float16')


    def sort_dcm(self, slices, fnames, verbose=False):
        """
        Method to sort DICOM objects by ImagePositionPatient
        """
        # --- Sort by instance
        slices_instance = [s for s in slices if hasattr(s, 'InstanceNumber')]
        if len(slices_instance) > int(len(slices) / 2):
            slices = slices_instance
            inns = np.array([int(s.InstanceNumber) for s in slices])
            indices = np.argsort(inns)
            slices = [slices[i] for i in indices]
            fnames = [fnames[i] for i in indices]

        # --- Determine orientation
        slices = [s for s in slices if hasattr(s, 'ImagePositionPatient')]
        ipps = np.array([s.ImagePositionPatient for s in slices])
        assert ipps.shape[1] == 3
        diff = []
        for ax in range(3):
            diff.append(np.max(ipps[:, ax]) - np.min(ipps[:, ax]))
        diff_max = np.argmax(diff)
        orient_key = {
            0: 'SAG',
            1: 'COR',
            2: 'AXI'}

        indices = np.argsort(ipps[:, diff_max].ravel())

        # --- Determine interleave pattern
        ipps = ipps[:, diff_max]
        u = np.unique(ipps)
        shape = [1, ipps.size]  # output shape in [channels x Z]

        four_d = ipps.size > u.size
        if four_d:
            if ipps.size % u.size == 0:
                n = int(ipps.size / u.size)
                indices = np.concatenate([np.sort(indices[i * n:(i + 1) * n]) for i in range(u.size)])
                indices = np.concatenate([indices[i::n] for i in range(n)])
                shape = [n, u.size]  # output shape in [channels x Z]
            else:
                if verbose:
                    print('Warning: 4D volume seems to be missing data within a channel (%s)' % os.path.dirname(fnames[0]))

        slices = [slices[i] for i in indices]
        fnames = [fnames[i] for i in indices]

        # --- Determine slice thickness
        if ipps.shape[0] > 1:
            st = np.sort(ipps.ravel())
            st = st[1:] - st[:-1]
            st = st[st > 0]
            st = np.sort(st)
            st = st[int(st.size / 2)]

        return slices, fnames, orient_key[diff_max], st, shape, four_d