"""

SOD display is a class containing various methods to display numpy or tensorflow data

"""

import cv2, imageio
import os
from moviepy.editor import ImageSequenceClip

from SODLoader import SODLoader

import numpy as np
import matplotlib.pyplot as plt

class SOD_Display(SODLoader):

    def __init__(self):

        """
        Initializes the class handler object
        """
        pass


    """
    Images
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

        # Set up the figure object
        fig = plt.figure()
        ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])

        # The rest is standard matplotlib fare
        plt.set_cmap(cmap)  # Print in greyscale
        ax.imshow(nda)

        if title: plt.title(title)
        if plot: plt.show()


    def display_mosaic(self, vol, plot=False, fig=None, title=None, size=[10, 10], vmin=None, vmax=None,
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

        if type(vol) is not np.ndarray: vol = np.asarray(vol)

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
            this_im = np.hstack(vol[int(len(vol) / sq) * i:int(len(vol) / sq) * (i + 1)])
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
            imax = ax.matshow(im.T, vmin=vmin, vmax=vmax, cmap='gray', **kwargs)
        else:
            imax = plt.imshow(np.rot90(im), interpolation='nearest')
            cbar = False
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

        # If we are displaying:
        if plot: plt.show()

        return returns


    def display_volume(self, volume, plot=False, cmap='gray'):

        """
        Displays a scrollable 3D volume slice by slice
        :param volume: input numpy array or array
        :param plot: Display now or not
        :param cmap: color map
        :return:
        """

        if type(volume) is not np.ndarray: volume = np.asarray(volume)

        #self._remove_keymap_conflicts({'j', 'k'})
        fig, ax = plt.subplots()
        ax.volume = volume
        ax.index = volume.shape[0] // 2
        ax.imshow(volume[ax.index], cmap=cmap)
        fig.canvas.mpl_connect('scroll_event', self._process_key)
        if plot: plt.show()


    def display_stack(self, stack, plot=False, rows=6, cols=6, start_with=10, show_every=3):
        """
        Displays a mosaic of images with skipped slices in between
        :param stack:
        :param rows:
        :param cols:
        :param start_with:
        :param show_every:
        :return:
        """

        if type(stack) is not np.ndarray: stack = np.asarray(stack)

        fig, ax = plt.subplots(rows, cols, figsize=[12, 12])
        for i in range(rows * cols):
            ind = start_with + i * show_every
            ax[int(i / rows), int(i % rows)].set_title('slice %d' % ind)
            ax[int(i / rows), int(i % rows)].imshow(stack[ind], cmap='gray')
            ax[int(i / rows), int(i % rows)].axis('off')
        if plot: plt.show()



    def draw_box_cv(self, img, boxes, text):

        """
        Draw box function
        :param img: ndarray input image
        :param boxes: ndarray all of the boxes in this image
        :param text: overlay text
        :return:
        """

        # Add a grayscale value to the image TODO: Was in color
        img = img + np.array([116.779])
        boxes = boxes.astype(np.int64)

        # Normalize image
        img = np.array(img * 255 / np.max(img), np.uint8)

        # Loop through boxes we need to draw. If only one box, don't loop
        if boxes.ndim==1:

            # Retreive coordinates and a random color
            xmin, ymin, xmax, ymax = boxes[0], boxes[1], boxes[0] + boxes[2], boxes[1] + boxes[3]
            color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))

            # Generate a rectangle
            cv2.rectangle(img, pt1=(xmin, ymin), pt2=(xmax, ymax), color=color, thickness=2)

        else:
            for box in boxes:
                # Retreive coordinates and a random color
                xmin, ymin, xmax, ymax = box[0], box[1], box[0] + box[2], box[1] + box[3]
                color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))

                # Generate a rectangle
                cv2.rectangle(img, pt1=(xmin, ymin), pt2=(xmax, ymax), color=color, thickness=2)

        # Generate overlay text
        text = str(text)
        cv2.putText(img, text=text, org=((img.shape[1]) // 2, (img.shape[0]) // 2), fontFace=3, fontScale=1, color=(255, 0, 0))

        #
        img = img[:, :, -1::-1]

        return img


    """
    Plots
    """


    def display_histogram(self, input_image, x_label, y_label ='Frequency', display=False, bins=50, color='c'):

        """
        Plots a histogram summary
        :param input_image: The input volume or image, numpy array
        :param x_label: Label for x axis
        :param y_label: Y axis label
        :param display: whether to show the plot now
        :param bins: How many different bins to plot
        :param color: Color
        :return:
        """

        if type(input_image) is not np.ndarray: input_image = np.asarray(input_image)

        # Create figure and a subplot
        fig, ax = plt.subplots()

        ax.hist(input_image.flatten(), bins=bins, color=color, )
        if display: plt.show()


    def display_histogram_v2(self, data, n_bins, cumulative=False, x_label="", y_label="", title="", plot=False):

        """
        Display data as a histogram
        :param data:
        :param n_bins:
        :param cumulative:
        :param x_label:
        :param y_label:
        :param title:
        :param plot:
        :return:
        """

        _, ax = plt.subplots()
        ax.hist(data, bins=n_bins, cumulative=cumulative, color='#539caf')
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        ax.set_title(title)

        if plot: plt.show()


    def display_overlaid_histogram(self, data1, data2, n_bins=0, data1_name="", data1_color="#539caf", data2_name="", data2_color="#7663b0", x_label="", y_label="", title="", plot=False):

        """
        Display a histogram overlaid on another one
        :param data1:
        :param data2:
        :param n_bins:
        :param data1_name:
        :param data1_color:
        :param data2_name:
        :param data2_color:
        :param x_label:
        :param y_label:
        :param title:
        :param plot:
        :return:
        """
        # Set the bounds for the bins so that the two distributions are fairly compared
        max_nbins = 10
        data_range = [min(min(data1), min(data2)), max(max(data1), max(data2))]
        binwidth = (data_range[1] - data_range[0]) / max_nbins

        if n_bins == 0:
            bins = np.arange(data_range[0], data_range[1] + binwidth, binwidth)
        else:
            bins = n_bins

        # Create the plot
        _, ax = plt.subplots()
        ax.hist(data1, bins=bins, color=data1_color, alpha=1, label=data1_name)
        ax.hist(data2, bins=bins, color=data2_color, alpha=0.75, label=data2_name)
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        ax.set_title(title)
        ax.legend(loc='best')

        if plot: plt.show()


    def display_scatterplot(self, x_data, y_data, x_label="", y_label="", title="", color="r", plot=False, yscale_log=False):

        """
        Can create a scatter plot with variant sizes
        :param x_data:
        :param y_data:
        :param x_label:
        :param y_label:
        :param title:
        :param color:
        :param plot: Whether to display or not
        :param yscale_log: for a logarithmic vertical scale
        :return:
        """

        # Create the plot object
        _, ax = plt.subplots()

        # Plot the data, set the size (s), color and transparency (alpha)
        # of the points
        ax.scatter(x_data, y_data, s=10, color=color, alpha=0.75)

        if yscale_log == True:
            ax.set_yscale('log')

        # Label the axes and provide a title
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        if plot: plt.show()


    def display_lineplot(self, x_data, y_data, x_label="", y_label="", title="", plot=False):

        """
        Can display a line plot or multiple line plots
        :param x_data:
        :param y_data:
        :param x_label:
        :param y_label:
        :param title:
        :param plot:
        :return:
        """
        # Create the plot object
        _, ax = plt.subplots()

        # Plot the best fit line, set the linewidth (lw), color and
        # transparency (alpha) of the line
        ax.plot(x_data, y_data, lw=2, color='#539caf', alpha=1)

        # Label the axes and provide a title
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        if plot: plt.show()


    def display_barplot(self, x_data, y_data, error_data, x_label="", y_label="", title="", plot=False):

        """
        Displays a simple bar chart
        :param y_data:
        :param error_data:
        :param x_label:
        :param y_label:
        :param title:
        :param plot:
        :return:
        """
        _, ax = plt.subplots()
        # Draw bars, position them in the center of the tick mark on the x-axis
        ax.bar(x_data, y_data, color='#539caf', align='center')
        # Draw error bars to show standard deviation, set ls to 'none'
        # to remove line between points
        ax.errorbar(x_data, y_data, yerr=error_data, color='#297083', ls='none', lw=2, capthick=2)
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        ax.set_title(title)
        if plot: plt.show()


    def display_stackedbarplot(self, x_data, y_data_list, colors, y_data_names="", x_label="", y_label="", title="", plot=False):

        """
        Displays a grouped bar plot
        :param x_data:
        :param y_data_list:
        :param colors:
        :param y_data_names:
        :param x_label:
        :param y_label:
        :param title:
        :param plot:
        :return:
        """

        _, ax = plt.subplots()
        # Draw bars, one category at a time
        for i in range(0, len(y_data_list)):
            if i == 0:
                ax.bar(x_data, y_data_list[i], color=colors[i], align='center', label=y_data_names[i])
            else:
                # For each category after the first, the bottom of the
                # bar will be the top of the last category
                ax.bar(x_data, y_data_list[i], color=colors[i], bottom=y_data_list[i - 1], align='center', label=y_data_names[i])
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        ax.set_title(title)
        ax.legend(loc='upper right')
        if plot: plt.show()


    def display_groupedbarplot(self, x_data, y_data_list, colors, y_data_names="", x_label="", y_label="", title="", plot=False):

        """
        Displays a 3D grouped bar plot
        :param x_data:
        :param y_data_list:
        :param colors:
        :param y_data_names:
        :param x_label:
        :param y_label:
        :param title:
        :param plot:
        :return:
        """
        _, ax = plt.subplots()
        # Total width for all bars at one x location
        total_width = 0.8
        # Width of each individual bar
        ind_width = total_width / len(y_data_list)
        # This centers each cluster of bars about the x tick mark
        alteration = np.arange(-(total_width / 2), total_width / 2, ind_width)

        # Draw bars, one category at a time
        for i in range(0, len(y_data_list)):
            # Move the bar to the right on the x-axis so it doesn't
            # overlap with previously drawn ones
            ax.bar(x_data + alteration[i], y_data_list[i], color=colors[i], label=y_data_names[i], width=ind_width)
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        ax.set_title(title)
        ax.legend(loc='upper right')
        if plot: plt.show()


    """
    Overlay returning functions
    """


    def return_image_text_overlay(self, text, image, dim_3d=False, color=1.0, scale=0.5, thickness=1):

        """
        This function displays text over an image
        :param text: The text to overlay
        :param image: The input image volume, 2D or 3D numpy array
        :param dim_3d: whether this is a 3d or 2d image
        :param color: the color of the text, 0 for black, 1 for white and grayscale in between
        :return: image: the image or volume with text overlaid
        """

        # Define color as grayscale between white and black based on max pixel value
        max_pixel = np.amax(image)
        text_color = (max_pixel*color, max_pixel*color, max_pixel*color)

        # Define the origin of the text
        if dim_3d: origin = (0, int(image.shape[2]*.9))
        else: origin = (0, int(image.shape[1]*.9))

        # Create a copy of the image with text
        if not dim_3d:
            texted_image = cv2.putText(img=np.copy(image), text=text, org=origin, fontFace=0, fontScale=scale,
                                   color=text_color, thickness=thickness)

        # For 3D, loop and addend a copied image volume
        else:

            # Copy image volume
            texted_image = np.copy(image)

            # addend every slice
            for z in range(image.shape[0]):
                texted_image[z] = cv2.putText(img=np.copy(image[z]), text=text, org=origin, fontFace=0, fontScale=0.5,
                                           color=text_color, thickness=1)


        return texted_image


    def return_image_overlay(self, img, mask):
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


    def return_shape_overlay(self, img, coordinates, color=0.5, shape='BOX'):

        """
        Draws a shape on the image
        :param img: Input image
        :param coordinates: box = (y1, x1, y2, x2), circle = (cy, cx, radius),
        :param color: Scalar for greyscale, RGB for color
        :param shape: BOX, RECTANGLE,  CIRCLE, LINE
        :return:
        """

        if shape == 'CIRCLE':

            y1, x1, r = coordinates
            return_image = cv2.rectangle(img, (y1, x1), r, color)

        elif shape == 'LINE':

            x1, y1, x2, y2 = coordinates
            return_image = cv2.line(img, (y1, x1), (y2, x2), color)

        else:

            x1, y1, x2, y2 = coordinates
            return_image = cv2.rectangle(img, (y1, x1), (y2, x2), color)

        return return_image


    def display_vol_label_overlay(self, volume, segments, title, display_non=False, plot=False, dim3D=True):

        """
        Shortcut to find the center of the 3D segments and display an overlay
        :param volume: Input 3D volume numpy array. any type
        :param segments: Input 3D segments numpy array any type
        :param display_non: Whether to display a non overlaid image
        :param title: what to title the image
        :param plot: Whether to plot and show the result right away
        :param dim3D: True for volumes, False for 2D images
        :return:
        """

        # Convert to float32
        vol, segs = volume.astype(np.float32), segments.astype(np.float32)

        if dim3D:

            # Find the largest blob of the segments
            _, cn = self.largest_blob(segments)

            # Retreive the overlaid image
            overlay = self.return_image_overlay(vol[cn[0]], segs[cn[0]])

            # Display orig
            if display_non: self.display_single_image(vol[cn[0]], False, title=title)

        else:

            # Retreive the overlaid image
            overlay = self.return_image_overlay(vol, segs)

            # Display orig
            if display_non: self.display_single_image(vol, False, title=title)

        # Display the overlay
        self.display_single_image(overlay, plot, title=title)


    def display_whole_vol_overlay(self, volume, segments, plot=False, ret=False):

        """
        Shortcut to display a whole volume overla
        :param volume: Input 3D volume numpy array. any type
        :param segments: Input 3D segments numpy array any type
        :param plot: Whether to plot and show the result right away
        :param return: whether to return the overlay
        :return:
        """

        # Convert to float32
        vol, segs = volume.astype(np.float32), segments.astype(np.float32)

        # Define the overlay volume
        overlay = np.zeros([vol.shape[0], vol.shape[1], vol.shape[2], 3], np.float32)

        # Loop through all the slices
        for z in range(vol.shape[0]):
            # Retreive the overlaid image
            overlay[z] = self.return_image_overlay(vol[z], segs[z])

        # Display the overlay
        self.display_volume(overlay, plot)

        if ret: return overlay

    def display_breast_mask(self, im, mask):

        f = plt.figure(figsize=(12, 12))
        ax = plt.subplot(111)
        ax.imshow(im, vmin=0, vmax=4096, cmap='gray')
        ax.imshow(mask, alpha=.3, cmap='inferno')  # alpha controls the transparency


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


    def save_vol_gif(self, filename, array, fps=10, scale=1.0):

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

        # Normalize the volume into 0-255 range for .gif file
        volume_norm = np.zeros_like(array, dtype=np.uint8)
        for z in range(array.shape[0]):
            volume_norm[z] = cv2.normalize(array[z], dst=volume_norm[z], alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        volume_norm, array = array, volume_norm
        del volume_norm

        # copy into the color dimension if the images are black and white
        if array.ndim == 3:
            array = array[..., np.newaxis] * np.ones(3)

        # make the moviepy clip
        clip = ImageSequenceClip(list(array), fps=fps).resize(scale)
        clip.write_gif(filename, fps=fps)
        return clip


    """
         Tool functions: Most of these are hidden
    """


    def _process_key(self, event):
        fig = event.canvas.figure
        ax = fig.axes[0]
        if event.button == 'down':
            self._previous_slice(ax)
        elif event.button == 'up':
            self._next_slice(ax)
        fig.canvas.draw()


    def _previous_slice(self, ax):
        volume = ax.volume
        ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
        ax.images[0].set_array(volume[ax.index])


    def _next_slice(self, ax):
        volume = ax.volume
        ax.index = (ax.index + 1) % volume.shape[0]
        ax.images[0].set_array(volume[ax.index])


    def _remove_keymap_conflicts(self, new_keys_set):
        for prop in plt.rcParams:
            if prop.startswith('keymap.'):
                keys = plt.rcParams[prop]
                remove_list = set(keys) & new_keys_set
                for key in remove_list:
                    keys.remove(key)