
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.figure

import matplotlib.lines as LineType

import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset, inset_axes
from matplotlib.backends.backend_pdf import PdfPages
from typing import Optional, Union, Literal, Tuple, Dict, List, Any

# xdf file importation
import pyxdf
# library for creating filters
from scipy.signal import welch, periodogram, get_window, argrelextrema, convolve
from scipy.signal.windows import hamming, boxcar


from my_filters import *

# =============================================================================
############################### General Tools  ################################
# =============================================================================


def merge_lists_alternatively(list1, list2) -> List:
    """
    Merges two lists by alternating the columns of each. 
    The first column of the merged list will be the first column of `list1`, and the second will be the first of `list2` and so on.

    Returns the filtered signal, and for filter characterization the frequency response (h) and its frequencies.

    Parameters:
    -----------
        - `list1` (list) : List 1 to merge, its first column will be the first column of the `merged_list`
        - `list2` (list) : List 2 to merge, its first column will be the second column of the `merged_list`


    Returns:
    -----------
        - `merged_list` (list) : Resulting alternating list of `list1` and `list2` columns

    """
    merged_list = []
    min_len = min(len(list1), len(list2))

    for i in range(min_len):
        merged_list.append(list1[i])
        merged_list.append(list2[i])

    # Append any remaining elements from the longer list
    merged_list.extend(list1[min_len:])
    merged_list.extend(list2[min_len:])

    return merged_list


def save_figures_to_pdf_single_per_page(pdf_filename: str, figures_list: list) -> None:
    """
    Saves a list of figures in a single PDF file.

    Parameters:
    ----------
        - pdf_filename (str): Path of the desired pdf file.
        - figures_list (list): list of figure objects.

    Returns:
    --------
        - `None` (bool): This function does not return a value. It saves the figures to the specified PDF file.

    Examples:
    --------
        # Create some sample data
        x = [1, 2, 3, 4, 5]
        y1 = [10, 5, 8, 4, 7]
        y2 = [8, 6, 5, 2, 4]

        # Create two Matplotlib figures
        fig1, ax1 = plt.subplots()
        ax1.plot(x, y1, label='Line 1')
        ax1.set_title('Figure 1')

        fig2, ax2 = plt.subplots()
        ax2.plot(x, y2, label='Line 2')
        ax2.set_title('Figure 2')

        # Store the figures in a list
        figure_list = [fig1, fig2]

        # Save the figures to a PDF file
        save_figures_as_pdf(pdf_filename='output.pdf', figures_list=figure_list)
    """
    pdf_file = PdfPages(pdf_filename)

    # iterating over the numbers in list
    for figure in figures_list:

        # and saving the files
        figure.savefig(pdf_file, format='pdf')

    # close the object
    pdf_file.close()


def save_figures_to_pdf_multiple_per_page(figures: List[matplotlib.figure.Figure], filename: str, rows_per_page: int = 2) -> None:
    """
    Save a list of Matplotlib figures to a PDF file with control over the number of
    rows of figures per page.

    Parameters:
    -----------
        - `figures` (list): List of Matplotlib figure objects.
        - `filename` (str): Name of the output PDF file.
        - `rows_per_page` (int): Number of rows of figures per page.

    Returns:
    -----------
        - `None` (bool): This function does not return a value. It saves the figures to the specified PDF file.
    """

    total_figures = len(figures)
    total_pages = (total_figures + rows_per_page - 1) // rows_per_page
    fig_width, fig_height = figures[0].get_size_inches()

    with PdfPages(filename) as pdf:
        for page_num in range(total_pages):
            fig, axs = plt.subplots(rows_per_page, 1, figsize=(
                fig_width, fig_height*rows_per_page))

            for row in range(rows_per_page):
                figure_index = page_num * rows_per_page + row
                if figure_index < total_figures:
                    figure = figures[figure_index]
                    axs[row].imshow(figure.canvas.renderer.buffer_rgba())
                    axs[row].axis('off')

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


# =============================================================================
############################### Manage Xdf files  #############################
# =============================================================================


def stream_Finder_by_name(datalist, name):
    """
    This function browse the dictionary  

    Finds the index of the datalist element loaded from the ".xdf file" and returns it
    input arguments the list containing the data, and the name of the stream to search for

    inputs: object (plt, or axis[a,b]), numpy.ndarray(2D).
    outputs: [None]
    """
    print("StreamFinder executed")
    streamindex = None
    for i in datalist:
        if any(str(name).upper() in string.upper() for string in i["info"]["name"]):
            streamindex = datalist.index(i)
            print(streamindex)
            # print(i["info"]["type"])
            return streamindex


def list_stream_ids(stream_list: list):
    """
    Lists the names and types of the streams founded by pyxdf's multistream importer.

    Parameters:
    ----------
        - stream_list (list): List of streams founded.

    Returns:
    --------
        - stream_ids(dict): Dictionary of stream names and types
            - "names": stream_names (list): list of founded streams names
            - "types": stream_types (list): list of founded streams types
    """
    stream_names = []
    stream_types = []

    for stream in stream_list:
        stream_name = stream["info"]["name"][0]
        stream_names.append(stream_name)
        stream_type = stream["info"]["type"][0]
        stream_types.append(stream_type)

    stream_ids = {"names": stream_names, "types": stream_types}

    return stream_ids


def retrieve_stream_data_from_xdf(xdf_input_filepath: Optional[str] = None, stream_type: Optional[str] = None,
                                  stream_name: Optional[str] = None) -> dict:
    """
    Retrieves data and recording setup information from a selected stream of an xdf file and return them tidied for ReArm needs.


    Returns the data as a nested dictionary.

    Note: Currently handeled stream types: "EEG", "Mocap", "Markers".

    Parameters:
    -----------
        - xdf_input_filepath (str): Filepath towards xdf file to analyze
        - stream_type (str): (optional) Type of the stream to look for.
        - stream_name (str): (optional) Name of the stream if multiple are found.

    Returns:
    -----------
        - stream_result_data (dict): 
            - "data" (dict): 
                - "time_series" : stream_data_samples (np.ndarray) - 2D array of raw samples (each column contains data from a channel)
                - "timestamps": stream_data_timestamps (np.ndarray) - 1D array of raw timestamps (associated to each sample)
            - "infos" (dict):
                - "sample_rate" (dict):
                    - "nominal_srate" : nominal_srate (float) - nominal sampling rate of the recording
                    - "effective_srate" : effective_srate (float) - effective sampling rate of the recording 
                - "channels" (dict):
                    - "names" : channel_names (list) - list of channel names
                    - "units" : channel_units (list) - list of channel data units

                - "recording_time_limits" (dict): 
                    - "recording_start" : recording_start (float) - first_timestamp (units?)
                    - "recording_end" : recording_end (float) - last_timestamp (units?)

    """

    if xdf_input_filepath is None:
        raise Exception(
            f"xdf_input_filepath was not specified: Argument must be specified to access xdf file and retrieve its data.")
    if stream_type is None and stream_name is None:
        stream_list, fileheader = pyxdf.load_xdf(xdf_input_filepath)
        stream_ids = list_stream_ids(stream_list)
        raise Exception(
            f"Both stream_type and stream_name were not specified: Please specify at least one of these arguments. \n Streams names found: {stream_ids['names']} \n Streams types found: {stream_ids['types']}")
    elif stream_type is not None and stream_name is not None:
        stream_list, fileheader = pyxdf.load_xdf(xdf_input_filepath, select_streams=[
                                                 {'type': stream_type, 'name': stream_name}])
    elif stream_name is not None:
        stream_list, fileheader = pyxdf.load_xdf(
            xdf_input_filepath, select_streams=[{'name': stream_name}])
        if len(stream_list) > 1:
            stream_ids = list_stream_ids(stream_list)
            raise Exception(
                f"More than one stream nammed '{stream_name}' was found - Corresponding stream types : {stream_ids['types']} \n Try specifying the stream type when calling the function. ")
    else:
        stream_list, fileheader = pyxdf.load_xdf(
            xdf_input_filepath, select_streams=[{'type': stream_type}])
        if len(stream_list) > 1:
            stream_ids = list_stream_ids(stream_list)
            raise Exception(
                f"More than one stream of type '{stream_type}' was found - Corresponding stream names : {stream_ids['names']} \n Try specifying the stream name when calling the function. ")

    # access stream
    stream_ids = list_stream_ids(stream_list)
    print(
        f"Accessing stream: \nType: '{stream_ids['types'][0]}' | Name: '{stream_ids['names'][0]}'")
    stream_data = stream_list[0]

    # get stream data
    stream_data_timestamps = stream_data["time_stamps"]
    stream_data_samples = stream_data["time_series"]

    # get channel names and units
    channel_names = []
    channel_units = []
    if stream_type == "EEG" or stream_type == "Accelerometer":
        for i in range(stream_data_samples.shape[-1]):
            channel_i_name = stream_data["info"]["desc"][0]["channel"][i]["name"][0]
            channel_i_unit = stream_data["info"]["desc"][0]["channel"][i]["unit"][0]
            channel_names.append(channel_i_name)
            channel_units.append(channel_i_unit)
    elif stream_type == "MoCap":
        for i in range(stream_data_samples.shape[-1]):
            channel_i_name = stream_data["info"]["desc"][0]["channels"][0]["channel"][i]["label"][0]
            channel_i_unit = stream_data["info"]["desc"][0]["channels"][0]["channel"][i]["unit"][0]
            channel_names.append(channel_i_name)
            channel_units.append(channel_i_unit)
    elif stream_type == "Markers":
        channel_names.append("Maker_labels")
        channel_units.append(None)
    else:
        raise Exception(f"stream type {stream_type} not handeled yet")

    # get sample rates
    nominal_srate = float(stream_data["info"]["nominal_srate"][0])
    effective_srate = float(stream_data["info"]["effective_srate"])

    # get recording landmarks
    recording_start = float(
        stream_data["footer"]["info"]["first_timestamp"][0])
    recording_end = float(stream_data["footer"]["info"]["last_timestamp"][0])

    # store data in dictionaries
    data_dict = {"time_series": stream_data_samples,
                 "timestamps": stream_data_timestamps}

    info_dict = {"sample_rate": {"nominal": nominal_srate, "effective": effective_srate},
                 "recording_time_limits": {"start": recording_start, "end": recording_end},
                 "channels": {"names": channel_names, "units": channel_units}}

    stream_result_data = {"data": data_dict, "infos": info_dict}
    return stream_result_data


def create_marker_times_labels_array(marker_time_stamps: Optional[np.ndarray] = None, marker_labels: Optional[np.ndarray] = None,
                                     xdf_input_filepath: Optional[str] = None) -> Union[np.ndarray, None]:
    """
    Create an array combining the markers labels and their timestamps.
        If xdf file specified, timestamps are retrieved from the file and processed to start relative to recording start, not to unix epoch anymore.
        If marker_time_stamps and marker_labels, the arrays are stacked and returned as defined.


    Parameters
    ----------
        marker_time_stamps(np.ndarray): 1D array containing the marker timestamps.
        marker_labels(np.ndarray): 1D array containing the markers labels.
        xdf_input_filepath(str): Filepath of the EEG recordings as xdf file.

    Returns
    -------
        markers_times_labels(np.ndarray): 2D array containing the markers's timestamps alongside their labels.
    """
    all_args = [xdf_input_filepath, marker_time_stamps, marker_labels]
    is_all_none = all(element is None for element in all_args)

    if is_all_none is None:
        print("No arguments specified.")
        markers_times_labels = None
    elif marker_time_stamps is None or marker_labels is None:
        print("marker_time_stamps and/ or marker_labels are none")
        markers_times_labels = None
    elif xdf_input_filepath:
        # Retrieve directly from xdf file markers timestamps relative to recording start and their labels
        xdf_data, header = pyxdf.load_xdf(xdf_input_filepath, select_streams=[{'type': 'EEG'}, {
            'type': 'Markers', 'name': 'MouseToNIC'}])
        EEG_stream = xdf_data[0]
        Mouse_markers_stream = xdf_data[1]
        Mouse_markers_labels = Mouse_markers_stream["time_series"]
        Mouse_markers_times = Mouse_markers_stream["time_stamps"] - \
            EEG_stream["time_stamps"][0]
        markers_times_labels = np.column_stack(
            (Mouse_markers_times, Mouse_markers_labels))
    else:
        # stack given arrays to create the marker_times_labels array
        markers_times_labels = np.column_stack(
            (marker_time_stamps, marker_labels))
    return markers_times_labels


def create_marker_times_labels_array2(marker_time_stamps: Optional[np.ndarray] = None, marker_labels: Optional[np.ndarray] = None,
                                      xdf_input_filepath: Optional[str] = None) -> np.ndarray:
    """
    Create an array combining the markers labels and their timestamps.
        If `xdf file specified`, timestamps are retrieved from the file (mouse marker stream).
        If `marker_time_stamps` and `marker_labels`, the arrays are stacked and returned as defined.

    Parameters
    ----------
        - `marker_time_stamps` (np.ndarray): 1D array containing the marker timestamps.
        - `marker_labels` (np.ndarray): 1D array containing the markers labels.
        - `xdf_input_filepath` (str): Filepath of the EEG recordings as xdf file.

    Returns
    -------
        - `markers_times_labels` (np.ndarray): 2D array containing the markers's timestamps alongside their labels in this order: [marker_timestamps,marker_labels].
    """
    all_args = [xdf_input_filepath, marker_time_stamps, marker_labels]
    is_all_none = all(element is None for element in all_args)
    if xdf_input_filepath is not None and marker_time_stamps is None and marker_labels is None:
        # Retrieve directly from xdf file markers timestamps relative to recording start and their labels
        mouse_markers_data = retrieve_stream_data_from_xdf(xdf_input_filepath=xdf_input_filepath,
                                                           stream_type="Markers", stream_name="MouseToNIC")
        mouse_markers_labels = mouse_markers_data["data"]["time_series"]
        mouse_markers_times = mouse_markers_data["data"]["timestamps"]
        markers_times_labels = np.column_stack(
            (mouse_markers_times, mouse_markers_labels))
    elif xdf_input_filepath is None and marker_time_stamps is not None and marker_labels is not None:
        # stack given arrays to create the marker_times_labels array
        markers_times_labels = np.column_stack(
            (marker_time_stamps, marker_labels))
    else:
        raise Exception(
            "Check arguments, please specify either xdf_input_filepath or marker_time_stamps with marker_labels")
    return markers_times_labels

# =============================================================================
############################# show_markers  #####################################
# =============================================================================


def show_markers(plot_type: plt.Axes, markers_times_array: np.ndarray) -> Tuple[LineType.Line2D, LineType.Line2D]:
    """
    Custom function to display event markers as vertical lines on a graph (plt or axis). 

    Inherits of the `plot_type` object to add marker to figure.

    Parameters:
    ------------
        - `markers_times_array` (np.ndarray): 2D array of markers with corresponding timestamps as [Marker,Timestamp]
        - `plot_type` (plt.Axes): plt.Axes object
    Returns:
    ------------
        - `marker111` (LineType.Line2D): Line for plot
        - `marker100` (LineType.Line2D): Line for plot
    """

# iterate over an array of markers
    marker111 = plot_type.axvline()
    marker100 = plot_type.axvline()
    for i in markers_times_array:
        if i[1] == 111:
            # print(i)
            # plot a line x=time_stamp associated to the i marker of type 111 (begining of task)
            # print("plot_type is : ", type(plot_type), plot_type)
            marker111 = plot_type.axvline(x=i[0], color="b", label="111")
        else:
            # print(i)
            # plot a line x=time_stamp associated to the i marker of type 110 (begining of task)
            marker100 = plot_type.axvline(x=i[0], color="r", label="100")
    return marker111, marker100


def show_markers2(plot_type: plt.Axes,
                  markers_times_array: np.ndarray) -> list:
    """
    Function that displays event markers as vertical lines on a graph (plt or axis). 
    Inherits of the plot_type object to add marker to figure.

    Parameters:
    ----------
    plot_type (plt.Axes): Parent graph object (plt.Axes)
    markers_times_array (np.ndarray): 2D array of markers where column 1 is timestamps and column 2 marker labels in this order.

    Returns:
    ----------
        markers (list): List of axvline methods with specific arguments given
    """
    unique_labels = set()
    markers = []

    # iterate over an array of markers
    for marker in markers_times_array:
        timestamp, label = marker
        label = str(int(label))
        # use different color for marker 111 and 100
        if label == "111":
            color = "b"
        elif label == "100":
            color = "r"
        else:
            color = "green"

        # each marker type corresponds to a unique label
        if label in unique_labels:
            label = None
        else:
            unique_labels.add(label)

        marker_axvline_obj = plot_type.axvline(
            x=timestamp, color=color, label=label)
        markers.append(marker_axvline_obj)

    return markers

# =============================================================================
############################# Single_plotter  #################################
# =============================================================================


def single_plot(filename: str, x: np.ndarray, y: np.ndarray, fig_title: str,
                xlabel: str, ylabel: str, markers_times_array: Optional[np.ndarray] = None, point_style: str = "-k",
                line_width: int | float = 1, fig_number: Optional[int] = None) -> None:
    """
    Custom multipurpose function that displays a single graph

    Single_plot uses the x and y datas as inputs for the plot.
    Optionally calls show_markers function.`

    Parameters:
    -----------
        - `filename` (str): Filename
        - `x` (np.ndarray): x series
        - `y` (np.ndarray): y series
        - `fig_title` (str): figure title
        - `xlabel` (str): x axis label
        - `ylabel` (str): y axis label
        - `markers_times_array`(optional(np.ndarray)): 
        - `point_style` (str): point style
        - `line_width` (float,int): line width
        - `fig_number` (optional(int)): figure number

    Returns:
    -----------
        - `None` (bool): This function does not return a value. It plots the figure.

    """

    # Creation of the figure
    """
    if fig_number:
        plt.figure(fig_number)
    plt.plot(x, y, point_style, lw=line_width)
    plt.title(str(fig_title+"\n"+filename))
    plt.xlabel(str(xlabel))
    plt.ylabel(str(ylabel))

    # Displays markers (optional)
    if markers_times_array is not None:
        show_markers(plt, markers_times_array)
    """

    figure, axis = plt.subplots(num=fig_number, layout="constrained")
    figure.suptitle(f"{fig_title} \n {filename}")
    axis.set_xlabel(f"{xlabel}")
    axis.set_ylabel(f"{ylabel}")
    axis.plot(x, y, point_style, lw=line_width)

    # Displays markers (optional)
    if markers_times_array is not None:
        show_markers(axis, markers_times_array)
    figure.legend()
    figure.show()

# =============================================================================
############################# Mosaic_plotter  #################################
# =============================================================================


def mosaic_plot(figure: matplotlib.figure.Figure, axis: np.ndarray, filename: str, x: np.ndarray, y: np.ndarray, fig_title: str,
                xlabel: str, ylabel: str, channels: dict, markers_labels_times: Optional[np.ndarray] = None) -> None:
    """
    Custom function that display a mosaic of graphs for each channel

    Mosaic_plot uses the figure and axis objects formerly instanciated to plot x and y data on each cell of the figure.
    The cells are defined by coordinates a and b (2,4 for now).Channel name is provided by the dictionary of channels. 
    Figure and plot titles are directly given by arguments.

    Plotting and labels goes by iteration on each cell before showing the figure with title.

    Parameters:
    -----------
        - `figure` (matplotlib.figure.Figure) : Figure object
        - `axis` (np.ndarray): array of axis objects of the Figure
        - `filename`: Filename
        - `x` (np.ndarray): x series as columns
        - `y` (np.ndarray): y series as columns
        - `fig_title` (str): figure title
        - `xlabel` (str): x axis label
        - `ylabel` (str): y axis label
        - `channels`(dict) : dictionary of channels names
        - `markers_labels_times`(np.ndarray): 2D array of markers with corresponding timestamps as [Marker,Timestamp]
    Results:
    -----------
        - `None` (bool): This function does not return a value. It plots the figure.
    """
    count = 0
    # Get the figure's geometry
    rows, cols = axis.shape
    print("FIGURE ROWS:", rows, "COLS:", cols)

    # Iterate on each coordinate of the plot to assign figure content and titles
    for a in range(rows):  # two rows of graphs
        for b in range(cols):  # each row conists of 4 graphs
            axis[a, b].plot(x, y[:, count])
            # show_markers("axis["+str(a)+","+str(b)+"]", Markers_times_labels)
            # Displays markers (optional)
            if markers_labels_times is not None:
                show_markers(axis[a, b], markers_labels_times)

            axis[a, b].set_title("Electrode "+str(count+1) +
                                 ":" + channels["Channel_"+str(count+1)])
            axis[a, b].set_xlabel(xlabel)
            axis[a, b].set_ylabel(ylabel)
            count = count+1
    plt.suptitle(fig_title+"\n"+filename)
    plt.legend()


def annotate_local_extrema(x: np.ndarray, y: np.ndarray, axis: plt.Axes, order: int = 1, window_size: int = 1):
    """
    Detects local extrema of a given distribution, and anotates a plot to highlight the extrema coordinates.
    - Relies on the scipy argrelextrema function to compare each value to its n=order neigboors on each side.
    - Relies on the scipy convolve function to compute a moving average of the distribution (extrema detection resolution).

    Parameters
    ----------

        - `x`: values of the distribution
        - `y`: amplitudes of the distribution
        - `axis`: parent plot on which annotation will be added
        - `order`: Number of neighboor points to consider for comparison
        - `window-size` (int): number of points used for computing the moving average of the distribution
    Returns:
    ----------
    """
    # Apply moving average smoothing
    smoothed_y = convolve(y, np.ones(window_size) / window_size, mode='same')

    # Find indices of local maxima and minima in the smoothed data
    local_maxima_indices = argrelextrema(
        smoothed_y, np.greater, order=order)[0]
    local_minima_indices = argrelextrema(smoothed_y, np.less, order=order)[0]

    # Annotate local maxima
    for i in local_maxima_indices:
        max_coords = (x[i], y[i])
        axis.annotate(f'Max: {max_coords}', xy=max_coords, xytext=(10, 10), textcoords='offset points', bbox=dict(boxstyle='round, pad=0.2', fc='white', ec='none', alpha=0.8),
                      arrowprops=dict(arrowstyle="->", color='r'))

    # Annotate local minima
    for i in local_minima_indices:
        min_coords = (x[i], y[i])
        axis.annotate(f'Min: {min_coords}', xy=min_coords, xytext=(10, -10), textcoords='offset points', bbox=dict(boxstyle='round, pad=0.2', fc='white', ec='none', alpha=0.8),
                      arrowprops=dict(arrowstyle="->", color='b'))
# =============================================================================
############################# Compute_FFT_on_channels  ########################
# =============================================================================


def fft_compute_on_single_channel2(signal, Fs):
    N = np.shape(signal)[0]
    f_res = Fs/N  # freq res

    freq_vect = np.linspace(0, (N-1)*f_res, N)
    amplitude_fft = np.fft.fft(signal)
    amplitude_fft_magnitude = np.abs(amplitude_fft)/N

    freq_vect_for_plot = freq_vect[0:int(N/2+1)]  # +1 cf slicing
    amplitude_fft_magnitude_for_plot = 2*amplitude_fft_magnitude[0:int(N/2+1)]
    return freq_vect_for_plot, amplitude_fft_magnitude_for_plot


def compute_fft_on_all_channels2(channels_signals: np.ndarray, Fs: int | float):
    channels_fft_frequencies = []
    channels_fft_magnitudes = []

    for signal in channels_signals.T:
        frequencies, magnitudes = fft_compute_on_single_channel2(signal, Fs)
        channels_fft_frequencies.append(frequencies)
        channels_fft_magnitudes.append(magnitudes)

    channels_fft_frequencies = np.transpose(np.array(channels_fft_frequencies))
    channels_fft_magnitudes = np.transpose(np.array(channels_fft_magnitudes))

    return {"FFT_frequencies": channels_fft_frequencies,
            "FFT_magnitudes": channels_fft_magnitudes}

# =============================================================================
###################### Marker Nearest time-index finder  ######################
# =============================================================================


def nearest_timestamps_array_finder2(signal_times_stamps: np.ndarray, markers: np.ndarray):
    """
    Finds the nearest timestamps to each marker in EEG signal timestamps array 

        Application: 
            Useful when the markers timestamps may not be found in the EEG data due to Srate and time launch.
                ie: if the marker timestamps are foudn between two signal samples.

    Parameters:
    -----------
        signal_times_stamps(np.ndarray): 1D array of the signal timestamps
        markers (np.ndarray): 2D array (markers_original_timestamps,marker_labels)
    Returns:
    -------
        nearest_indices_timestamps(np.ndarray): 2D array  as [nearest_time_stamps_indices EEG,nearest_time_stamps]
    inputs: numpy.ndarray(1D),numpy.ndarray(1D)

    """
    nearest_time_stamps = []
    nearest_time_stamps_indices = []
    print("MARKERS LEN:", len(markers))
    for i in range(len(markers)):
        original_time_stamp = markers[i, 0]
        # array of differences beween the eeg times and the original marker' timestamp
        difference_array = np.absolute(
            signal_times_stamps-original_time_stamp)
        # find the index of the minimal difference in the markers times stamps (nearest timestamp)
        index = difference_array.argmin()
        # append it to the liste of nearest timestamps
        nearest_time_stamps.append(signal_times_stamps[index])
        nearest_time_stamps_indices.append(index)

    nearest_time_stamps = np.array(nearest_time_stamps)
    nearest_time_stamps_indices = np.array(
        nearest_time_stamps_indices)
    nearest_indices_timestamps = np.column_stack(
        (nearest_time_stamps_indices, nearest_time_stamps))

    return nearest_indices_timestamps


def nearest_timestamps_array_finder(signal_times_stamps: np.ndarray, markers: np.ndarray):
    """
    Finds the nearest marker timestamps corresponding to a sample in `signal_timestamps` array.

    For each `markers` timestamp, the function looks for the nearest timestamp in `signal_timestamps` array
    and its corresponding index.

    Application: 
        Useful when the markers timestamps may not be found in the signal due to sampling rate and time syncrhonization considerations.
            ie: if the marker timestamps are found between two signal samples.

    Parameters:
    -----------
        `signal_times_stamps` (np.ndarray): 1D array of the signal timestamps
        `markers` (np.ndarray): 2D array (markers_original_timestamps,marker_labels)

    Returns:
    -------
        `nearest_indices_timestamps` (dict): Dictionary of 3 key-value pairs, each value being a 1D array
            "markers_timestamps" :  1D array of the timestamps associated to each marker.\n
            "markers_timestamp_indices" : 1D array of the index values of the associated timestamps in the signal time array.\n
            "marker_labels" : 1D array of the marker labels.\n
    """
    nearest_time_stamps = []
    nearest_time_stamps_indices = []
    print("MARKERS LEN:", len(markers))

    # iterate over the  marker timestamps
    for y in markers[:, 0]:
        original_time_stamp = y
        # array of differences beween the eeg times and the original marker' timestamp
        difference_array = np.absolute(
            signal_times_stamps-original_time_stamp)
        # find the index of the minimal difference in the markers times stamps (nearest timestamp)
        index = difference_array.argmin()
        # append it to the liste of nearest timestamps
        nearest_time_stamps.append(signal_times_stamps[index])
        nearest_time_stamps_indices.append(index)

    # convert the list to array
    nearest_time_stamps = np.array(nearest_time_stamps)
    nearest_time_stamps_indices = np.array(
        nearest_time_stamps_indices)
    marker_labels = markers[:, 1]

    # stack arrays (nearest_index,nearest_timestamp,label)
    """
    nearest_indices_timestamps = np.column_stack(
        (nearest_time_stamps_indices, nearest_time_stamps,marker_labels))
    """
    # store data in dictionary
    nearest_indices_timestamps = {
        "markers_timestamp_indices": nearest_time_stamps_indices,
        "markers_timestamps": nearest_time_stamps,
        "marker_labels": marker_labels
    }

    return nearest_indices_timestamps

# =============================================================================
############################ Compute lagged PSD  ##############################
# =============================================================================


def get_segment_coordinates(reference_index: int, segment_length: int, debug: bool = False) -> Tuple[int, int, int]:
    """
    [obsolete] Computes the coordinates of a segment for psd calculation relative to a reference.

    Parameters:
    -----------
        - `reference_index` (int): index of referece
        - `segment_length` (int): length of the desired segment
        - `debug` (bool): option to display segment coordinates for control

    Returns:
    -----------
        - `lower_end` (int): index of segments lower end
        - `higher_end`(int): index of segments higher end
        - `reference_end`(int): reference end  index

    """

    # Define the segment coordinates (start,end)
    # PSD on a range of time delta_time before the time stamp
    lower_end = int(reference_index)-segment_length
    # print("lower_end:", lower_end)

    # PSD on a range of time delta_time after the time stamp
    higher_end = int(reference_index)+segment_length
    # print("Higher_end:", higher_end)

    # index_range=np.arange(lower_range,int(timestamp_index)+1)
    # +1 to inclue the value at the time stamp
    reference_end = int(reference_index)

    if debug is True:

        print("segment coordinates before marker:", "(", lower_end, ";",
              reference_end, "), delta_index:", segment_length)

        print("segment coordinates after marker:", "(", reference_end, ";",
              higher_end, "), delta_index:", segment_length)
    return lower_end, higher_end, reference_end


def compute_welch_estimation_on_segment(signal: np.ndarray, direction: str, sample_rate: Union[float, int],
                                        reference_end: int, lower_end: int, higher_end: int, delta_index: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    [obsolete] Computes the psd estimation(welch method) on a specific segment of a time signal.

    inputs:numpy.ndarray(1D),str,numpy.ndarray(2D),float,str
    outputs:numpy.ndarray(1D),numpy.ndarray(1D) as columns
    """
    freq = np.ndarray([1, 1])
    Pxx_density = np.ndarray([1, 1])
    if direction == "before":
        freq, Pxx_density = welch(signal[lower_end:reference_end+1],
                                  fs=sample_rate, window="hann",
                                  nperseg=delta_index, noverlap=delta_index//2, axis=0, detrend=False)  # type: ignore
    elif direction == "after":
        freq, Pxx_density = welch(signal[reference_end:higher_end+1],
                                  fs=sample_rate, window="hann",
                                  nperseg=delta_index, noverlap=delta_index//2, axis=0, detrend=False)  # type: ignore
    else:
        print("Wrong direction provided, please specify either 'before' or 'after'")
    return freq, Pxx_density


def compute_lagged_psds_one_signal(signal: np.ndarray, Srate: Union[float, int], markers: np.ndarray,
                                   time_lag: Union[float, int] = 1, direction: str = "before") -> Tuple[np.ndarray, np.ndarray]:
    """
    [obsolete] Computes psd estimation (welch) on segments of a time signal around list of references.

    For each index references (markers) the function delimits a segment of chosen time length and direction (before after ref).
    Performs the welch method on the segment and returns two 1D arrays(column) on for frequencies other for psd resutls.
    The resulting columns are stacked in two respective 2D arrays to make a layer representing the PSDs and Freqs of an electrode.

    inputs:numpy.ndarray(1D),float,numpy.ndarray(2D),float,str
    outputs:numpy.ndarray(2D),numpy.ndarray(2D)
    """
    # time expressed in number of points, Srate number of points per sec
    delta_index = int(Srate*time_lag)
    electrode_lagged_PSDS = []
    elecrode_frequencies = []

    electrode_stacked_frequencies = np.ndarray([1, 1])
    electrode_stacked_markers = np.ndarray([1, 1])
    # iterate on the markers to compute on each the psd on a given direction
    for timestamp_index in markers[:, 0]:
        # get the coordinates of the segment on which the psd will be computed
        lower_end, higher_end, reference_end = get_segment_coordinates(
            reference_index=timestamp_index, segment_length=delta_index)

        # Compute the welch method on the segment in accordance with the deisred direction
        freq, Pxx_density = compute_welch_estimation_on_segment(
            signal, direction, Srate, reference_end, lower_end, higher_end, delta_index)
        # Store the result columns in lists (Nmarkers length)
        electrode_lagged_PSDS.append(Pxx_density)
        elecrode_frequencies.append(freq)

        # Create layers: Stack elements of the lists (1D array) to create two 2D arrays (x=PSD,y=markeri) (x=freqs,y=markeri)
        electrode_stacked_markers = np.column_stack(electrode_lagged_PSDS)
        electrode_stacked_frequencies = np.column_stack(elecrode_frequencies)
    return electrode_stacked_frequencies, electrode_stacked_markers


def compute_lagged_psd2_all_electrodes(EEG_data: np.ndarray, Srate: float | int, markers: np.ndarray,
                                       time_lag: float | int = 1, direction: str = "before"):
    """
    [obsolete] Computes psd estimation (welch) on segments of multiple time signals around list of references

    inputs:numpy.ndarray(2D),float,numpy.ndarray(2D),float,str
    outputs:numpy.ndarray(3D),numpy.ndarray(3D)
    """
    layers_psds = []
    layers_frequencies = []
    for electrode in EEG_data.T:  # iterate on electrodes
        # Produce a layer (2d array) of PSDs for an electrode
        electrode_stacked_frequencies, electrode_stacked_markers = compute_lagged_psds_one_signal(electrode, Srate, markers,
                                                                                                  time_lag=time_lag, direction=direction)
        # store the layers in a list
        layers_psds.append(electrode_stacked_markers)
        layers_frequencies.append(electrode_stacked_frequencies)

    # Stack each layer to get a 3d array (x=psdx or freqx ,y=markery,z=electrodez)
    tridi_PSDs = np.stack(layers_psds, axis=2)
    tridi_frequencies = np.stack(layers_frequencies, axis=2)

    return tridi_frequencies, tridi_PSDs

# =============================================================================
##################### Plot temporal signal and its DSPs  ######################
# =============================================================================


def create_positive_frequency_vector(Fs: Union[float, int], N: int) -> np.ndarray:
    """
        Create positive frequency vector.

        Parameters:
        -----------
        - `Fs` (float,int): Sampling frequency of the signal (in Hz)
        - `N `(int): Length of the signal

        Returns:
        -----------
        - `frequencies` (np.ndarray): 1D array of frequencies ranging from 0 to the Nyquist frequency by Fs/2 step.
    """
    # Calculate the frequency resolution
    freq_resolution = Fs / N

    # Create the frequency vector from 0 Hz to Fs/2
    frequencies = np.linspace(0, Fs/2, N//2 + 1)

    return frequencies


def compute_signal_time_dsps(signal: np.ndarray, sample_rate: Union[float, int]) -> Tuple[dict, dict, dict, dict]:
    """
    Computes the PSD of a signal using 3 different methods (via FFT, via Scipy's periodogram and welch functions).

    Parameters:
    -----------
        - `signal` (np.ndarray): 1D array of amplitudes
        - `sample_rate` (float,int): sampling rate of the signal

    Returns:
    -----------
        - `time_signal` (dict): Dictionary containing signal's timepoints and amplitudes as ndarray under key1 "time_vector" and key2 "amplitudes".
        - `PSD_fft` (dict) : Dictionary containing signal's DSP results computed via FFT: frequencies and amplitudes as ndarray under key1 "frequencies" and key2 "psds".
        - `PSD_p` (dict) : Dictionary containing signal's DSP results computed via periodogram: frequencies and amplitudes as ndarray under key1 "frequencies" and key2 "psds".
        - `PSD_w` (dict) : Dictionary containing signal's DSP results computed via welch: frequencies and amplitudes as ndarray under key1 "frequencies" and key2 "psds".
    """
    N = len(signal)
    print("N: ", N)
    duration = N/sample_rate
    print("duration: ", duration)
    time_vector = np.arange(0, duration, 1/sample_rate)
    print("time_vector shape: ", time_vector.shape)

    # compute FFT of the signal
    signal_fft = np.fft.fft(signal)
    # signal_frequency_vector = np.fft.fftfreq(N, 1/sample_rate)
    signal_frequency_vector = create_positive_frequency_vector(
        Fs=sample_rate, N=N)
    freq_vector_len = len(signal_frequency_vector)

    # signal_frequency_vector = np.arange(0,(sample_rate//2)+freq_res,freq_res)
    print(
        f"signal_frequency_vector before crop len:{len(signal_frequency_vector)},half_val: {signal_frequency_vector[-(freq_vector_len//2)]}")

    # Only keep the positive frequencies and associated amplitudes
    """
    signal_frequency_vector = signal_frequency_vector[:(
        ((freq_vector_len//2 +1)))]  # +1 due to python intervals
    """
    print(f"signal_frequency_vector last freq : {signal_frequency_vector[-1]}")
    signal_fft = signal_fft[:((N//2)+1)]

    # compute PSD via FFT
    psd_from_fft = (np.abs(signal_fft)**2)/(N*sample_rate)

    # compute PSD via periodogram
    freq1, Pxx_density1 = periodogram(
        signal,  fs=sample_rate, window=boxcar(N), detrend=False)  # type: ignore
    # print(type(psd_from_periodogram))
    freq2, Pxx_density2 = welch(signal, fs=sample_rate, window=hamming(1000),
                                nperseg=1000, noverlap=500, nfft=N, detrend=False,  # type: ignore
                                axis=0)

    # create dictionaries of frequencies with psd results for each method to return
    time_signal = {"time_vector": time_vector, "amplitudes": signal}
    PSD_fft = {"frequencies": signal_frequency_vector, "psds": psd_from_fft}
    PSD_p = {"frequencies": freq1, "psds": Pxx_density1}
    PSD_w = {"frequencies": freq2, "psds": Pxx_density2}
    return time_signal, PSD_fft, PSD_p, PSD_w


def add_inset_zoom(ax: plt.Axes, x_data: tuple, y_data: tuple, zoom_region: tuple):
    """
    Add a child inset axes plot to the given Axes object. Can show multiple overlapping series.
    The child inset axes object inherits the line style of the parent plot `ax`.

    Parameters:
    -----------
        - `ax` (plt.Axes): The axes object to add the inset zoom to.
        - `x_data` (tuple of 1D array or array-like): x-coordinates of the data points.
        - `y_data` (tuple of 1D array or array-like): y-coordinates of the data points.
            If a single array is provided, it represents a single series.
            If a tuple of arrays is provided, array represents a separate series.
        - `zoom_region` (tuple): The region to be shown in the zoomed inset in the format (x1, x2, y1, y2).

    Returns: 
    -----------
        - axins object
    """
    # axins = zoomed_inset_axes(ax, zoom=zoom_factor, loc='upper right')
    axins = inset_axes(ax, width=2, height=0.7, loc='upper right')

    # Get the lines settings from the main plot to inherit them
    lines = ax.get_lines()
    if isinstance(y_data, tuple):
        # if multiple series provided
        for xserie, yserie, line in zip(x_data, y_data, lines):
            axins.plot(xserie, yserie, color=line.get_color(), linestyle=line.get_linestyle(),
                       linewidth=line.get_linewidth())

    else:
        line = lines[0]
        # if single series provided
        axins.plot(x_data, y_data, color=line.get_color(), linestyle=line.get_linestyle(),
                   linewidth=line.get_linewidth())

    x1, x2, y1, y2 = zoom_region
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none",
               ec="0.5", lw=1.5, linestyle="--")
    axins.tick_params(axis='both', which='both',
                      labelleft=True, labelbottom=True)
    axins.grid(visible=True, linestyle='--', linewidth=0.5)


def plot_single_signal_time_dsps(fig_number: int, signal: np.ndarray, sample_rate: Union[float, int], fig_title: str, external_results: Optional[str] = None):
    """
    Plots a single time signal alongside its 3 PSDs.

    Calls compute_signal_time_dsps() and plots the results as figure of 4 subplots (lines).
    If `external_results` specified, calls `import_psd_results2() `to superimpose the external results to each corresponding PSD subplot and add an inset zoom to check differences.

    Returns a dictionary `psd_results` containing two dictionaries (`psd_python_results`, `psd_matlab_results`), each containing the PSD results (from fft,periodogram,welch) of the desired signal.
        For both, results are stored under respective keys (key1,key2,key3)=("PSD_FFT","PSD_P","PSD_W")

    Parameters:
    ----------
        - `fig_number` (int): Number of the figure.
        - `signal` (np.ndarray): 1D array of amplitudes.
        - `sample_rate` (float,int): sampling rate (in Hz).
        - `fig_title` (str): Title of the figure.
        - `external_results` (str): external psd results filename with extension.csv (cf function import_psd_results2()).
            - External results must be an array of 6 columns arranged by two, as 3*(frequenceies,psds), for each PSD calculation method.

    Returns:
    ----------
        `psd_results` (dict) : Dictionary of two key:value pairs containing a signal's DSP results (FFT,periodogram,welch).
            key1:"Python_PSD_results"; value1: Dictionary of results from `compute_signal_time_dsps()` function
            key2:"Matlab_PSD_results"; value1: Dictionary of results from `import_psd_results2()` function
    """
    # compute the PSDs of a signal using 3 different methods
    time_signal, PSD_fft, PSD_p, PSD_w = compute_signal_time_dsps(
        signal=signal, sample_rate=sample_rate)
    psd_python_results = {'PSD_FFT': PSD_fft, 'PSD_P': PSD_p, 'PSD_W': PSD_w}

    # Show the time signal and the 3 different results of the PSD
    figure, axis = plt.subplots(4, figsize=(
        10, 7), layout="constrained", num=fig_number)
    figure.suptitle(fig_title + " :\n Time-signal and DSPs")

    # plot time signal
    axis[0].plot(time_signal["time_vector"], time_signal["amplitudes"], "-k")
    # axis[0].set_title('Time signal')
    axis[0].set_ylabel("Amplitude(µV)")
    axis[0].set_xlabel("time(s)")
    # axis[0].set_xlim(0)
    axis[0].grid()

    # plot signal's DSP via FFT
    axis[1].plot(PSD_fft["frequencies"], PSD_fft["psds"], label="Python")
    # axis[1].set_title('PSD from FFT')
    # axis[1].set_xlim(0)
    axis[1].set_ylabel("PSD from \n FFT (µV²/Hz)")
    axis[1].set_xlabel("Frequency (Hz)")
    axis[1].grid()

    # plot signal's DSP via periodogramm
    axis[2].plot(PSD_p["frequencies"], PSD_p["psds"], label="_Python")
    # axis[2].set_title('PSD from periodogramm (µV²/Hz)')
    # axis[2].set_xlim(0)
    axis[2].set_ylabel("PSD from \n periodogramm \n (µV²/Hz)")
    axis[2].set_xlabel("Frequency (Hz)")
    axis[2].grid()

    # plot signal's DSP via scipy.signal.welch
    axis[3].plot(PSD_w["frequencies"], PSD_w["psds"], label="_Python")
    # axis[3].set_title('DSP')
    # axis[3].set_xlim(0)
    axis[3].set_ylabel("PSD signal.welch \n (µV²/Hz)")
    axis[3].set_xlabel("Frequency (Hz)")
    axis[3].grid()

    # Superimpose to each PSD subplot other results (ex from matlab)
    if external_results is not None:
        PSD_fft_external, PSD_p_external, PSD_w_external = import_psd_results2(
            psd_results_file_name=external_results)
        psd_matlab_results = {'PSD_FFT': PSD_fft_external,
                              'PSD_P': PSD_p_external, 'PSD_W': PSD_w_external}
        print(
            f"len psd matlab fft:{len(psd_matlab_results['PSD_FFT']['frequencies'])}")

        axis[1].plot(PSD_fft_external["frequencies"],
                     PSD_fft_external["psds"], '--r', label="Matlab")
        axis[2].plot(PSD_p_external["frequencies"],
                     PSD_p_external["psds"], '--r', label="_Matlab")
        axis[3].plot(PSD_w_external["frequencies"],
                     PSD_w_external["psds"], '--r', label="_Matlab")

        add_inset_zoom(ax=axis[1], x_data=(PSD_fft["frequencies"], PSD_fft_external["frequencies"]), y_data=(PSD_fft["psds"], PSD_fft_external["psds"]),
                       zoom_region=(0, 30, 0, np.max(np.maximum(PSD_fft_external["psds"], PSD_fft["psds"]))))
        add_inset_zoom(ax=axis[2], x_data=(PSD_p["frequencies"], PSD_p_external["frequencies"]), y_data=(PSD_p["psds"], PSD_p_external["psds"]),
                       zoom_region=(0, 30, 0, 10))
        add_inset_zoom(ax=axis[3], x_data=(PSD_p["frequencies"], PSD_w_external["frequencies"]), y_data=(PSD_w["psds"], PSD_w_external["psds"]),
                       zoom_region=(0, 30, 0, 6))
        psd_results = {'Python_PSD_results': psd_python_results,
                       'Matlab_PSD_results': psd_matlab_results}

    else:
        add_inset_zoom(ax=axis[1], x_data=PSD_fft["frequencies"], y_data=PSD_fft["psds"],
                       zoom_region=(0, 40, 0, np.max(PSD_fft["psds"])))
        psd_results = {'Python_PSD_results': psd_python_results}

    figure.legend(title="Results source", loc="upper right")
    return psd_results


def plot_multi_signal_time_dsps(multi_channel_signals: np.ndarray, sample_rate: float,
                                channels_dict: dict, selected_channel_numbers: np.ndarray, input_signal_filename: str):
    """
    Plots multiple time signals alongside their respective 3 PSDs.
    Based on the plot_single_signal_time_dsps() which is repeated for each selected channel.
        If multiple channels (n) are selected, plot_single_signal_time_dsps() is called n times.
    Calls list_matlab_psd_results_filenames() to list external psd results filename of selected channels.

    Returns a dictionary containing for each electrodes the results of plot_single_signal_time_dsps() function

    Parameters:
    ----------
        multi_channel_signals(np.ndarray): 2D array of amplitudes
        sample_rate (int): sampling rate (in Hz)
        channels_dict(dict): dictionary of channels names
        selected_channel_numbers(np.ndarray): 1D array of channel numbers to study
        input_signal_filename(str): name of the input file containing time-signals (used also to retrieve matlab results)

    Return:
    ----------
        electordes_results_dict (dict) : Dictionary of n key:value pairs containing n signal's (python and matlab) DSP results (FFT,periodogram,welch).
            keyi:"channel_name_i" ; valuei:psd_results
            where psd_results(dict) is output of plot_single_signal_time_dsps().
    """
    selected_channel_indexes = selected_channel_numbers-1

    # For each selected electrode, determine the filenames of the appropriate external (psd) results
    MATLAB_PSD_results_filenames_list = list_matlab_psd_results_filenames(input_signal_filename=input_signal_filename,
                                                                          channels_dict=channels_dict,
                                                                          selected_channel_numbers=selected_channel_numbers)
    electordes_results_dict = {}
    for iter, (channel_num, channel_index, matlab_results_filename) in enumerate(zip(selected_channel_numbers, selected_channel_indexes, MATLAB_PSD_results_filenames_list)):
        channel_name = f"Channel_{str(channel_num)}_{channels_dict['Channel_'+str(channel_num)]}"
        print("channel name:", channel_name)
        figure_title = f"{input_signal_filename}\n {matlab_results_filename} \n {channel_name}"
        print(figure_title)
        # select correct signal
        channel_i_signal = multi_channel_signals[:, channel_index]
        electrode_i_psd_results = plot_single_signal_time_dsps(fig_number=iter, signal=channel_i_signal,
                                                               sample_rate=sample_rate, fig_title=figure_title,
                                                               external_results=matlab_results_filename)
        key = f"{channel_name}"
        value = electrode_i_psd_results
        # Ajoute la paire clé-valeur au dictionnaire
        electordes_results_dict[key] = value
    return electordes_results_dict

# =============================================================================
############################ Generate test signal  ############################
# =============================================================================


def generate_sine_wave(amplitude, frequency, duration, change_time, new_amplitude, sample_rate):
    """
    Generate a sine wave that change amplitude after a specific amount of time.

    Parameters:
        amplitude (float): Signal's original amplitude (A.U).
        frequency (float): Signal's frequency (in Hz).
        duration (float): Duration of the signal (in seconds).
        change_time (float): Time  of amplitude change (in seconds).
        new_amplitude (float): Signal's new amplitude (A.U).
        sample_rate (float): Sampling rate of the signal (in Hz). 

    Returns:
        t (ndarray): Array of timepoints (in seconds).
        signal (ndarray): Array of signal's amplitudes (A.U)
    """
    """    t = np.linspace(0, duration, num=int(duration*sample_rate))
    signal = amplitude * np.cos(2 * np.pi * frequency * t)
    """

    # t = np.arange(0, (sample_rate*duration), (1/sample_rate))
    t = np.linspace(0, duration, num=int(duration*sample_rate), endpoint=False)
    signal = amplitude * np.sin(2 * np.pi * frequency * t)
    signal2 = new_amplitude * np.sin(2 * np.pi * frequency * t)

    # Change amplitude after a certain time
    change_index = int(change_time * sample_rate)
    signal[change_index:] = signal2[change_index:]

    return t, signal
# =============================================================================
######################## Compute average ERSP on blocks  ######################
# =============================================================================


def compute_average_ratio_for_event_on_blocks_for_all_electrodes(mat3d):
    """
    # compute the average densities among trials of a block and returns 3d arrays of 1 column (average), PSD values, 8
    # returns array1{block1_event_typex_averagedPSDs,dim=[averaged_ratios_block1_eventx,electrodei]},array2{block2_event_typex_averagedPSDs,dim=[averaged_ratios_block2_eventx,electrodei]]}

    inputs: numpy.ndarray(3D)
    outputs: numpy.ndarray(2D),numpy.ndarray(2D) as mean_per_row_per_plane_test_block1, mean_per_row_per_plane_test_block2
    """
    mean_per_row_per_plane_blocks = []
    mean_per_row_per_plane_test_block1 = []
    mean_per_row_per_plane_test_block2 = []
    # Boucle sur les plans (matrices 2D) de la matrice 3D
    for i in range(mat3d.shape[0]):
        current_plane = mat3d[i]
        mean_per_row_per_plane_test_block1.append(
            np.mean(current_plane[:, :3], axis=1))
        mean_per_row_per_plane_test_block2.append(
            np.mean(current_plane[:, 3:], axis=1))
    mean_per_row_per_plane_test_block1 = np.column_stack(
        mean_per_row_per_plane_test_block1)
    mean_per_row_per_plane_test_block2 = np.column_stack(
        mean_per_row_per_plane_test_block2)

    return mean_per_row_per_plane_test_block1, mean_per_row_per_plane_test_block2

# =============================================================================
############################ New compute lagged PSD  ##########################
# =============================================================================


def get_segment_coordinates2(reference_index: int, segment_time_length, sample_rate) -> dict:
    """
    Gets the coordinates of segments of specified time length on each side of a reference index.

    Returns 3 coordinates defining 2 segments:
        (Lower end, reference_index) and (reference_index,higher_end)

    Application: Allows to retieve the coordinates of the 2 segments surrounding a marker :
        ie:1s before the marker and 1s after the marker.

    Parameters:
    -----------
        - `reference_index` (int): sample index used as reference to define a segment
        - `segment_time_length` (int): expressed in seconds
        - `sample_rate` (int): Sample rate

    Returns:
    -----------
        - `segments_coordinates` (dict): coordinates of the segments as indexes
            key1:"reference_end"
            key2:"lower_end"
            key3: "higher_end"
    """

    # Length of the segment expressed in number of points
    nperseg = segment_time_length*sample_rate

    lower_end = reference_index - nperseg
    higher_end = reference_index + nperseg
    reference_end = reference_index
    # print("lower end: ", lower_end, ",reference: ",reference_end, "higher end: ", higher_end)

    segments_coordinates = {"reference_end": reference_end,
                            "lower_end": lower_end, "higher_end": higher_end}
    return segments_coordinates


def get_signal_segement2(signal: np.ndarray, lower_end: int, higher_end: int) -> Tuple[np.ndarray, int]:
    """
    Get a segment of a given signal.
        The segment to extract is defined by its coordinates (lower_end,higher_end) expressed as the corresponding indexes of the signal array.
            Note: array slicing already takes into account python [`lower_end`,`higher_end`+1]

    Parameters:
    -----------
        - `lower_end` (int): index of the lower end of the segment
        - `higher_end` (int): index of the higher end of the segment

    Retruns:
    -----------
        - `signal_segment` (np.ndarray): signal segment data
        - `signal_length` (int): length of original segment
    """
    signal_length = len(signal)
    signal_segment = signal[lower_end:(higher_end+1)]  # cf python slicing
    segment_length = len(signal_segment)
    return signal_segment, signal_length


def extract_data_epochs(signal: np.ndarray, sample_rate: float, markers_labels_times: dict,
                        select_events: tuple, epoch_limits: tuple[float, float]) -> dict:
    """
    Extract epochs from a signal.

    Parameters:
    -----------
        - `signal` (np.array): 1D array of samples
        - `srate` (float): sampling rate of the signal
        - `marker_labels_times` (dict): Dictionary of 1D arrays of marker information under keys ("markers_timestamp_indices","markers_timestamps", "marker_labels")
        - `select_events` (tuple): tuple of selected event types ex (111,100)
        - `epoch_limits` (tuple): 2 element tuple specifying the start and end of the epoch relative to the event (in seconds). 
            ex1: (0,4) - From 0 sec before to 4 sec after the time-locking event.
            ex2: (1,2) - From 1 sec before to 2 sec after the time-locking event.
            ex3: (-1,2) - From 1 sec before to 2 sec after the time-locking event.


    Returns:
    -----------
        - `epoched_signals` (dict): Dictionary of epoched signals for each marker type
            Keys (str): "label_markertype"
            values (np.ndarray): 2D array of signal data epochs arranged as a column per event

    Examples:
    -----------
    ## Create a 1D array of signals

    ### Define the sample rate and duration
    srate= 12  # samples per second
    duration = 8

    ### Calculate the total number of samples
    num_samples = int(srate * duration)

    ### Generate a time array with equally spaced time points
    times = np.linspace(0, duration, num_samples, endpoint=False)

    ### Generate a 2d array of samples  composed of 1 signals (columns)
    samples = np.random.uniform(low=-20, high=20, size=(num_samples,1))

    ## Create a 2d array of marker timestamps and labels
        labels = [100,111,100,111,100,111]
        timestamps = [1.2,2.2,3.2,4.2,5.2,6.2]
        marker_timestamps_labels = create_marker_times_labels_array2(marker_labels=labels,marker_time_stamps=timestamps)

    ## Find the nearest sample timestamp to each marker
        nearest_markers_array = nearest_timestamps_array_finder(signal_times_stamps=times,markers=marker_timestamps_labels)
        print(nearest_markers_array["markers_timestamps"])

    ## Epoch the signals at once
    epoch_limits=(0,1)
    eeg_signals_epoched=extract_data_epochs(signals=samples,sample_rate=srate,markers_labels_times=nearest_markers_array,
                                                         select_events=(100,111),epoch_limits=epoch_limits)

    ## Note: negative first epoch limit means before the event, postive means after
        ex1: (0,4) - From 0 sec before(/after) to 4 sec after the time-locking event.
        ex2: (1,2) - From 1 sec after to 2 sec after the time-locking event.
        ex3: (-1,2) - From 1 sec before to 2 sec after the time-locking event.

    """
    # convert the marker_labels_times dictionary to a 2D array
    array_markers_labels_times = np.column_stack(
        list(markers_labels_times.values()))

    # create the time vector of the entire signal
    signal_times = np.arange(0, len(signal))*(1/sample_rate)

    # convert the epoch limits in number of points
    n_points_before = epoch_limits[0]*sample_rate
    n_points_after = epoch_limits[1]*sample_rate

    print(
        f"Epoch limits relative to events (in sec): start: {epoch_limits[0]}s - end: {epoch_limits[1]}s")
    print(
        f"Epoch limits relative to events (in samples): n_points_before_marker: {n_points_before} - n_points_after_marker: {n_points_after}")

    epoched_signals = {}
    for event in select_events:
        print(f"Event type : {event} ------------ epochs :")
        signal_segments = []
        time_segments = []
        for row in array_markers_labels_times:
            if row[2] == event:
                marker_index = row[0]
                # calculate the coordinate of the segments
                first_seg_coord = int(marker_index+n_points_before)
                second_seg_coord = int(marker_index+n_points_after)
                print(
                    f"first_seg_coord: {first_seg_coord} - second_seg_coord: {second_seg_coord}")
                # extract the segments and list them
                signal_segment, signal_len = get_signal_segement2(
                    signal=signal, lower_end=first_seg_coord, higher_end=second_seg_coord)
                time_segment = signal_times[first_seg_coord:second_seg_coord+1]
                signal_segments.append(signal_segment)
                time_segments.append(time_segment)
        # stack segments as columns in a 2D array
        signal_segments = np.column_stack(signal_segments)
        time_segments = np.column_stack(time_segments)
        print(f"signal_segments shape: {signal_segments.shape}")
        print(f"time_segments shape: {time_segments.shape}")
        # store the stacked segments under key value corresponding to event label
        key = f"label_{event}"
        epoched_signals[key] = {
            "signal_segments": signal_segments, "time_segments": time_segments}
    return epoched_signals


def compute_welch_estimation_on_segment2(segment: np.ndarray, sample_rate: float, nfft: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute PSD using welch method on a segment.
        Relies on the scipy.welch function, which is parametered to divide the given signal into 4 segments, and use an overlap of 50%
        For given signal of length N (samples), the frequency resolution will be Fr=Sample_rate/(N/4). For Fr=Fe/N, argument `nfft`=N must be specified .

    Parameters:
    -----------
        - `segment` (np.ndarray): samples of signal. 
        - `sample_rate` (float) : sampling rate
        - `nfft` (int): Length of the FFT used, for zero padding welch.

    Retruns:
    -----------
        - `freqs` (np.ndarray): array of welch frequencies.
        - `Pxx_density` (np.ndarray): array of welch power densities.
    """

    segment_length = segment.shape[0]  # number of points in the segment
    sub_segment_length = segment_length/4
    # split the segment in two sub segments with overlap of 50%
    """

    freqs, Pxx_density = welch(segment, fs=sample_rate,
                               window="hann",
                               nperseg=sub_segment_length, detrend=False,
                               noverlap=sub_segment_length//2, nfft=None, axis=0)
    """
    print(f"nfft: {nfft}")
    freqs, Pxx_density = welch(segment, fs=sample_rate,
                               window="hann", nperseg=sub_segment_length,
                               detrend=False, nfft=nfft,  # type: ignore
                               noverlap=sub_segment_length//2, axis=0)
    return freqs, Pxx_density


def compute_psds_for_each_epoch(epochs: np.ndarray, sample_rate: float, nfft: Optional[int] = None) -> dict:
    """
    Computes the the PSD of multiple signal epochs.

    Parameters:
    ----------
        - `epochs` (np.ndarray): 2D array of signals, each column represents a signal
        - `sample_rate` (float): Sampling rate of the signals
        - `nfft` (int): Length of the FFT used, for zero padding welch

    Returns:
    ----------
        - `psd_results_all_epochs` (dict): Dictionary of PSD estimation results:
            "PSD_frequencies": 2D array of frequency results, each column corresponds to a signal\n
            "PSD_magnitudes" : 2D array of PSD magnitudes results, each column corresponds to a signal\n
    """
    print(f"segments shapes to psd - {epochs.shape}")
    frequencies = []
    Pxx_densities = []
    # compute the PSD estimation for each trial
    for epoch in epochs.T:
        freqs, Pxx_density = compute_welch_estimation_on_segment2(
            segment=epoch, sample_rate=sample_rate, nfft=nfft)
        frequencies.append(freqs)
        Pxx_densities.append(Pxx_density)

    Pxx_densities = np.column_stack(Pxx_densities)
    frequencies = np.column_stack(frequencies)
    psd_results_all_epochs = {
        "PSD_frequencies": frequencies, "PSD_magnitudes": Pxx_densities}

    return psd_results_all_epochs


def compute_averaged_psds_over_trials(trials: np.ndarray, sample_rate: float, nfft: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the the PSD of several sample trials and averages their results.

    Parameters:
    ----------
        - `trials` (np.ndarray): 2D array of trials where each column represents the signal of a trial
        - `sample_rate` (float): Sampling rate of the signal
        - `nfft` (int): Length of the FFT used, for zero padding welch

    Returns:
    ----------
        - `mean_frequencies` (np.ndarray): 1D array of PSD estimation frequency results
        - `mean_pxx_densities` (np.ndarray): 1D array of averaged PSD estimation power results
    """
    psd_results_all_epochs = compute_psds_for_each_epoch(
        epochs=trials, sample_rate=sample_rate, nfft=nfft)

    # average the PSDs
    # equals every frequency column of the frequencies array anyway
    mean_frequencies = np.mean(
        psd_results_all_epochs["PSD_frequencies"], axis=1)
    mean_pxx_densities = np.mean(
        psd_results_all_epochs["PSD_magnitudes"], axis=1)

    return mean_frequencies, mean_pxx_densities


def signal_comparison(signal_1: np.ndarray, signal_2: np.ndarray, sample_rate: float, labels: tuple[str, str]):
    """
    Displays two time signals and their respective PSD estimation using welch method.
        To compute the PSD estimates, `compute_psds_for_each_epoch()` is called.

    Parameters:
    ----------
        - `signal_1` (np.ndarray): 2D array with col_1:times,col_2=samples
        - `signal_2` (np.ndarray): 2D array with col_1:times,col_2=samples
        - `sample_rate` (float): signals sampling rate
        - `labels` (tuple): length 2 tuple of signals labels (label_signal_1,label_signal_2)
    Returns:
    ----------
        - `None` (bool): This function does not return a value. It shows a figure.

    """
    signals_to_compare = np.column_stack((signal_1[:, 1], signal_2[:, 1]))
    print(f"signals to compare shape - {signals_to_compare.shape}")
    N = len(signals_to_compare)
    signals_psds = compute_psds_for_each_epoch(
        epochs=signals_to_compare, sample_rate=sample_rate, nfft=N)

    figure, axis = plt.subplots(3, 1, figsize=(15, 7), layout="constrained")
    figure.suptitle("Signal comparison")

    axis[0].set_title(f"Time signal 1: {labels[0]}")
    axis[0].set_ylabel("Amplitudes (µV)")
    axis[0].plot(signal_1[:, 0], signal_1[:, 1], color="blue")

    axis[1].set_title(f"Time signal 2: {labels[1]}")
    axis[1].set_xlabel("Time (s)")
    axis[1].set_ylabel("Amplitudes (µV)")
    axis[1].plot(signal_2[:, 0], signal_2[:, 1], color="red")

    axis[2].set_title("Epochs PSDs")
    axis[2].set_xlabel("Frequencies (Hz)")
    axis[2].set_ylabel("Power (µV²/Hz)")
    axis[2].plot(signals_psds["PSD_frequencies"][:, 0],
                 signals_psds["PSD_magnitudes"][:, 0], label=f'{labels[0]}', color='blue')
    axis[2].plot(signals_psds["PSD_frequencies"][:, 1],
                 signals_psds["PSD_magnitudes"][:, 1], label=f'{labels[1]}', color='red')
    axis[2].set_xlim(0, 60)
    axis[2].legend()
    plt.show()


def compute_welch_on_a_signal_before_each_marker(signal: np.ndarray, sample_rate: float, markers_array: np.ndarray,
                                                 segment_duration: float, nfft: Optional[int] = None) -> dict:
    """
    Computes the DSP estimations of all segments of a signal.

    The signal has as many segments as it has of markers. Each segment streches over the specified segment duration and ends at a marker.
    The DSPs are then computed BEFORE each marker

    Notes: 
    -----------
        - Uses the `markers_array` indexes to define the segments.
        - For opposite operation, see the complementary function `compute_welch_on_a_signal_after_each_marker()`.


    Parameters:
    -----------
        - `signal` (np.ndarray): 1D array of signal samples.
        - `sample_rate` (float) : Signal sampling rate.
        - `markers_array` (dict): Dictionary of 2D arrays under 3 keys as ("markers_timestamp_indices","markers_timestamps", "marker_labels")
        - `segment_duration` (float): length of the segment on which to compute the DSP estimation.

    Retruns:
    -----------
        - `results` (dict): Dictionary of PSD estimation results:
            - "freqs" (np.ndarray): array of welch frequencies.
            - "Pxx_density" (np.ndarray): array of welch power densities.
    """
    N = len(signal)
    freqs_for_all_markers = []
    PSDs_for_all_markers = []
    for marker in markers_array["markers_timestamp_indices"]:

        segment_coordinates = get_segment_coordinates2(
            reference_index=marker, segment_time_length=segment_duration, sample_rate=sample_rate)

        higher_end = int(segment_coordinates["reference_end"])
        lower_end = int(segment_coordinates["lower_end"])

        signal_segment, full_signal_length = get_signal_segement2(
            signal=signal, lower_end=lower_end, higher_end=higher_end)
        freqs, Pxx_density = compute_welch_estimation_on_segment2(
            signal_segment, sample_rate=sample_rate, nfft=nfft)
        freqs_for_all_markers.append(freqs)
        PSDs_for_all_markers.append(Pxx_density)

    freqs_for_all_markers = np.column_stack(freqs_for_all_markers)
    PSDs_for_all_markers = np.column_stack(PSDs_for_all_markers)

    return {"PSD_frequencies": freqs_for_all_markers, "PSD_magnitudes": PSDs_for_all_markers}


def compute_welch_on_a_signal_after_each_marker(signal: np.ndarray, sample_rate: float, markers_array: np.ndarray,
                                                segment_duration: float, nfft: Optional[int] = None) -> dict:
    """
    Computes the DSP estimations of all segments of a signal.

    The signal has as many segments as it has of markers. Each segment starts at a marker and streches over the specified segment duration.
    The DSPs are then computed AFTER each marker

    Notes: 
    -----------
        - Uses the `markers_array` indexes to define the segments.
        - For opposite operation, see the complementary function `compute_welch_on_a_signal_before_each_marker()`.

    Parameters:
    -----------
        - `signal` (np.ndarray): 1D array of signal samples.
        - `sample_rate` (float) : Signal sampling rate.
        - `markers_array` (dict): Dictionary of 2D arrays under 3 keys as ("markers_timestamp_indices","markers_timestamps", "marker_labels")
        - `segment_duration` (float): length of the segment on which to compute the DSP estimation.

    Retruns:
    -----------
        - `results` (dict): Dictionary of PSD estimation results:
            - "freqs" (np.ndarray): array of welch frequencies.
            - "Pxx_density" (np.ndarray): array of welch power densities.
    """

    N = len(signal)
    freqs_for_all_markers = []
    PSDs_for_all_markers = []
    for marker in markers_array["markers_timestamp_indices"]:

        segment_coordinates = get_segment_coordinates2(
            reference_index=marker, segment_time_length=segment_duration, sample_rate=sample_rate)

        higher_end = int(segment_coordinates["higher_end"])
        reference_end = int(segment_coordinates["reference_end"])

        signal_segment, full_signal_length = get_signal_segement2(
            signal=signal, lower_end=reference_end, higher_end=higher_end)
        freqs, Pxx_density = compute_welch_estimation_on_segment2(
            signal_segment, sample_rate=sample_rate, nfft=nfft)
        freqs_for_all_markers.append(freqs)
        PSDs_for_all_markers.append(Pxx_density)

    freqs_for_all_markers = np.column_stack(freqs_for_all_markers)
    PSDs_for_all_markers = np.column_stack(PSDs_for_all_markers)

    return {"PSD_frequencies": freqs_for_all_markers, "PSD_magnitudes": PSDs_for_all_markers}


def extract_data_epochs_from_all_signals(signals: np.ndarray[Any, np.dtype[Union[np.float32, np.float64, np.int32]]], sample_rate: Union[float, int],
                                         markers_labels_times: dict, select_events: tuple,
                                         epoch_limits: tuple[float, float]) -> dict:
    """
    Extracts data epochs from multiple signals at once.

    Based on extract_data_epochs which is called for each signal and stores the epochs as a dictionary where each signal is a key.

    Parameters:
    -----------
        - `signals` (np.ndarray): 2D array of signals where each signal represents a column (axis=1)
        - `sample_rate` (float, int): Signal sampling rate
        - `markers_labels_times` (dict): Dictionary of 1D arrays of marker information under keys ("markers_timestamp_indices","markers_timestamps", "marker_labels")
        - `epoch_limits` (tuple): 2 element tuple specifying the start and end of the epoch relative to the event (in seconds). 

    Returns:
    -----------
        - `epoched_signals` (dict): dictionary of epoched signals for all signals
            - keys (str): `channel_name` ("Channel_i")
            - value (dict): `epochs_dict`
                - key (str): "Epochs"
                - value (dict) : `time_signal_epochs_dict`
                    - key (str): "time_signals"
                    - value(dict) : epoched_signal (<= extract_data_epochs output)
                        - keys(str) : epoch_labels ("label_markertype")
                        - value (dict) : time_signals
                            - keys (str): "signal_segments"|"time_segments"
                            - values (np.ndarray): 2D array of signal data epochs arranged as a column per event

    Examples:
    ----------
    # Create a 2D array of signals
        # Define the sample rate and duration
        srate= 12  # samples per second
        duration = 8
        # Calculate the total number of samples
        num_samples = int(srate * duration)

        # Generate a time array with equally spaced time points
        times = np.linspace(0, duration, num_samples, endpoint=False)

        # Generate a 2d array of samples  composed of 6 signals (columns)
        samples = np.random.uniform(low=-20, high=20, size=(num_samples,6))

    # Create a 2d array of marker timestamps and labels
        labels = [100,111,100,111,100,111]
        timestamps = [1.2,2.2,3.2,4.2,5.2,6.2]
        marker_timestamps_labels = create_marker_times_labels_array2(marker_labels=labels,marker_time_stamps=timestamps)

    # Find the nearest sample timestamp to each marker
        nearest_markers_array = nearest_timestamps_array_finder(signal_times_stamps=times,markers=marker_timestamps_labels)
        print(nearest_markers_array["markers_timestamps"])

    # Epoch the signals at once
    epoch_limits=(0,1)
    eeg_signals_epoched=extract_data_epochs_from_all_signals(signals=samples,sample_rate=srate,markers_labels_times=nearest_markers_array,
                                                         select_events=(100,111),epoch_limits=epoch_limits)

    Note: negative first epoch limit means before the event, postive means after
        ex1: (0,4) - From 0 sec before(/after) to 4 sec after the time-locking event.
        ex2: (1,2) - From 1 sec after to 2 sec after the time-locking event.
        ex3: (-1,2) - From 1 sec before to 2 sec after the time-locking event.

    """

    epoched_signals = {}
    for signal_index in range(signals.shape[1]):
        epochs_dict = {}
        time_signal_epochs_dict = {}
        print(len(signals[:, signal_index]))
        epoched_signal = extract_data_epochs(signal=signals[:, signal_index], sample_rate=sample_rate,
                                             markers_labels_times=markers_labels_times, select_events=(111, 100), epoch_limits=epoch_limits)

        time_signal_epochs_dict["time_signals"] = epoched_signal
        epochs_dict["Epochs"] = time_signal_epochs_dict

        channel_name = f"Channel_{signal_index+1}"
        epoched_signals[channel_name] = epochs_dict

    return epoched_signals


def compute_psds_for_each_epoch_all_signals(input_dict: dict, sample_rate: Union[float, int], nfft: Optional[int] = None) -> dict:
    """
    Computes the PSDs of each epoch for all signals

    Parameters:
    ----------
        - `input_dict`(dict): Dictionary of epoched signals (output of `extract_data_epochs_from_all_signals()`)
        - `sample_rate`(float,int): Sampling rate of the signals.
        - `nfft`(float,int): length of the FFT (if zero padded fft wanted), Defaults to None

    Retruns:
    -----------
        Input_dict with new "PSDs" key in Epochs dictionary alongside "time_signals" as output_dict
        - `output_dict` (dict): Dictionary of epoched signals for all signals and their PSDs
            - keys (str): `channel_name` ("Channel_i")
            - value (dict): `epochs_dict`
                - key (str): "Epochs"
                - value (dict) : `time_signal_epochs_dict`
                    - keys (str): "time_signals", "PSDs"
                    - values (dict) : epoched_signal (<= extract_data_epochs output), psd_results_all_epochs_of_type_i (<= compute_psds_for_each_epoch output)

    Examples:
    ----------
    # Create a 2D array of signals
        # Define the sample rate and duration
        srate= 12  # samples per second
        duration = 8
        # Calculate the total number of samples
        num_samples = int(srate * duration)

        # Generate a time array with equally spaced time points
        times = np.linspace(0, duration, num_samples, endpoint=False)

        # Generate a 2d array of samples  composed of 6 signals (columns)
        samples = np.random.uniform(low=-20, high=20, size=(num_samples,6))

    # Create a 2d array of marker timestamps and labels
        labels = [100,111,100,111,100,111]
        timestamps = [1.2,2.2,3.2,4.2,5.2,6.2]
        marker_timestamps_labels = create_marker_times_labels_array2(marker_labels=labels,marker_time_stamps=timestamps)

    # Find the nearest sample timestamp to each marker
        nearest_markers_array = nearest_timestamps_array_finder(signal_times_stamps=times,markers=marker_timestamps_labels)
        print(nearest_markers_array["markers_timestamps"])

    # Epoch the signals at once
        epoch_limits = (0,1)
        signals_epoched = extract_data_epochs_from_all_signals(signals=samples,sample_rate=srate,markers_labels_times=nearest_markers_array,
                                                            select_events=(100,111),epoch_limits=epoch_limits)
    # Compute PSD for all epochs of all signals at once
        fft_length=len(signals_epoched["Channel_1"]["Epochs"]["time_signals"]["label_111"]["signal_segments"])
        signals_epochs_psds = compute_psds_for_each_epoch_all_signals(input_dict=signals_epoched,sample_rate=srate,
                                                                        nfft=fft_length)
        signals_epochs_psds["Channel_1"]["Epochs"]["PSDs"]["label_100"]["PSD_magnitudes"]

    Note: By default the length of the fft used to perform dsp estimation is the length of the epoch (nfft=None)
    """
    output_dict = input_dict.copy()

    for channel_name in input_dict:
        psd_results_all_epochs_of_type_i = {}
        # print(f"key1= {channel_name}")
        for epoch_label in input_dict[channel_name]["Epochs"]["time_signals"]:
            # print(f"key3= {epoch_label}")
            # print(input_dict[channel_name]["Epochs"]["time_signals"][epoch_label]["signal_segments"])
            psd_results_all_epochs = compute_psds_for_each_epoch(input_dict[channel_name]["Epochs"]["time_signals"][epoch_label]["signal_segments"],
                                                                 sample_rate=sample_rate, nfft=nfft)
            psd_results_all_epochs_of_type_i[epoch_label] = psd_results_all_epochs
        output_dict[channel_name]["Epochs"]["PSDs"] = psd_results_all_epochs_of_type_i

    return output_dict


# =============================================================================
############################# Signal preprocessing  ###########################
# =============================================================================
def detrend_signals(raw_signals: np.ndarray) -> np.ndarray:
    """
    Remove linear trends from multiple signals.
        Computes the mean of each signal, and substract its mean to each signal

    Parameters:
    -----------
        - `raw_signals` (np.ndarray): 2D Array of raw signals arranged as columns.

    Returns:
    -----------
        - `EEG_amplitudes_centered` (np.ndarray): Array of detrended signals arranged as columns (same shape as raw_signals).
    """

    print(f"input_signals shape:\n {raw_signals.shape}")
    print(f"input_signals mean per signal:\n {np.mean(raw_signals,axis=0)}")
    EEG_amplitudes_centered = raw_signals-np.mean(raw_signals, axis=0)
    return EEG_amplitudes_centered


def rereference_signals(input_signals: np.ndarray) -> np.ndarray:
    """
    Reference multiple signals to average of all signals(Average rereference).
        Computes the whole mean of signals at each times, and substract its mean to each signal


    Parameters:
    -----------
        - `input_signals` (np.ndarray): Array of signals arranged as columns.

    Returns:
    -----------
        - `EEG_amplitudes_rereferenced` (np.ndarray):  Array of signals arranged as columns (shape identical to input_signals).
    """
    print(f"input_signals shape:{input_signals.shape}")
    print(f"input_signals whole mean:{np.mean(input_signals)}")
    mean_vector = np.mean(input_signals)
    print(f"mean_vector shape: {mean_vector.shape}")
    EEG_amplitudes_rereferenced = input_signals-mean_vector

    return EEG_amplitudes_rereferenced


def filter_signal(input_signals: np.ndarray, sample_rate: Union[int, float], order: int,
                  cutofffreq: Union[Tuple[float, float, float], Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies a series of filters on signals.
    The first two input frequencies of the tuple cutofffreq are corrected before use by the adequate filters.

    Parameters:
    -----------
        - `input_signals` (np.ndarray): 2D Array of signals arranged as columns.
        - `sample_rate` (int): Sampling rate of the signals.
        - `order` (int) : Order of the band pass filter. Also required to apply right correction to cutofffreq[0,2]
        - `cutofffreq` (tuple): Cutoff frequencies to be used by filters. 
            - Tuple must be len(cutofffreq) = 2 or 3.
            - Ordered as: `cutofffreq` =(`low_cutoff_freq`,`low_cutoff_freq`,`notch_cutoff_freq`)

    Returns:
    -----------
        - `EEG_Filtered_NOTCH_BP` (ndarray): Array of signals arranged as columns (same shape as input_signals).
        - `freq_test_BP` (ndarray): Frequency vector for verification of the filter response.
        - `magnitude_test_BP` (ndarray): Magnitude vector for verification of the filter response.
    """
    if len(cutofffreq) < 2 or len(cutofffreq) > 3:
        raise ValueError(
            f"cutofffreq tuple length:{len(cutofffreq)} - Input tuple length must be between 2 and 3")
    else:
        LOW_CUTOFF_FREQ_THEORETICAL = 1
        HIGH_CUTOFF_FREQ_THEORETICAL = 1
        if len(cutofffreq) == 3:
            LOW_CUTOFF_FREQ_THEORETICAL, HIGH_CUTOFF_FREQ_THEORETICAL, NOTCH_CUTOFF_FREQ = cutofffreq

        elif len(cutofffreq) == 2:
            LOW_CUTOFF_FREQ_THEORETICAL, HIGH_CUTOFF_FREQ_THEORETICAL = cutofffreq
            NOTCH_CUTOFF_FREQ = None

        # cutoff frequency correction for filtfilt application
        LOW_CUTOFF_FREQ_CORRECTED = filtfilt_cutoff_frequency_corrector(
            order, LOW_CUTOFF_FREQ_THEORETICAL, sample_rate, pass_type="high_pass")

        HIGH_CUTOFF_FREQ_CORRECTED = filtfilt_cutoff_frequency_corrector(
            order, HIGH_CUTOFF_FREQ_THEORETICAL, sample_rate, pass_type="low_pass")

        """        print("LOW_CUTOFF_FREQ_THEORETICAL="+str(LOW_CUTOFF_FREQ_THEORETICAL) +
            ", HIGH_CUTOFF_FREQ_THEORETICAL="+str(HIGH_CUTOFF_FREQ_THEORETICAL))
                print("LOW_CUTOFF_FREQ_CORRECTED="+str(LOW_CUTOFF_FREQ_CORRECTED) +
            ", HIGH_CUTOFF_FREQ_CORRECTED="+str(HIGH_CUTOFF_FREQ_CORRECTED))"""

        print(
            f"LOW_CUTOFF_FREQ_THEORETICAL={LOW_CUTOFF_FREQ_THEORETICAL},HIGH_CU-TOFF_FREQ_THEORETICAL={HIGH_CUTOFF_FREQ_THEORETICAL}")
        print(
            f"LOW_CUTOFF_FREQ_CORRECTED={LOW_CUTOFF_FREQ_CORRECTED},HIGH_CUTOFF_FREQ_CORRECTED={HIGH_CUTOFF_FREQ_CORRECTED}")

        # Filtering on all channels
        if NOTCH_CUTOFF_FREQ is not None:
            # 1-Notch-filter 50Hz [49,51] the centered-rereferenced signal
            print(f"NOTCH_CUTOFF_FREQ={NOTCH_CUTOFF_FREQ}")

            input_signals, freqs_test_NOTCH, magnitudes_test_NOTCH = notch_filter(input_signal=input_signals,
                                                                                  sample_rate=sample_rate,
                                                                                  cutoff_freq=NOTCH_CUTOFF_FREQ,
                                                                                  stop_band_width=2)

        # 2-Then Band-pass filter the signal filtered by notch
        EEG_Filtered_NOTCH_BP, freq_test_BP, magnitude_test_BP = band_pass_filter(input_signal=input_signals,
                                                                                  sample_rate=sample_rate,
                                                                                  low_cutoff_freq=LOW_CUTOFF_FREQ_CORRECTED,
                                                                                  high_cutoff_freq=HIGH_CUTOFF_FREQ_CORRECTED,
                                                                                  filter_order=order)
        print("Filtered signal shape:", np.shape(EEG_Filtered_NOTCH_BP))
    return EEG_Filtered_NOTCH_BP, freq_test_BP, magnitude_test_BP

# =============================================================================
##############################  Motion tracking  ##############################
# =============================================================================


def compute_tangential_speed(coordinates: np.ndarray, sample_rate: float) -> np.ndarray:
    """
    Computes the tangential speed from xy coordinates.

    Parameters:
    ----------
        - `coordinates` (np.ndarray): 2D array containing X and Y mouse positions in first and second columns respectively
        - `sample_rate` (float) : sampling rate of the coordinates

    Returns:
    ----------
        - `vt` (np.ndarray): 1D array instantaneous tangential speeds.

    """
    # sampling period
    Ts = 1/sample_rate

    x_coordinates = coordinates[:, 0]
    y_coordinates = coordinates[:, 1]

    # position difference between two consecutive samples
    dx = np.diff(x_coordinates, prepend=0)
    dy = np.diff(y_coordinates, prepend=0)

    # compute the instantaneous speeds in each direction
    vx = dx/Ts
    vy = dy/Ts

    # compute the total tangential speed
    vt = np.sqrt(vx**2+vy**2)

    return vt


def compute_total_acceleration(accelerations_array: np.ndarray) -> np.ndarray:
    """
    Computes the euclidian norm of the acceleration among multiple axis

    Parameters:
    ----------
        - `accelerations_array` (np.ndarray): 2D array of accelerations, each column represents the accelerations of a given axis.

    Retruns:
    -----------
        - `accelerations_norms` (np.ndarray): 1D array of acceleration norms.
    """
    # Compute the Euclidean norm (magnitude) for each row (acceleration vector)
    accelerations_norms = np.linalg.norm(accelerations_array, axis=1)
    return accelerations_norms

# =============================================================================
############################## Matlab vs Python  ##############################
# =============================================================================


def rms(series1, series2, name: str, units: str = "Units (NA)") -> float:  # litteral formula
    """
    Compute the root mean squared error of two series.
    Returns the value and prints it

    Parameters:
    ----------
        - `series1` (np.ndarray): 1D series to compare
        - `series2` (np.ndarray): 1D series to compare

    Returns:
    ----------
        - `rms` (float): root mean squared error of two series
    """
    # rms=np.sqrt(((python - matlab) ** 2).mean())

    diff = series1-series2
    squared_diff = diff**2
    mean_squared_diff = np.mean(squared_diff)
    rms = np.sqrt(mean_squared_diff)
    print(name+" = ", rms, " (µV²/Hz)")
    return rms


def cv_percent(series1: np.ndarray, series2: np.ndarray) -> np.ndarray:
    """
    [UNUSED]
    Compute coeffcient of variation (CV) of two series.
    Returns the result expressed in (%)

    Parameters:
    ----------
        - `series1` (np.ndarray): 1D series to compare
        - `series2` (np.ndarray): 1D series to compare

    Returns:
    ----------
        - `cv` (np.ndarray): coeffcient of variation  of two series
    """
    diff = series1-series2
    # diff=np.std([series1,series2])

    squared_diff = diff**2
    var = np.sqrt(squared_diff)
    mean = (series1+series2)/2
    cv = (var/mean)*100
    return cv


def abs_distance(series1, series2) -> np.ndarray:
    """
    Compute the absolute differences of two series elementwise.
    Returns a series of absolute differences.

    Parameters:
    ----------
        - `series1` (np.ndarray): 1D series to compare
        - `series2` (np.ndarray): 1D series to compare

    Returns:
    ----------
        - `absolute_diff` (np.ndarray): array of absolute differences.
    """
    diff = series1-series2
    absolute_diff = abs(diff)
    return (absolute_diff)


def list_matlab_psd_results_filenames(input_signal_filename: str, channels_dict: dict[str, str], selected_channel_numbers: np.ndarray) -> list:
    """
    Lists the expected csv filenames of the matlab psd results for the selected channels.

    Parameters:
    ----------
        - `input_signal_filename` (str): name of the input eeg data file (with its csv extension)
        - `channels_dict` (dict): Dictionary of channel numbers as keys (ie Channel_x ) and names as values (ie C3)
        - `selected_channel_numbers` (np.ndarray): 1D array of the selected channel numbers (integers)

    Returns:
    ----------
        -`matlab_results_filename_list` (list): dictionary of the frequencies and PSDs estimates from matlabs FFT

    """
    matlab_results_filename_list = []
    # channel_indexes = [x - 1 for x in selected_channel_numbers]
    # old bug fix?
    channel_indexes = selected_channel_numbers-1
    print("selected channels :")
    for (i, y) in zip(selected_channel_numbers, channel_indexes):
        print(f"channel number:{i}, channel index:{y}")
        channel_name = f"Channel_{str(i)}_{channels_dict['Channel_'+str(i)]}"
        print(channel_name)

        # get name of corresponding matlab psd result
        filenamei = f"MATLAB_PSD_res_EEG_{channel_name}_{input_signal_filename }"

        matlab_results_filename_list.append(filenamei)

    print(f"matlab psd results file names: {matlab_results_filename_list}")

    return matlab_results_filename_list


def import_psd_results2(psd_results_file_name: str) -> Tuple[dict, dict, dict]:
    """
    Imports psd data results generated (beforehand) by the matlab script.
    Matlab psd results must be stored in the `STAGE_SIGNAL_PHYSIO/DAT/OUTPUT/Matlab_PSD_Results` folder as csv files to be retrieved.
        Returns 3 dictionaries (for each PSD estimation method) of two key-value pairs : 
            (key1:"frequencies", value:(1D)array of frequencies)\n
            (key2:"psds", value: (1D)array of PSD)

    Parameters:
    ----------
        - `psd_results_file_name` (str) :name of the input eeg data file (with its extension)

    Returns:
    ----------
        - `PSD_fft_results` (dict): dictionary of the frequencies and PSDs estimates from matlabs FFT
        - `PSD_p_results` (dict): dictionary of the frequencies and PSDs estimates from matlabs periodogram function
        - `PSD_w_results` (dict): dictionary of the frequencies and PSDs estimates from matlabs welch function

    """
    # filename="MATLAB_PSD_res_EEG_Channel_5_C3_001_MolLud_20201112_1_c_preprocessed_499.998_Hz"

    filepath = f"./DAT/OUTPUT/Matlab_PSD_Results/{psd_results_file_name}"
    print(filepath, type(filepath))

    matlab_data = np.genfromtxt(filepath, delimiter=';', skip_header=1)

    PSD_fft_results = {
        "frequencies": matlab_data[:, 0], "psds": matlab_data[:, 1]}
    PSD_p_results = {
        "frequencies": matlab_data[:, 2], "psds": matlab_data[:, 3]}
    PSD_w_results = {
        "frequencies": matlab_data[:, 4], "psds": matlab_data[:, 5]}

    return PSD_fft_results, PSD_p_results, PSD_w_results


def export_xdf_eeg_to_csv(xdf_filepath: str, PROCESS_SIGNAL: bool = False) -> str:
    """
    Access the xdf file, finds eeg stream and exports all channels data to csv.

    Parameters:
    ----------
        - `xdf_filepath` (str): filepath to the xdf file.
        - `PROCESS_SIGNAL` (bool): boolean to specify if xdf data must be preprocessed or not before export

    Returns:
    ----------
         - `exportfilename` (str): filename of the exported file
    """
    # import raw data

    # Define xdf file path
    input_filepath = xdf_filepath
    INPUT_FILENAME = os.path.splitext(os.path.basename(input_filepath))[0]
    # path=os.path.normpath("../DAT/Input/001_MolLud_20201112_1_c.xdf")

    print("Input filepath: ", input_filepath)
    print("Input filename: ", INPUT_FILENAME)

    # Loading streams of interest

    # load_xdf returns selected streams as list of streams (ie.dictionary)
    EEG_Stream, EEG_fileheader = pyxdf.load_xdf(
        input_filepath, select_streams=[{'type': 'EEG'}])
    Mouse_markers_Stream, Mouse_markers_header = pyxdf.load_xdf(
        input_filepath, select_streams=[{'type': 'Markers', 'name': 'MouseToNIC'}])

    # in case multiple streams havee been found
    if len(EEG_Stream) and len(Mouse_markers_Stream) != 1:
        raise ValueError("Multiple streams matching type restriction")
    else:
        # access to the only EEG stream of the list
        EEG_Stream = EEG_Stream[-1]
        Mouse_markers_Stream = Mouse_markers_Stream[-1]

    # Get sampling rate
    Srate = EEG_Stream["info"]["effective_srate"]

    # Manage EEG data
    # Get amplitudes of each electrode
    EEG_raw_amplitudes = EEG_Stream["time_series"]

    # EEG channel name definition
    channels_dict = {"Channel_1": "C4",
                     "Channel_2": "FC2",
                     "Channel_3": "FC6",
                     "Channel_4": "CP2",
                     "Channel_5": "C3",
                     "Channel_6": "FC1",
                     "Channel_7": "FC5",
                     "Channel_8": "CP1"}

    # Format EEG times_stamps (in unix time epoch) to seconds relative to the execution time of the recording
    EEG_times = EEG_Stream["time_stamps"]-EEG_Stream["time_stamps"][0]

    # Manage Markers data
    # Select the marker labels
    Markers_labels = Mouse_markers_Stream["time_series"]

    # Markers start with a time relative to the execution time of the recording
    Marker_times = (
        Mouse_markers_Stream["time_stamps"]-EEG_Stream["time_stamps"][0])

    # Stacking maker timestamps and labes as 2D array : [[markers_timesstamps],[markers_labels]]
    Markers_times_labels = np.column_stack((Marker_times, Markers_labels))

    # Prepare data for export
    # Process signals or not?
    print("PROCESS_SIGNAL ? --", PROCESS_SIGNAL)

    if PROCESS_SIGNAL is False:
        print("--Keeping raw signals...")
        EEG_for_export = EEG_raw_amplitudes
        DATA_STATUS = "raw"

    elif PROCESS_SIGNAL is True:
        print("--Processing signals")
        print("Detrending...")
        EEG_amplitudes_centered = detrend_signals(EEG_raw_amplitudes)
        print("Rereferencing...")
        EEG_amplitudes_rereferenced = rereference_signals(
            input_signals=EEG_amplitudes_centered)
        print("Filtering...")
        EEG_amplitudes_centered_filtered, _, _ = filter_signal(input_signals=EEG_amplitudes_rereferenced,
                                                               sample_rate=Srate,
                                                               order=8, cutofffreq=(5, 100, 50))
        EEG_for_export = EEG_amplitudes_centered_filtered
        DATA_STATUS = "prepro"
    else:
        raise ValueError('PROCESS_SIGNAL must be Boolean (True or False)')

    print(f"EEG_for_export shape : {EEG_for_export.shape}")

    # Stack columns as electrodes signals followed by column of timestamps
    amplitudes_times = np.column_stack((EEG_for_export, EEG_times))

    # Export to CSV file
    print(f"--Exporting")

    # Create header for CSV
    header = ', '.join(
        [f"{key}:{value}" for key, value in channels_dict.items()])
    header = header+',time(sec)'
    print("export data header :", header)

    # Create filename
    exportfilename = f"{INPUT_FILENAME}_{DATA_STATUS}_{round(Srate,3)}_Hz.csv"
    exportfilepath = os.path.normpath("DAT/INPUT/"+exportfilename)

    print(f"Input filepath : {input_filepath}")
    print(f"Output filepath : {exportfilepath}")

    # Export
    np.savetxt(exportfilepath, amplitudes_times, delimiter=',',
               header=header, comments='', fmt='%d')
    # np.savetxt(exportfilepath,times_amplitudes,delimiter=",")

    return exportfilename
