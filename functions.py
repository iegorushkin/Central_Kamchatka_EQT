import obspy
import numpy as np
import pandas as pd
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def distances_eqs_stations(eq_data, stations_data, save_flag=0, save_path=""):
    """
    Calculates the distances between earthquakes and seismic stations using 
    the haversine formula for surface distance and Pythagorean theorem for 
    total distance.

    Parameters:
    -----------
    eq_data : str
        Path to the CSV file containing earthquake data. 
        The file should have columns: 'id', 'mag', 'time', 'depth', 
        'latitude', and 'longitude'.
    
    stations_data : str
        Path to the Excel file containing seismic station data. 
        The file should have columns: 'station name', 'elevation', 
        'latitude', and 'longitude'.
    
    save_flag : int, optional (default=0)
        If nonzero, saves the resulting DataFrame to an Excel file.
    
    save_path : str, optional (default="")
        Directory where the output Excel file will be saved if `save_flag`
        is set.

    Returns:
    --------
    results_df : pandas.DataFrame
        A DataFrame containing earthquake-station distances with the 
        following columns:
        - 'eq_id': Earthquake ID
        - 'station_name': Station name
        - 'eq_time': Earthquake time
        - 'eq_mag': Earthquake magnitude
        - 'distance': Total distance (km) between earthquake and station
        - 'eq_lat', 'eq_lon': Earthquake latitude and longitude
        - 'eq_depth': Earthquake depth (km)
        - 'station_lat', 'station_lon': Station latitude and longitude
        - 'station_elev': Station elevation (km)

    Notes:
    ------
    - Earthquake longitudes are adjusted if negative (i.e., converted to 
      0-360° format).
    - Station elevations are converted from meters to kilometers.
    """

    try:
        eqs_df = pd.read_csv(eq_data)[['id', 'mag', 'time', 'depth', 'latitude',
                                      'longitude',]]
        stations_df = pd.read_excel(stations_data)[['station name', 'elevation',
                                                    'latitude', 'longitude']]
    except Exception as e:
        print(e, type(e))
        sys.exit()

    eqs_df.rename(columns={'id': 'eq_id',
                           'mag': 'eq_mag',
                           'time': 'eq_time',
                           'latitude': 'eq_lat',
                           'longitude': 'eq_lon',
                           'depth': 'eq_depth'},
                  inplace=True)
    # Convert EQ longitude if necessary
    eqs_df['eq_lon'] = eqs_df['eq_lon'] \
        .apply(lambda x: x + 360 if x < 0 else x)

    stations_df.rename(columns={'station name': 'station_name',
                                'latitude': 'station_lat',
                                'longitude': 'station_lon',
                                'elevation': 'station_elev'},
                       inplace=True)
    # convert m to km
    stations_df['station_elev'] = stations_df['station_elev'] / 1000

    # Approximate Earth radius (km)
    R = 6371
    # A list for placing intermediate results
    results_list = []

    # Loop for every earthquake
    for i in range(eqs_df.shape[0]):
        # Calculating the horizontal distance using the haversine formula
        # print(i)
        delta_lat = (np.radians(stations_df['station_lat'])
                     - np.radians(eqs_df['eq_lat'][i]))
        delta_lon = (np.radians(stations_df['station_lon'])
                     - np.radians(eqs_df['eq_lon'][i]))
        a = (
             np.sin(delta_lat / 2)**2
             + (np.cos(np.radians(eqs_df['eq_lat'][i]))
                * np.cos(np.radians(stations_df['station_lat']))
                * np.sin(delta_lon / 2)**2)
        )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        dist_surface = R * c
    
        # Calculating the total distance considering the depth of the earthquake
        dist_total = np.sqrt(
            dist_surface**2
            + (eqs_df['eq_depth'][i] + stations_df['station_elev'])**2
        )

        # Creating a table containing, among other things, the distance between
        # the current earthquake and each of the stations under consideration.
        new_columns_df = pd.DataFrame({
            'distance': dist_total.values,
            # NOTE: np.tile()
            'eq_id': np.tile(eqs_df.iloc[i, :]['eq_id'], stations_df.shape[0]),
            'eq_mag': np.tile(eqs_df.iloc[i, :]['eq_mag'], stations_df.shape[0]),
            'eq_time': np.tile(eqs_df.iloc[i, :]['eq_time'], stations_df.shape[0]),
            'eq_lat': np.tile(eqs_df.iloc[i, :]['eq_lat'], stations_df.shape[0]),
            'eq_lon': np.tile(eqs_df.iloc[i, :]['eq_lon'], stations_df.shape[0]),
            'eq_depth': np.tile(eqs_df.iloc[i, :]['eq_depth'], stations_df.shape[0])
            })
        cur_results_df = pd.concat([new_columns_df, stations_df], axis=1)
        # Adding this intermediate DataFrame cur_results_df to the list.
        results_list.append(cur_results_df)

    # Combining the tables from the list into one final table.
    results_df = pd.concat(results_list, axis=0)
    # Rearranging its the columns for a more logical display
    results_df = results_df[['eq_id', 'station_name', 'eq_time', 'eq_mag',
                             'distance', 'eq_lat', 'eq_lon', 'eq_depth',
                             'station_lat', 'station_lon', 'station_elev']]

    # Save the result (if required)
    if save_flag != 0:
        results_df.to_excel(save_path + "earthquakes_stations.xlsx",
                            sheet_name="Main", index=False)

    return results_df


def generate_stream(file_folder='', eq_time=''):
    """
    Generate an ObsPy Stream object by collecting traces from miniSEED files
    in the provided directory (Path object) based on the given earthquake time.

    Parameters:
    file_folder (pathlib.Path object): Path to the folder containing miniSEED
    files.
    eq_time (obspy.UTCDateTime object): Earthquake time.

    Returns:
    obspy.Stream: A Stream object containing the collected traces.
    """
    # Check of some file_folder was provided
    if not file_folder or not eq_time:
        print("You need to provide correct inputs!")

    # Create a list containing file paths
    file_pathlist = [item for item in file_folder.iterdir() if item.is_file()]

    # Collect all traces into 1 stream object
    # 1. Create an empty stream
    st = obspy.Stream()
    # 2. Read every mseed and append the trace contained in it to st
    for i in file_pathlist:
        # Extract datetime from the current filename (as a string, ofc)
        i_datetime_str = i.name.split('.')[1]
        i_datetime = obspy.UTCDateTime.strptime(i_datetime_str, '%y%m%d%H%M%S')
        # The current file is read if:
        # 1. It contains the hour in which the event occurred.
        # 2. It contains the hour following the hour in which the event occurred,
        # and it happened more than 20 minutes after the start of its hour.
        if i_datetime_str == eq_time.strftime('%y%m%d%H0000'):
            cur_trace = obspy.read(i)[0]
            st.append(cur_trace)
        elif ((i_datetime - eq_time)/60 <= 40.0 and
              (i_datetime - eq_time)/60 > 0):
            cur_trace = obspy.read(i)[0]
            st.append(cur_trace)

    # Merge ObsPy Trace objects with same IDs (if they exist)
    st.merge()

    return st


def plot_stream(st, eq_id, eq_time, eq_mag, avg_dist,
                dpi=170, x_size=1664, y_size=936,
                display_plot=True, save_path=''):
    """
    Plot seismograms from an ObsPy Stream object and put information about
    the earthquake on it. Optionally save this visualization.

    Parameters:
    st (obspy.Stream): The Stream object containing seismogram data.
    eq_id (str): The earthquake event ID.
    eq_time (obspy.UTCDateTime): The time of the earthquake event.
    eq_mag (float): The magnitude of the earthquake.
    avg_dist (float): The average distance of the recording stations from the epicenter
    (km).
    dpi (int, optional): The resolution of the plot in dots per inch.
    Default is 170.
    x_size (int, optional): The width of the plot in pixels.
    Default is 1664.
    y_size (int, optional): The height of the plot in pixels.
    Default is 936.
    display_plot (bool, optional): Whether to display the plot. Default is True.
    save_path (Path, optional): Path to save the plot.
    If not provided, the plot will not be saved.

    Returns:
    None
    """
    # Visualize these raw seismograms
    fig = st.plot(method='full', dpi=170, size=(x_size, y_size), handle=True)

    # Show the plot or not
    if not display_plot:
        plt.close(fig)  # So that the figure is not shown

    # Create informative title
    fig.suptitle(t=f"Event ID: {eq_id}, time: {eq_time.strftime('%Y-%m-%d %H:%M')},"
                 + f' mean distance: {avg_dist} km, mag: {eq_mag}',
                 fontsize=12)

    # Customize axis
    for ax in fig.axes:
        # Mark the eq_time; need to convert eq_time to matplotlib date format
        ax.axvline(x=eq_time.matplotlib_date, color='r', linewidth=1.5)
        #
        ax.tick_params(axis='y', which='both', labelsize=7)

    # Save the result (if required)
    if save_path:
        iterator = 1
        while True:
            if save_path.joinpath(f'{eq_id}_{iterator}.png').exists():
                iterator += 1
            else:
                break
        output_file = save_path.joinpath(f'{eq_id}_{iterator}.png')
        fig.savefig(fname=output_file, dpi='figure', bbox_inches='tight')
        print(f"The current figure saved to {output_file}")


def plot_eq_stats(eq_id, lons, lats, k_1=0.1, k_2=0.035,
                  labels=False, station_names=False,
                  dpi=150, x_size=1408, y_size=792,
                  display_plot=True, save_path=''):
    """
    Plot a map with one earthquake (eq_id) and stations
    using given longitude and latitude (numpy) arrays.

    Parameters:
    eq_id (str): The earthquake event ID.
    lons (np.array): Array of longitudes, with the last element being
    the earthquake longitude.
    lats (np.array): Array of latitudes, with the last element being
    the earthquake latitude.
    k_1 (float, optional): Determines the percentage by which the plot limits
    will be shifted. Default is 0.1.
    k_2 (float, optional): Determines the distance of the label from
    the marker on the y-axis as a percentage of max(delta longitude, delta latitude).
    Default is 0.035.
    labels (bool, optional): Whether to actually add labels to the station
    and earthquake markers.
    Default is False.
    station_names (pd.Series, optional): What to write in the station labels.
    dpi (int, optional): The resolution of the plot in dots per inch.
    Default is 150.
    x_size (int, optional): The width of the plot in pixels. Default is 1408.
    y_size (int, optional): The height of the plot in pixels. Default is 792.
    display_plot (bool, optional): Whether to display the plot. Default is True.
    save_path (Path, optional): Path to save the plot.
    If not provided, the plot will not be saved.

    Returns:
    None
    """
    # Construct a map with earthquakes and stations.
    # First, let's determine the boundaries of the future map
    lon_max = np.max(lons)
    lon_min = np.min(lons)
    delta_lons = lon_max - lon_min
    #
    lat_max = np.max(lats)
    lat_min = np.min(lats)
    delta_lats = lat_max - lat_min
    # Find the largest range and extend the other one to make the map more square-like
    if delta_lats > delta_lons:
        lon_center = (lon_max + lon_min) / 2
        lon_min = lon_center - delta_lats / 2
        lon_max = lon_center + delta_lats / 2
    else:
        lat_center = (lat_max + lat_min) / 2
        lat_min = lat_center - delta_lons / 2
        lat_max = lat_center + delta_lons / 2

    # The resulting extent of the plot
    extent = [lon_min - k_1*np.max([delta_lons, delta_lats]),
              lon_max + k_1*np.max([delta_lons, delta_lats]),
              lat_min - k_1*np.max([delta_lons, delta_lats]),
              lat_max + k_1*np.max([delta_lons, delta_lats])]

    # Create a Figure and an GeoAxis with the PlateCarree projection
    fig = plt.figure(figsize=(1408/dpi, 792/dpi), dpi=dpi, layout='tight')
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=170))

    # Add features to the map
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAKES)
    ax.add_feature(cfeature.RIVERS)

    # Set extent to zoom in on the area of interest
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # Add a grid
    ax.gridlines(draw_labels={"bottom": "x", "left": "y"}, zorder=1)

    # Plot the earthquake
    ax.plot(lons[-1], lats[-1], linestyle='', marker='*',
            markerfacecolor='red', markeredgecolor='k', markersize=14,
            zorder=2, transform=ccrs.PlateCarree())
    # Plot the stations
    ax.plot(lons[:-1], lats[:-1], linestyle='', marker='^',
            markerfacecolor='w', markeredgecolor='k', markersize=10,
            zorder=2, transform=ccrs.PlateCarree())

    # Add labels
    if labels:
        ax.text(
            lons[-1], lats[-1] + k_2*np.max([delta_lons, delta_lats]),
            f"{eq_id}", ha='center', fontstyle='normal', fontsize=8,
            bbox=dict(facecolor='white', alpha=0.65, edgecolor='none', pad=1),
            transform=ccrs.PlateCarree(), zorder=3,
        )
    if station_names:
        for i in range(len(lons[:-1])):
            ax.text(
                lons[i], lats[i] + k_2*np.max([delta_lons, delta_lats]),
                f"{station_names.iloc[i]}",
                ha='center', fontstyle='normal', fontsize=8,
                bbox=dict(facecolor='white', alpha=0.65, edgecolor='none', pad=1),
                transform=ccrs.PlateCarree(), zorder=3
            )

    # Show the plot or not
    if not display_plot:
        plt.close(fig)

    # Save the figure (if required)
    if save_path:
        iterator = 1
        while True:
            if save_path.joinpath(f'{eq_id}_{iterator}.png').exists():
                iterator += 1
            else:
                break
        output_file = save_path.joinpath(f'{eq_id}_map.png')
        fig.savefig(fname=output_file, dpi='figure', bbox_inches='tight')
        print(f"The current figure saved to {output_file}")
