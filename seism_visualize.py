"""
List of earthquakes (by increasing distance from CK):
1) https://earthquake.usgs.gov/earthquakes/eventpage/us70006a9e/executive
2) https://earthquake.usgs.gov/earthquakes/eventpage/us60006m7z/executive
3) https://earthquake.usgs.gov/earthquakes/eventpage/us70006f6d/executive
4) https://earthquake.usgs.gov/earthquakes/eventpage/us70006cb6/executive
5) https://earthquake.usgs.gov/earthquakes/eventpage/us6000632m/executive
"""

# Imports
import os
import functions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#
import obspy
from pathlib import Path
from random import randint
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Inputs
hour_folder = Path('mseed')
# hour_pathlist = [item for item in hour_folder.iterdir() if item.is_file()]
eq_stat_df = pd.read_excel('earthquakes_stations.xlsx')
inv_temp = obspy.read_inventory('meta_data/inventory_temp_stations.xml')
inv_perm = obspy.read_inventory('meta_data/inventory_perm_stations.xml')
pre_filt = [0.025, 0.03, 1, 1.2]  # remove_response profilter

# Outputs
pics_folder = Path('pics')
#%%
# Combine inventories for permanent and temporary netrworks
inv = inv_temp + inv_perm
del inv_temp, inv_perm

# Loop through every earthquake
for i in eq_stat_df['eq_id'].unique():
    # It is assumed that each mseed file contains one trace.
    # All traces start and end at the same time.

    # Leave in eq_stat_df only the data corresponding to the event in question
    mod_eq_stat_df = eq_stat_df.query("eq_id==@i")
    # Timestamp of the event
    eq_time = obspy.core.UTCDateTime(mod_eq_stat_df['eq_time'].iloc[0])
    # Round down to the nearest minute
    eq_time = eq_time.replace(second=0, microsecond=0)
    # ID of the event
    eq_id = mod_eq_stat_df['eq_id'].iloc[0]
    # Magnitude of the event
    eq_mag = mod_eq_stat_df['eq_mag'].iloc[0]
    # Coordinates of the event
    eq_loc = mod_eq_stat_df[['eq_lat', 'eq_lon']].iloc[0]
    # Avg distance to the stations
    avg_dist = np.round(mod_eq_stat_df['distance'].mean(), 2)

    st = functions.generate_stream(hour_folder, eq_time)

    # If nothing was added to st, ring the alarm!
    if len(st) == 0:
        print(f'No data was found for {eq_id}!')
        continue

    # Set a new common start and end time of the modified seismograms
    # based on the time of the earthquake. Take 2 minutes before the event
    # and 10 minutes after.
    starttime = eq_time - 2*(avg_dist // 500)*60
    endttime = eq_time + 8*(avg_dist // 500)*60
    st.trim(starttime, endttime)

    # Visualize these raw seismograms
    functions.plot_stream(st, eq_id, eq_time, eq_mag, avg_dist,
                          display_plot=False, save_path=pics_folder)

    # A list for placing dfs with station name and coordinates
    stat_list = []

    # Introduce instrumental corrections to the data from all the stations,
    # but visualize the process for only one example.
    # Selection of the station for which the plot will be saved.
    random_index = randint(0, len(st)-1)
    # # Numpy variant (a lot more convoluted for such a simple task)
    # np.random.default_rng().integers(0, 5, 1)

    st[random_index].stats.station + '_remove_response.png'

    for k in range(len(st)):
        if k != random_index:
            st[k].remove_response(inventory=inv, output='VEL',
                                  pre_filt=pre_filt, zero_mean=True,
                                  taper=False, plot=False)
        else:
            output_file = pics_folder.joinpath(f'{eq_id}_'
                                               + st[k].stats.station
                                               + '_remove_response.png')
            st[k].remove_response(inventory=inv, output='VEL',
                                  pre_filt=pre_filt, zero_mean=True,
                                  taper=False,
                                  plot=(output_file))
            # del output_file
        # For the purpose of further visualization,
        # let's save the information about the current station
        # (name and coordinates).
        # The @ symbol is used to indicate that cur_stat_name is a variable,
        # not a string literal.
        station_name = st[k].stats.station
        cur_stat = mod_eq_stat_df.query("station_name==@station_name")
        cur_stat = cur_stat[['station_name', 'station_lat', 'station_lon']]
        stat_list.append(cur_stat)

    # Combining the tables from the list into one table.
    stat_df = pd.concat(stat_list, axis=0).drop_duplicates()
    del stat_list

    # Build a new figure with all seismograms plotted on it
    functions.plot_stream(st, eq_id, eq_time, eq_mag, avg_dist,
                          display_plot=False, save_path=pics_folder)

    # Construct a map with earthquakes and stations.
    # Longitudes of both the earthquake and the stations (np.array)
    lons = np.append(stat_df['station_lon'].values, eq_loc['eq_lon'])
    # Latitudes of both the earthquake and the stations (np.array)
    lats = np.append(stat_df['station_lat'].values, eq_loc['eq_lat'])
    functions.plot_eq_stat(eq_id, lons, lats, display_plot=False,
                           save_path=pics_folder)
