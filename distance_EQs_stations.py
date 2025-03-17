# -*- coding: utf-8 -*-
"""
This is outdated code. Now its functionality lives in functions.distances_eqs_stations()!
"""

# Imports
import pandas as pd
import numpy as np

# Inputs
# Note: earthquake .csv file should be taken from https://earthquake.usgs.gov/
eq_csv = 'E:/Work/projects/surface_waves_earthquakes/earthquakes.csv'
stations_info = 'E:/Work/projects/surface_waves_earthquakes/used_stations_info.xlsx'

# Reading (only the specified columns)
# and modifying files with various earthquake and station geodata
#
eq_df = pd.read_csv(eq_csv)[['id', 'mag', 'time', 'depth',
                             'latitude', 'longitude',]]
eq_df.rename(columns={'id': 'eq_id',
                      'mag': 'eq_mag',
                      'time': 'eq_time',
                      'latitude': 'eq_lat',
                      'longitude': 'eq_lon',
                      'depth': 'eq_depth'},
             inplace=True)
# Convert longitude if necessary
eq_df['eq_lon'] = eq_df['eq_lon'].apply(lambda x: x + 360 if x < 0 else x)
#
stations_df = pd.read_excel(stations_info)[['station name', 'elevation',
                                            'latitude', 'longitude']]
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
for i in range(eq_df.shape[0]):
    # Calculating the horizontal distance using the haversine formula
    # print(i)
    delta_lat = (np.radians(stations_df['station_lat'])
                 - np.radians(eq_df['eq_lat'][i]))
    delta_lon = (np.radians(stations_df['station_lon'])
                 - np.radians(eq_df['eq_lon'][i]))
    a = (
         np.sin(delta_lat / 2)**2
         + (np.cos(np.radians(eq_df['eq_lat'][i]))
            * np.cos(np.radians(stations_df['station_lat']))
            * np.sin(delta_lon / 2)**2)
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    dist_surface = R * c

    # Calculating the total distance considering the depth of the earthquake
    dist_total = np.sqrt(dist_surface**2
                         + (eq_df['eq_depth'][i] + stations_df['station_elev'])**2)

    # Creating a table containing, among other things, the distance between
    # the current earthquake and each of the stations under consideration.
    new_columns_df = pd.DataFrame({
        'distance': dist_total.values,
        # NOTE: np.tile()
        'eq_id': np.tile(eq_df.iloc[i, :]['eq_id'], stations_df.shape[0]),
        'eq_mag': np.tile(eq_df.iloc[i, :]['eq_mag'], stations_df.shape[0]),
        'eq_time': np.tile(eq_df.iloc[i, :]['eq_time'], stations_df.shape[0]),
        'eq_lat': np.tile(eq_df.iloc[i, :]['eq_lat'], stations_df.shape[0]),
        'eq_lon': np.tile(eq_df.iloc[i, :]['eq_lon'], stations_df.shape[0]),
        'eq_depth': np.tile(eq_df.iloc[i, :]['eq_depth'], stations_df.shape[0])
        })
    cur_results_df = pd.concat([new_columns_df, stations_df], axis=1)
    # Adding this intermediate DataFrame cur_results_df to the list.
    results_list.append(cur_results_df)

# Combining the tables from the list into one final table.
results_df = pd.concat(results_list, axis=0)
# Rearranging its the columns for a more logical display
results_df = results_df[['eq_id', 'station_name', 'eq_time', 'eq_mag', 'distance',
                         'eq_lat', 'eq_lon', 'eq_depth',
                         'station_lat', 'station_lon', 'station_elev']]
results_df.to_excel("earthquakes_stations.xlsx", sheet_name="Main", index=False)
#%%
# ## First, let's work out the proccess using only 1 earthquke and 1 station
# eq = eq_df.iloc[1, :]
# station = stations_df.iloc[1, :]

# # Approximate Earth radius (km)
# R = 6371

# # Calculating the horizontal distance using the haversine formula
# delta_lat = np.radians(station['station_lat']) - np.radians(eq['eq_lat'])
# delta_lon = np.radians(station['station_lon']) - np.radians(eq['eq_lon'])
# a = (
#      np.sin(delta_lat / 2)**2
#      + (np.cos(np.radians(eq['eq_lat']))
#         * np.cos(np.radians(station['station_lat']))
#         * np.sin(delta_lon / 2)**2)
# )
# c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

# d_surface = R * c

# # Calculating the total distance considering the depth of the earthquake
# d_total = np.sqrt(d_surface**2 + (eq['eq_depth'] + station['station_elev'])**2)
