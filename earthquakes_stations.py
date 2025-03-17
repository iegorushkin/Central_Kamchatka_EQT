import functions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path

# Note: earthquake .csv file should be taken from https://earthquake.usgs.gov/
eq_data = '/Users/igor/Desktop/Work/IPGG/Central_Kamchatka_EQT/earthquakes.csv'
stations_data = '/Users/igor/Desktop/Work/IPGG/Central_Kamchatka_EQT/used_stations_info.xlsx'
eqs_stats = functions.distances_eqs_stations(eq_data, stations_data, save_flag=1)
# eqs_stats = pd.read_excel('earthquakes_stations.xlsx')

pics_folder = Path('pics')
# %%
# 1. Let's find the records from which stations are present in the working directory.
mseed_folder = Path('mseed/RAW')
# Generate a list with all files located in the working directory.
mseed_pathlist = [item for item in mseed_folder.iterdir() if item.is_file()]
# Create a DataFrame from this list.
df = pd.DataFrame(data={'mseed_pathlist': mseed_pathlist})
# Add a column to the DataFrame with the station names extracted from
# the Path instances.
df['station_names'] = df['mseed_pathlist'].apply(lambda x: x.name.split('.')[0])
# Save the unique station names that occur in the data.
station_names = df['station_names'].unique()
# A small cleanup
del mseed_folder, mseed_pathlist, df

## Old Code
# # Loop through all the file names,
# # extract station names from them, and save these names to a list.
# temp_list = []
# for item in mseed_pathlist:
#     temp_list.append(item.name.split('.')[0])
# # That's the interesting part.
# # Passing the list to set() causes the repeated names to be removed.
# # And then passing set() to list() converts the data back into a convenient type.
# station_names = list(set(temp_list))
# %%
# 2. Let's build a DataFrame containing information about the stations
# whose records are present in the working directory.
# Contains non-repeating lines of station information.
stations_info = eqs_stats.drop_duplicates(subset=['station_name']) \
    [['station_name', 'station_lon', 'station_lat']]
# Select only the rows corresponding to stations from station_names.
# The @ symbol is used to refer to variables from the Python environment
# within the query string.
stations_info = stations_info.query('station_name in @station_names')
# or
# stations_info = stations_info[stations_info['station_name'].isin(station_names)]
# %%
# 3. Let's gather all the information needed about earthquakes.
eqs_info = eqs_stats.drop_duplicates(subset=['eq_id']) \
    [['eq_id', 'eq_mag', 'eq_lon', 'eq_lat', 'eq_depth']]
# %%
# 4. Construct a map with earthquakes and stations.
# NOTE: For one earthquake a function from functions.py was written!
# Longitudes of both the stations and the earthquakes (np.array)
lons = np.append(stations_info['station_lon'].values, eqs_info['eq_lon'])
# Latitudes of both the stations and the earthquakes (np.array)
lats = np.append(stations_info['station_lat'].values, eqs_info['eq_lat'])
# Let's determine the boundaries of the future map
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
# Determines the percentage by which the plot limits will be shifted
k_1 = 0.1
# Setup the extent of the plot
# By using coeff k_1
extent = [lon_min - k_1*np.max([delta_lons, delta_lats]),
          lon_max + k_1*np.max([delta_lons, delta_lats]),
          lat_min - k_1*np.max([delta_lons, delta_lats]),
          lat_max + k_1*np.max([delta_lons, delta_lats])]
# # More manual approach
# extent = [lon_min - k_1*np.max([delta_lons, delta_lats]),
#           lon_max + k_1*np.max([delta_lons, delta_lats]),
#           42.0,
#           62.0]
# Create a Figure and an GeoAxis with the PlateCarree projection
dpi = 170
fig = plt.figure(figsize=(792/dpi, 792/dpi), dpi=dpi, layout='tight')
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=170))
# Add features to the map
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAKES)
ax.add_feature(cfeature.RIVERS)
# Set extent to zoom in on the area of interest
ax.set_extent(extent, crs=ccrs.PlateCarree())
# Add a grid. There are some problems with x-axis.
# ax.gridlines(draw_labels={"bottom": "x", "left": "y"}, zorder=1)
# Custom ticks and labels on x-axis
gl = ax.gridlines(draw_labels=True, zorder=1)
gl.top_labels = False
gl.right_labels = False
# + eastern hemisphere, - western hemisphere
gl.xlocator = plt.FixedLocator([150, 155, 160, 165, 170, 175, 180, -175, -170])
# Plot the stations
ax.plot(lons[:stations_info.shape[0]], lats[:stations_info.shape[0]],
        linestyle='', marker='^', markerfacecolor='w', markeredgecolor='k',
        markersize=8, zorder=2, transform=ccrs.PlateCarree())
# Plot the earthquakes
ax.plot(lons[stations_info.shape[0]:], lats[stations_info.shape[0]:],
        linestyle='', marker='*', markerfacecolor='red', markeredgecolor='k',
        markersize=12, zorder=2, transform=ccrs.PlateCarree())
# Display text labels above the earthquake markers
# k_2 determines the distance of the label from the marker on the y-axis
# as a percentage of max(delta longitude, delta latitude)
k_2 = 0.045
# Flag for a simple sign change
i = 1
for row in eqs_info.itertuples(index=True):
    ax.text(row.eq_lon,
            row.eq_lat + i*k_2*np.max([delta_lons, delta_lats]),
            f"{row.eq_id}", ha='center', fontstyle='normal', fontsize=6.5,
            bbox=dict(facecolor='white', alpha=0.65, edgecolor='none', pad=1),
            transform=ccrs.PlateCarree(), zorder=3)
    i = i*(-1)
# Show the result
fig.show()
# Save the result
output_file = pics_folder.joinpath('all_eq_map.png')
fig.savefig(fname=output_file, dpi='figure', bbox_inches='tight')
print(f"The current figure saved to {output_file}")
