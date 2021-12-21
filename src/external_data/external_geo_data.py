import pandas as pd
from geopy import distance

import src.constants as cst

def count_close_stations_all_housing_units_in_train_test(train: pd.DataFrame, test: pd.DataFrame, transports: pd.DataFrame) -> pd.DataFrame:
    """Main function to count the number of metros/trains within short_distance_station km from each housing unit in the training set.

    Args:
        train (pd.DataFrame): DVF train dataset.
        test (pd.DataFrame): DVF test dataset.
        transports (pd.DataFrame): open dataset from Ile-de-France Mobilités.
        short_distance_station (float, optional): distance in kilometers for which we compute the # of connections. Defaults to 0.5.

    Returns:
        pd.DataFrame: [description]
    """
    # Set parameters
    limit_distance = cst.limit_distance_district_center
    short_distance_station = cst.short_distance_station

    # Preprocess the `transports` dataset
    transports = get_transportation_count_per_station(transports)

    # Preprocess the `data` dataset by adding coordinates
    train = create_housing_unit_coordinates_col(train)
    test = create_housing_unit_coordinates_col(test)

    # Get the district centers
    average_coordinates = compute_average_coordinates_by_district(train)

    # Get the stations that lie within limit_distance kilometers from each district center
    closest_stations_per_district = get_closest_stations_dict_per_district(average_coordinates, transports, limit_distance)

    # Count the number of stations close to each housing unit (in train)
    print('Preprocessing train...')
    train = count_close_stations_all_housing_units_in_dataset(train, closest_stations_per_district, short_distance_station)

    print('Preprocessing test...')
    test = count_close_stations_all_housing_units_in_dataset(test, closest_stations_per_district, short_distance_station)
    return train, test


def count_close_stations_all_housing_units_in_dataset(data: pd.DataFrame, closest_stations_dict_per_district: dict, short_distance_station: float):
    n_metros_units = []
    n_trains_units = []

    count = 0

    for _, unit_row in data.iterrows():
        # For tracking purposes in the notebook (could be replaced by a logger)
        count += 1 
        if count%1000==0:
            print(f'Processed {count} units...')
        
        unit_coordinates = unit_row['coordinates']
        unit_district = unit_row['district']
        n_transports = count_close_stations_housing_unit(
            unit_coordinates, unit_district, closest_stations_dict_per_district, short_distance_station
        )
        n_metros_units.append(n_transports['n_metros'])
        n_trains_units.append(n_transports['n_trains'])

    data[f'n_metros_within_{short_distance_station}km'] = n_metros_units
    data[f'n_trains_within_{short_distance_station}km'] = n_trains_units
    return data


def count_close_stations_housing_unit(unit_coordinates: tuple, unit_district: str, closest_stations_dict_per_district: dict, short_distance_station: float) -> dict:
    """Function to count the number of stations that lie within 0.5km of a given housing unit. 

    Args:
        unit_coordinates (tuple): coordinates of the housing unit.
        unit_district (str): district of the housing unit.
        closest_stations_dict_per_district (dict): dictionary with stations within 1.2km of the district center for each district.
        short_distance_station (float): geographic perimeter to count metro/train stations. Defaults to 0.5.

    Returns:
        dict: {n_metros: x,  n_trains: y} within short_distance_station kilometers from the housing unit.
    """
    
    # Get stations that are less than 1.2km away from the district center (for computation time purpose)
    closest_stations_district_df = closest_stations_dict_per_district[unit_district]
    closest_stations_district_df['distance_to_unit'] = closest_stations_district_df['station_coordinates'].apply(
        lambda station_coord: distance.geodesic(station_coord, unit_coordinates)
    )

    # Filter stations that are below short_distance_station kilometers (0.5 by default)
    short_distance_stations = closest_stations_district_df.query('distance_to_unit <= @short_distance_station')

    # Count the total number of outbound connections from short distance stations (metro & train/RER)
    n_transports = {'n_metros': 0, 'n_trains': 0}
    if closest_stations_district_df.shape[0] > 0:
        n_transports['n_metros'] = short_distance_stations['metro'].sum()
        n_transports['n_trains'] = short_distance_stations[['train', 'rer']].sum().sum()

    return n_transports


def get_closest_stations_dict_per_district(average_coordinates: pd.DataFrame, transports: pd.DataFrame, limit_distance: float) -> dict: 
    """Function to store a dataframe with all stations below limit_distance kilometers from a district's center, for each district.
    The goal is to limit the number of stations for which we compute the exact distance to a given housing unit. 
    We assume that stations that are further than 1.2km away from a unit's district center will not lie within 0.5km from the unit. 

    Args:
        average_coordinates (pd.DataFrame): a dataset with the district center coordinates for each district.
        transports (pd.DataFrame): a dataset with IDF stations and their geographic coordinates.
        limit_distance (float): distance limit in kilometers from the district center . Defaults to 1.2.

    Returns:
        dict: a dictionary with districts as keys and stations within 1.2km from the center stored in a dataframe with coordinates.
    """
    closest_stations_per_district = {}

    for _, district_row in average_coordinates.iterrows():
        district_name = district_row['district']
        district_coordinates = district_row['coordinates']

        district_transports = transports.copy()
        district_transports['distance_from_district_center'] = district_transports['station_coordinates'].apply(
            lambda station_coord: distance.geodesic(station_coord, district_coordinates)
        )
        district_transports = district_transports.query("distance_from_district_center <= @limit_distance")
        closest_stations_per_district[district_name] = district_transports

    return closest_stations_per_district


def get_transportation_count_per_station(transports: pd.DataFrame) -> pd.DataFrame:
    """Function to group station rows and count the total number of transportation means by station

    Args:
        transports (pd.DataFrame): open dataset from Ile-de-France mobilités with inbound/outbound connections

    Returns:
        pd.DataFrame: open dataset with a unique row per station
    """
    transports_cols = ['fer', 'train', 'rer', 'metro']
    func_agg_cols = {'Geo Point': 'first'} | {col: 'sum' for col in transports_cols}

    transports = transports.groupby('nom_long').agg(func_agg_cols)
    transports['station_coordinates'] = transports['Geo Point'].apply(
        lambda point: get_coordinates_tuple_from_geopoint(point)
        )

    return transports


def compute_average_coordinates_by_district(data: pd.DataFrame) -> pd.DataFrame:
    """Function that computes the district center coordinates for each district using the average from the training set 

    Args:
        data (pd.DataFrame): DVF dataset

    Returns:
        pd.DataFrame: dataset with the district name and associated district center as a tuple
    """
    average_coordinates = data.groupby('district')[['lat', 'lon']].mean().reset_index()
    average_coordinates = create_housing_unit_coordinates_col(average_coordinates)
    return average_coordinates.drop(columns=['lat', 'lon'])


def create_housing_unit_coordinates_col(data: pd.DataFrame) -> pd.DataFrame:
    """Create a column with coordinates as a tuple from lat and lon columns

    Args:
        data (pd.DataFrame): any dataset with lat/lon columns stored as floats

    Returns:
        pd.DataFrame: same dataset with coordinates stored in a column
    """
    data['coordinates'] = data['lat'].astype(str) + ',' + data['lon'].astype(str)
    data['coordinates'] = data['coordinates'].apply(
        lambda point: get_coordinates_tuple_from_geopoint(point)
        )
    return data
    

def get_coordinates_tuple_from_geopoint(geopoint: str) -> tuple:
    """Process a geopoint stored as a string in "lat, lon" format into a coordinates tuple

    Args:
        geopoint (str): "lat, lon" string-type geopoint

    Returns:
        tuple: coordinates stored in a tuple
    """
    coords = [float(c) for c in geopoint.split(',')]
    return tuple(coords)