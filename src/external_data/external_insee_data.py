import pandas as pd
from geopy import distance

import src.external_data.external_geo_data as ext_geo

def add_external_insee_revenue_data(train: pd.DataFrame, test: pd.DataFrame, revenus: pd.DataFrame) -> pd.DataFrame:
    """Add 2018 median revenue data by IRIS code in train and test data. 

    Args:
        train (pd.DataFrame): train data
        test (pd.DataFrame): test data
        revenus (pd.DataFrame): INSEE revenue data

    Returns:
        pd.DataFrame, pd.DataFrame: train and test sets with revenue data
    """
    # Fix type of IRIS codes
    train['code_iris'] = train['code_iris'].astype(int).astype(str)
    test['code_iris'] = test['code_iris'].astype(int).astype(str)

    # Get a dictionary to map missing IRIS zones to the closest zone
    closest_iris_to_non_mapped_dict = get_closest_iris_zone_to_missing_codes(train, test, revenus)

    # Replace missing IRIS
    train['code_iris_mappable'] = train['code_iris'].replace(closest_iris_to_non_mapped_dict)
    test['code_iris_mappable'] = test['code_iris'].replace(closest_iris_to_non_mapped_dict)

    # Merge median revenue value for 2018
    median_revenue_dict = dict(revenus.set_index('IRIS')['DISP_MED18'])
    train['med_revenue_iris_2018'] = train['code_iris_mappable'].map(median_revenue_dict)
    test['med_revenue_iris_2018'] = test['code_iris_mappable'].map(median_revenue_dict)

    train.drop(columns=['code_iris', 'code_iris_mappable'], inplace=True)
    test.drop(columns=['code_iris', 'code_iris_mappable'], inplace=True)

    return train, test

def get_closest_iris_zone_to_missing_codes(train: pd.DataFrame, test: pd.DataFrame, revenus: pd.DataFrame) -> dict:
    """Get a dictionary with the closest available IRIS code for each IRIS code from train/test that is not available in INSEE data.

    Args:
        train (pd.DataFrame): train data
        test (pd.DataFrame): test data
        revenus (pd.DataFrame): INSEE revenue data

    Returns:
        dict: mapping of missing IRIS codes to the closest IRIS zone
    """
    iris_mapped, iris_non_mapped = get_iris_codes(train, test, revenus)

    train_test_concat = pd.concat([train, test])[['code_iris', 'lat', 'lon']]
    avg_coordinates_by_iris = train_test_concat.groupby('code_iris').mean()
    avg_coordinates_by_iris = ext_geo.create_housing_unit_coordinates_col(avg_coordinates_by_iris)
    avg_coordinates_by_iris_mapped = avg_coordinates_by_iris.loc[iris_mapped, :]

    closest_iris_to_non_mapped_dict = {}
    for code in iris_non_mapped:
        code_mean_coordinates = avg_coordinates_by_iris.loc[code, 'coordinates']
        avg_coordinates_by_iris_mapped['dist_to_code'] = avg_coordinates_by_iris_mapped['coordinates'].apply(
            lambda iris_coord: distance.geodesic(iris_coord, code_mean_coordinates).km
        )
        closest_mapped_iris = avg_coordinates_by_iris_mapped.sort_values('dist_to_code', ascending=True).index[0]
        closest_iris_to_non_mapped_dict[code] = closest_mapped_iris

    return closest_iris_to_non_mapped_dict 

def get_iris_codes(train: pd.DataFrame, test: pd.DataFrame, revenus: pd.DataFrame) -> tuple:
    """Get lists of IRIS codes in train/test that are either available or unavailable in INSEE data.

    Args:
        train (pd.DataFrame): train data
        test (pd.DataFrame): test data
        revenus (pd.DataFrame): INSEE revenue data

    Returns:
        tuple(list, list): lists available and unavailable IRIS codes
    """
    # Get all IRIS codes in train & test
    iris_codes = set(list(train['code_iris']) + list(test['code_iris']))

    # Get IRIS codes available in INSEE data
    insee_iris = set(revenus['IRIS'])

    # Separate mappable/non mappable codes
    iris_mapped, iris_non_mapped = [], []
    for code in iris_codes:
        if code in insee_iris:
            iris_mapped.append(code)
        else:
            iris_non_mapped.append(code)

    return iris_mapped, iris_non_mapped