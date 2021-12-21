import itertools

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from category_encoders.target_encoder import TargetEncoder
from feature_engine.encoding import CountFrequencyEncoder

def build_pipeline(estimator, features_dict: dict):
    """Build a pipeline from an estimator and dictionary of feature preprocessing.

    Args:
        estimator: model object
        features_dict (dict): dictionary of feature preprocessing

    Returns:
        pipeline: sklearn pipeline object
    """
    # First, get the encoder
    encoder = get_encoder(features_dict)
    
    # If set to true, we add a "district rarity" feature that counts the frequency of each district in X_train
    feature_engine = CountFrequencyEncoder(encoding_method='frequency', variables=features_dict['count_freq_encoded'])

    pipeline = Pipeline([
        ("feature_engine", feature_engine), 
        ("encoder", encoder), 
        ("estimator", estimator)]
    )

    return pipeline

def get_encoder(features_dict: dict):
    """Get sklearn encoder from a dictionary of feature preprocessing.
    The dictionary should follow the structure:

    features_dict = {
        'min_max_scaled': list, 
        'standard_scaled': list, 
        'one_hot_encoded': list, 
        'target_encoded': list, 
        'count_freq_encoded': list,
        'unprocessed': list,
    }

    List values can be empty. If add_district_frequency is set to True, a new feature is added: the district rarity. 
    It corresponds to the frequency of district values in train. 

    Args:
        features_dict (dict): dictionary of feature preprocessing.

    Returns:
        encoder: sklearn column transformer.
    """
    encoder = make_column_transformer(
        (MinMaxScaler(), features_dict['min_max_scaled']), 
        (StandardScaler(), features_dict['standard_scaled']), 
        (OneHotEncoder(handle_unknown="ignore"), features_dict['one_hot_encoded']),
        (TargetEncoder(), features_dict['target_encoded']), 
        remainder="passthrough", verbose=0
    )
    return encoder

def get_features_from_dict(features_dict: dict) -> list:
    """Extract a list of features from the features_dict 

    Args:
        features_dict (dict): dictionary of preprocessing steps to apply and associated features

    Returns:
        list: list of features used
    """
    features_dict_copy = features_dict.copy()
    features = list(itertools.chain.from_iterable(features_dict_copy.values()))
    return features