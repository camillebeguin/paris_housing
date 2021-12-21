import os

# Path
RAW_NB_PATH = os.path.join('..', 'data', 'raw')
RAW_NB_TRAIN_PATH = os.path.join(RAW_NB_PATH, 'dvf', 'dvf_train.csv')
RAW_NB_TEST_PATH = os.path.join(RAW_NB_PATH, 'dvf', 'dvf_test.csv') 

RAW_TRANSPORTS_PATH = os.path.join(RAW_NB_PATH, 'external', 'emplacement-des-gares-idf.csv') 
RAW_INSEE_PATH = os.path.join(RAW_NB_PATH, 'external', 'BASE_TD_FILO_DISP_IRIS_2018.csv') 

PREPROCESSED_NB_PATH = os.path.join('..', 'data', 'preprocessed')
PREPROCESSED_NB_TRAIN_PATH = os.path.join(RAW_NB_PATH, 'dvf', 'dvf_train_preprocessed.csv')
PREPROCESSED_NB_TEST_PATH = os.path.join(RAW_NB_PATH, 'dvf', 'dvf_test_preprocessed.csv')

PREDICTIONS_PATH = os.path.join('..', 'data', 'predictions', 'test_predictions.csv')

# Column selection and name mapping
train_cols = ['Date mutation', 'Nature mutation', 'Type de voie', 'Commune',
       'Surface Carrez du 1er lot', 'Surface Carrez du 2eme lot', 'Nombre de lots',
       'Type local', 'Surface reelle bati', 'Nombre pieces principales', 
       'lon', 'lat', 'code_district_custom', 'code_IRIS', 'Valeur fonciere']

test_cols = ['Date mutation', 'Nature mutation', 'Type de voie', 'Commune',
       'Surface Carrez du 1er lot', 'Surface Carrez du 2eme lot', 'Nombre de lots',
       'Type local', 'Surface reelle bati', 'Nombre pieces principales', 
       'lon', 'lat', 'code_district_custom', 'code_IRIS']

column_names_mapping = {
       'Date mutation': 'date_mutation', 
       'Nature mutation': 'nature_mutation', 
       'Type de voie': 'type_voie', 
       'Commune': 'commune',
       'Surface Carrez du 1er lot': 'surface_carrez_1er_lot', 
       'Surface Carrez du 2eme lot': 'surface_carrez_2e_lot', 
       'Surface Carrez du 3eme lot': 'surface_carrez_3e_lot', 
       'Nombre de lots': 'nb_lots',
       'Type local': 'type_local', 
       'code_district_custom': 'district',
       'code_IRIS': 'code_iris',
       'Surface reelle bati': 'surface_reelle_bati', 
       'Nombre pieces principales': 'nb_pieces', 
       'Valeur fonciere': 'valeur'
}

# Parameters for external geo data
limit_distance_district_center = 1.2
short_distance_station = 0.5

# Parameters for models
random_seed = 40
train_size = 0.75
raw_target_col = "valeur"
target_col = "valeur_m2"