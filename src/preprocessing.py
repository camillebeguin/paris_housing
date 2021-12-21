import src.constants as cst

def preprocess_housing_data(data, is_train):
    """Preprocess training or test data.

    Args:
        data (pd.DataFrame): training data with selected and renamed columns.
        is_train (bool, optional): boolean to indicate if train or test dataset. Defaults to True.
    """
    # Mask "ventes" and rows with non-null district
    data = data.query("(nature_mutation=='Vente') & (district==district)")

    # Fill missing surface Carrez values
    data.loc[:, 'surface_carrez_1er_lot'] = data.loc[:, 'surface_carrez_1er_lot'].fillna(data['surface_reelle_bati'].astype(str))

    # Clean float values
    if is_train: 
        float_cols_to_clean = ['surface_carrez_1er_lot', 'surface_carrez_2e_lot', 'valeur']
    else: 
        float_cols_to_clean = ['surface_carrez_1er_lot', 'surface_carrez_2e_lot']

    for col in float_cols_to_clean:
        data.loc[:, col] = data.loc[:, col].str.replace(',', '.').astype(float)
        data.loc[:, col] = data.loc[:, col].fillna(0)

    # Clip zero values in nb_pieces to 1
    data.loc[:, 'nb_pieces'] = data.loc[:, 'nb_pieces'].clip(lower=1)
    return data

def prepare_housing_data(data, is_train):
    """Function to select relevant columns and rename them

    Args:
        data (pd.DataFrame): raw train or test dataset.
        is_train (bool, optional): boolean to indicate if train or test dataset. Defaults to True.

    Returns:
        pd.DataFrame
    """
    if is_train:
        data = data.loc[:, cst.train_cols]
    else:
        data = data.loc[:, cst.test_cols]

    # Rename columns
    data = data.rename(columns=cst.column_names_mapping)

    return data