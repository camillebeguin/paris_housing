# Paris housing

The goal of this project was to predict the selling price of housing units in different Paris arrondissements. 
Available features included housing surface, number of rooms, location (latitude/longitude), district. 

### 1. Data preprocessing 

The first step was the preprocessing of the train and test set. I restricted the training set to "Ventes" only. 
I cleaned surface and price columns and clipped the number of rooms to 1 to remove zeros. 

### 2. External data 

Then, I added external data from the following sources: 
- https://data.iledefrance-mobilites.fr/explore/dataset/referentiel-des-lignes/ (open public transportation data)
- https://www.insee.fr/fr/statistiques/5055909

The first dataset lists all public transportation stations in Ile-de-France (métro, trains, RER) and their geographic coordinates.
My hypothesis was that the number of inbound/outbound connections within 500m of a housing unit in Paris has an impact on price. 
The second dataset lists the 2018 median revenue by IRIS code (an information that is available in the housing dataset). 
My initial hypothesis was that this information could be used as a proxy for living conditions, but it did not improve predictions.

### 3. Modeling 
After preparing the dataset, I tested different regression models. My goal was to predict a price per m2 for each housing unit rather 
than the final price. Business intuition back by feature importance analyses showed that major features included surface/apartment type (e.g. 
one-bedroom apartments are more expensive by m2 than three-bedrooms) and geographic location. 

I tried basic models such as Linear Regression or Decision trees, then moved to more complex models such as Random Forest Regressor, 
XGBoost, LGBM and Catboost. I performed some hyperparameter tuning on the most promising models, and selected the final model using cross-validation.
I built functions to preprocess/encode features based on a feature preprocessing dictionary. Preprocessing steps included min-max scaling of coordinates,
count frequency encoder (i.e. creating a new feature from the `code_district`column), one-hot encoding or target encoding for `commune` etc.

### 4. Predictions 

The last step was to predict price per m2 on the test set using the final selected pipeline, then convert predictions back to a total price.
