#Importing libraries
import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
import pickle as pkl

### Data Preparation and Feature Extraction Process

#Reading the fifa dataset
fifa_data = pd.read_csv("male_players (legacy).csv", low_memory = False)

fifa_data.head()

fifa_data.describe()

fifa_data.info()

#Visualizing the data
#fifa_data.hist(bins=50, figsize=(20,15))
#plt.show()

#Dropping columns with more than 30% null values
less_null = []
more_null = []
for i in fifa_data.columns:
    if((fifa_data[i].isnull().sum())< (0.3 * (fifa_data.shape[0]))):
        less_null.append(i)
    else:
        more_null.append(i)

less_null

#Reassigning relevant features to dataset
fifa_data = fifa_data[less_null]

fifa_data.info()

#Separating the numeric and quantitative variables
numeric_data = fifa_data.select_dtypes(include = np.number)
non_numeric = fifa_data.select_dtypes(include = ['object'])

numeric_data 

### Feature Subsetting

corr_matrix = numeric_data.corr()
corr_matrix

#correlation between all the values listed and overall rating
corr_matrix = corr_matrix["overall"].sort_values(ascending=False) 
corr_matrix

# Subsetting features with a correlation greater than 0.4
relevant_features = corr_matrix[abs(corr_matrix) > 0.4 ].index
num = numeric_data
numeric_data = numeric_data[relevant_features]
numeric_data

irrelevant_features = corr_matrix[abs(corr_matrix) < 0.4 ].index


#Adding features that were dropped but are relevant as per domain knowledge
new_relevant = irrelevant_features.difference(['player_id','nationality_id','league_id','club_team_id','club_jersey_number','fifa_version','weight_kg','movement_balance','club_contract_valid_until_year','height_cm','fifa_update','mentality_composure']) 
new_relevant

numeric_data

num = num[new_relevant]

num.info()

numeric_data.info()

numeric_data = pd.concat([numeric_data, num], axis = 1)

numeric_data.info()

#Filling in all missing values with the median
#multivariate imputation 
imp = IterativeImputer(max_iter = 10, random_state = 0)
numeric_data = pd.DataFrame(np.round(imp.fit_transform(numeric_data)), columns = numeric_data.columns)#this line learns the data and imputes the missing features

numeric_data.info()

### Handling the non-numeric data

#Using domain knowledge to identify the unnecessary non - numeric featurs
non_numeric = fifa_data.select_dtypes(include = ['object'])
unnecessary_objects = ['player_url','fifa_update_date','player_face_url','dob','short_name', 'long_name','league_name','club_name','club_position','club_joined_date','real_face','body_type']

#Dropping unnecessary non-numeric features
non_numeric.drop(columns = unnecessary_objects, axis = 1 , inplace = True)

#Importing relevant libraries to impute non numeric data
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector

#Creating a pipeline to impute the categorical variables with the mode
cat_pipe = Pipeline([
 ("impute", SimpleImputer(strategy="most_frequent")),
])

#Creating a full pipeline
full_pipe = ColumnTransformer([
    ("cat", cat_pipe,make_column_selector(dtype_include = object))
])

#Imputing and fit transforming the non numeric data
piped = full_pipe.fit_transform(non_numeric)

#putting the new non numeric into a dataframe 
non_numeric = pd.DataFrame(data = piped, columns = full_pipe.get_feature_names_out())

non_numeric.info()

#Using binary encoder to encode the categorical varibles
encoder = ce.BinaryEncoder(cols = non_numeric.columns)
non_numeric = encoder.fit_transform(non_numeric)

non_numeric

fifa_data = pd.concat([numeric_data, non_numeric], axis = 1)
fifa_data

#Declaring our y variable (the dependent variable)
y = fifa_data['overall']

### Scaling the data

X = fifa_data.drop('overall', axis = 1)

X.info()

#Scaling the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

X

### Training and testing the data

Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,y,test_size = 0.2,random_state = 42, stratify = y)

#Using a decision tree
from sklearn.tree import DecisionTreeRegressor
dtree = DecisionTreeRegressor(max_depth = 12)

dtree.fit(Xtrain, Ytrain)

#saving as a pickle

pkl.dump(dtree, open( dtree.__class__.__name__+'.pkl','wb'))

#Importing relevant metrics
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score

#Evaluating the model using mean_absolute_error, mean_squared_error, mean_squared_log_error and r2_score
#Test the model
y_pred = dtree.predict(Xtest)
print(f"""  Mean Absolute Error = {mean_absolute_error(y_pred, Ytest)},
            Mean Squared Error = {mean_squared_error(y_pred, Ytest)}
            Root Mean Squared Error = {np.sqrt(mean_squared_error(y_pred, Ytest))},
            R2 Score = {r2_score(y_pred,Ytest)}
            """)

#Using the SGD Regressor
from sklearn.linear_model import SGDRegressor
sgd = SGDRegressor()

sgd.fit(Xtrain, Ytrain)

#saving as a pickle
pkl.dump(sgd, open(sgd.__class__.__name__+'.pkl','wb'))

#Evaluating the model
y_pred = sgd.predict(Xtest)
print(f"""  Mean Absolute Error = {mean_absolute_error(y_pred, Ytest)},
            Mean Squared Error = {mean_squared_error(y_pred, Ytest)}
            Root Mean Squared Error = {np.sqrt(mean_squared_error(y_pred, Ytest))},
            R2 Score = {r2_score(y_pred,Ytest)}
            """)

#Using a random forest regressor
from sklearn.ensemble import RandomForestRegressor

# Initializing the Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

rf_regressor.fit(Xtrain, Ytrain)

y_pred = rf_regressor.predict(Xtest)

#Run thissssssssssssssssssssssssssssssssssssssssssssss if the evaluation is good
#saving as a pickle
pkl.dump(rf_regressor, open(rf_regressor.__class__.__name__+'.pkl','wb'))

#Evaluating the model
print(f"""  Mean Absolute Error = {mean_absolute_error(y_pred, Ytest)},
            Mean Squared Error = {mean_squared_error(y_pred, Ytest)}
            Root Mean Squared Error = {np.sqrt(mean_squared_error(y_pred, Ytest))},
            R2 Score = {r2_score(y_pred,Ytest)}
            """)

#Using cross validation to train the random forest again
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# Initialize the RandomForestRegressor
rf_regressor = RandomForestRegressor(random_state=42)

# Initialize k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
cv_scores = cross_val_score(rf_regressor, Xtrain, Ytrain, cv=kf, scoring='neg_mean_squared_error')

# Print the cross-validation scores and their mean
print(f"Cross-Validation Scores (Negative MSE): {cv_scores}")
print(f"Mean Cross-Validation Score (Negative MSE): {np.mean(cv_scores)}")

# Fit the model on the entire dataset for comparison
rf_regressor.fit(Xtrain, Ytrain)





y_pred = rf_regressor.predict(Xtest)


pkl.dump(rf_regressor, open(rf_regressor.__class__.__name__+'.pkl','wb'))

#Evaluating the model
print(f"""  Mean Absolute Error = {mean_absolute_error(y_pred, Ytest)},
            Mean Squared Error = {mean_squared_error(y_pred, Ytest)}
            Root Mean Squared Error = {np.sqrt(mean_squared_error(y_pred, Ytest))},
            R2 Score = {r2_score(y_pred,Ytest)}
            """)

numeric_data = 0

#Importing relevant libraries to impute non numeric data
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
#Creating a function to prepare new data sets
def process_fifa_data(file_path):
    # Load the dataset
    fifa_data = pd.read_csv(file_path)
    #fifa_data.drop('dob', axis=1, inplace=True )

    
    # Dropping columns with more than 30% missing values
    less_null = []
    more_null = []
    for i in fifa_data.columns:
        if (fifa_data[i].isnull().sum()) < (0.3 * (fifa_data.shape[0])):
            less_null.append(i)
        else:
            more_null.append(i)
    fifa_data = fifa_data[less_null]

    # Splitting the data into numeric and non-numeric
    numeric_data = fifa_data.select_dtypes(include=np.number)
    non_numeric = fifa_data.select_dtypes(include=['object'])
    non_numeric.drop('nation_flag_url' , axis=1, inplace=True )
    #numeric_data.drop('mentality_vision', axis=1, inplace=True )

    



    # Finding relevant features based on correlation with 'overall'
    corr_matrix = numeric_data.corr()
    corr_matrix = corr_matrix["overall"].sort_values(ascending=False)
    relevant_features = corr_matrix[abs(corr_matrix) > 0.4].index
    numeric_data = numeric_data[relevant_features]
    irrelevant_features = corr_matrix[abs(corr_matrix) < 0.4].index

    new_relevant = irrelevant_features.difference(['player_id','sofifa_id' ,'nationality_id', 'league_id', 'club_team_id', 'club_jersey_number', 
                       'fifa_version', 'weight_kg', 'movement_balance', 'club_contract_valid_until_year', 
                       'height_cm', 'fifa_update','nation_team_id'])

    num = fifa_data[new_relevant]

    # Multivariate imputation for numeric data
    imp = IterativeImputer(max_iter=10, random_state=0)
    numeric_data = pd.DataFrame(np.round(imp.fit_transform(numeric_data)), columns=numeric_data.columns)
    numeric_data = pd.concat([numeric_data, num], axis=1)
    numeric_data = pd.DataFrame(np.round(imp.fit_transform(numeric_data)), columns=numeric_data.columns)

    # Dropping unnecessary non-numeric features
    unnecessary_objects = ['player_url', 'fifa_update_date', 'player_face_url', 'dob', 'short_name', 
                       'long_name', 'league_name', 'club_name', 'club_position', 'club_joined_date', 
                       'real_face', 'body_type']
    non_numeric.drop(columns = unnecessary_objects, axis=1, inplace=True,errors = 'ignore')
    

    # Creating a pipeline to impute the categorical variables with the mode
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
    ])
    
    # Creating a full pipeline for non-numeric data
    full_pipe = ColumnTransformer([
        ("cat", cat_pipe, make_column_selector(dtype_include=object))
    ])
    
    # Imputing and transforming the non-numeric data
    piped = full_pipe.fit_transform(non_numeric)
    non_numeric = pd.DataFrame(data=piped, columns=full_pipe.get_feature_names_out())
       
    columns_to_drop_2 = ['sofifa_id','player_url','player_face_url','dob','short_name', 'long_name',
                                  'league_name','club_team_id','club_jersey_number','club_loaned_from',
                                  'nationality_id','nation_team_id','nation_jersey_number','real_face',
                                  'body_type','release_clause_eur','player_tags','player_traits',
                                  'mentality_composure','nation_position', 'goalkeeping_speed','club_joined',
                                  'club_contract_valid_until' ]
    for i in columns_to_drop_2:
        if i in non_numeric:
            non_numeric.drop(i, axis=1, inplace=True )
    for i in columns_to_drop_2:
        if i in numeric_data:
            numeric_data.drop(i, axis=1, inplace=True )

    
    # Encoding the categorical variables
    encoder = ce.BinaryEncoder(cols=non_numeric.columns)
    non_numeric = encoder.fit_transform(non_numeric)

    # Putting numeric and non-numeric data together
    fifa_data = pd.concat([numeric_data, non_numeric], axis=1)

    return fifa_data




# Preparing the second dataset for testing
processed_data = process_fifa_data('players_22-1.csv')


processed_data

#processed_data.to_csv('data.csv', index=False)

#selecting new X for training and Y
X_new = processed_data.drop(columns=['overall'])  # Features
y_new = processed_data['overall']  # Target variable



#Scaling the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_new = scaler.fit_transform(X_new)


with open("RandomForestRegressor.pkl", 'rb') as file:
    best_model = pkl.load(file)

#Importing relevant metrics
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score
# Measure performance on the new dataset
new_predictions = best_model.predict(X_new)
#Evaluating the model
print(f"""  Mean Absolute Error = {mean_absolute_error(y_new, new_predictions)},
            Mean Squared Error = {mean_squared_error(y_new, new_predictions)}
            Root Mean Squared Error = {np.sqrt(mean_squared_error(y_new, new_predictions))},
            R2 Score = {r2_score(y_new,new_predictions)}
            """)

# Load the trained model
with open("RandomForestRegressor.pkl", 'rb') as file:
    model = pkl.load(file)



# Define the prediction function
def player_rating(features):
    prediction = model.predict(np.array(features).reshape(1, -1))
    return prediction[0]



import streamlit as st
# Streamlit app interface
st.title('Player Rating ')



# Input features from user
features = [ 'potential',  'mentality_vision','value_eur', 
    'wage_eur', 'age', 'league_level', 'weak_foot', 'skill_moves', 
    'international_reputation', 'pace', 'shooting', 'passing', 'dribbling', 
    'defending', 'physic', 'attacking_crossing', 'attacking_finishing', 
    'attacking_heading_accuracy', 'attacking_short_passing', 
    'attacking_volleys', 'skill_dribbling', 'skill_curve', 'skill_fk_accuracy', 
    'skill_long_passing', 'skill_ball_control', 'movement_acceleration', 
    'movement_sprint_speed', 'movement_agility', 'movement_reactions', 
    'power_shot_power', 'power_jumping', 'power_stamina', 'power_strength', 
    'power_long_shots', 'mentality_aggression', 'mentality_interceptions', 
    'mentality_positioning', 'mentality_penalties', 
    'mentality_composure', 'defending_marking_awareness', 
    'defending_standing_tackle', 'defending_sliding_tackle', 
    'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking', 
    'goalkeeping_positioning', 'goalkeeping_reflexes','cat_player_positions', 'catnationality_name', 'cat_preferred_foot', 
    'cat_work_rate', 'catls', 'catst', 'catrs', 'catlw', 'cat_lf', 
    'cat_cf', 'catrf', 'catrw', 'catlam', 'catcam', 'cat_ram', 
    'cat_lm', 'catlcm', 'catcm', 'catrcm', 'catrm', 'cat_lwb', 
    'cat_ldm', 'catcdm', 'catrdm', 'catrwb', 'catlb', 'cat_lcb', 
    'cat_cb', 'catrcb', 'catrb', 'cat_gk'
]


input_ = []

for i in features:
    value = st.number_input(f'Enter {i}', value=0.0)
    input_.append(value)



if st.button('Predict Rating'):
    rating = player_rating(input_data)
    st.write(f'Predicted Player Rating: {rating}')







