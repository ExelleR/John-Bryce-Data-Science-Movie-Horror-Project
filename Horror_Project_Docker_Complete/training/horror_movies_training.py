# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mt
import pickle

import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import sklearn as skl

from xgboost import XGBRegressor
import multiprocessing as mp
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
import numpy as np


import sweetviz as sv
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

import missingno as msno


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor



print("Matplotlib version:", mt.__version__)
print("Seaborn version:", sns.__version__)
print("NumPy version:", np.__version__)
print("Sklearn version:", skl.__version__)

# Load the dataset
file_path = './training/horror_movies.csv'  # Update with the correct path
df = pd.read_csv(file_path)
df.head()




del  df['id']

df['budget'].isnull()


df.info()

# import sweetviz as sv
# sweet_report = sv.analyze(df)
# sweet_report.show_html('horror_movies_sweetwiz_report.html')

pd.isnull(df).sum()[pd.isnull(df).sum() > 0]


categorical_columns = ['original_title', 'title', 'original_language', 'tagline', 'status', 'genre_names', 'collection_name']

df['original_language'].value_counts()

language_counts = df['original_language'].value_counts()

# Creating a seaborn bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x=language_counts.index, y=language_counts.values)
plt.title('Frequency of Each Language in the Dataset')
plt.xlabel('Original Language')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()

# TODO: change 1 in languages to Other category unitl we get 10 categories. - DONE


# Assuming df is your DataFrame and 'original_language' is the column name
top_9_languages = df['original_language'].value_counts().nlargest(9).index
df['original_language'] = df['original_language'].apply(lambda x: x if x in top_9_languages else 'Other')
df['original_language'].value_counts()


language_counts = df['original_language'].value_counts()

# Creating a seaborn bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x=language_counts.index, y=language_counts.values)
plt.title('Frequency of Each Language in the Dataset')
plt.xlabel('Original Language')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()

# Check the types of the rest fo the categories
df.columns





date_columns = ['release_date']
integer_columns = ['vote_count', 'budget', 'revenue', 'runtime', 'collection']

# Applying changes
for col in categorical_columns:
    if col in df.columns:
        df[col] = df[col].astype('category')

for col in date_columns:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')  # Coerce invalid dates to NaT

for col in integer_columns:
    if col in df.columns:
        if df[col].isna().any():
            df[col] = df[col].astype('Int64')  # Capital 'I' allows for NaNs
        else:
            df[col] = df[col].astype('int')

# Final data types after conversion
final_dtypes = df.dtypes

# Displaying the initial and final data types
# print("Initial Data Types:\n", initial_dtypes)
print("\nFinal Data Types:\n", final_dtypes)

df['original_title'] = df['original_title'].astype(str)
df['original_title'] = df['original_title'].astype(str)

df.info()

df.shape

df.describe()

pd.isnull(df).sum()[pd.isnull(df).sum() > 0]


# Type of valaues 
df.dtypes.to_excel("horror_movies_datatype.xlsx",
             sheet_name='data_type')
# Maximum valaues 
df.max(numeric_only=True).to_excel("max_horror_movies.xlsx",
             sheet_name='max')
# Minimum Values
df.min(numeric_only=True).to_excel("min_horror_movies.xlsx",
             sheet_name='min')
# Missing Values
df.isnull().sum(axis=0).to_excel("NA_horror_movies.xlsx",
             sheet_name='NA')
# exporting results to the protocol
df.nunique().to_excel("unique_horror_movies.xlsx",
             sheet_name='unique')

# import sweetviz as sv
# sweet_report = sv.analyze(df)
# sweet_report.show_html('horror_movies_sweetwiz_report.html')



df.to_csv("horror_movies_data_protocol.csv")

df.info()




# Assuming df is your DataFrame
# Identify numeric columns
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

# Create boxplots for each numeric column
plt.figure(figsize=(15, 5 * len(numeric_columns)))
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(len(numeric_columns), 1, i)
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot for {col}')

plt.tight_layout()
plt.show()

df['popularity'].count()

# df.to_csv("horror_movies_before_cleaning_outliers.csv")

df.columns

df.info()


# Selecting numerical columns for correlation analysis
numerical_data = df.select_dtypes(include=['float64', 'int64'])
numerical_data


pd.isnull(df).sum()[pd.isnull(df).sum() > 0]


# Calculating correlations
correlation_matrix = numerical_data.corr()
correlation_matrix



# Set up the matplotlib figure
fig, ax = plt.subplots(figsize=(10, 8))

# Draw the heatmap using the matshow function
cax = ax.matshow(correlation_matrix, cmap='coolwarm')
fig.colorbar(cax)

# Set ticks and labels for axes
ax.set_xticks(np.arange(len(correlation_matrix.columns)))
ax.set_yticks(np.arange(len(correlation_matrix.columns)))
ax.set_xticklabels(correlation_matrix.columns, rotation=90)
ax.set_yticklabels(correlation_matrix.columns)

# Loop over data dimensions and create text annotations.
for i in range(len(correlation_matrix.columns)):
    for j in range(len(correlation_matrix.columns)):
        ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                ha="center", va="center", color="w")

# Show the plot
plt.show()



## There is a hight coreelation between vote-count and revenue , revenue and budget 
## It seems like the more budge its put for the production of the horror film - the better is revenue for that film. 

df['popularity'].value_counts()

df['revenue'].value_counts() # We see that the most of the values in revenue is 0 - we going to replace them with NuN

df['revenue'] = df['revenue'].replace(0, np.nan)


df['revenue'].value_counts()

df['budget'].value_counts() # We see that the most of the values in budget is 0 - we going to replace them with NuN

df['budget'] = df['budget'].replace(0, np.nan)


df['budget'].value_counts()

pd.isnull(df).sum()[pd.isnull(df).sum() > 0]




df['popularity'].value_counts()

df['revenue'].value_counts()

df.shape


# Add new categories for categorical columns
if df['tagline'].dtype.name == 'category':
    df['tagline'] = df['tagline'].cat.add_categories('Unknown')
if df['collection_name'].dtype.name == 'category':
    df['collection_name'] = df['collection_name'].cat.add_categories('Unknown')

# Fill NaN values for categorical columns
df['tagline'].fillna('Unknown', inplace=True)
df['collection_name'].fillna('Unknown', inplace=True)

# Fill NaN values for 'collection' (treated as integer type)
df['collection'].fillna(-1, inplace=True)

# Fill NaN values for other columns
df['overview'].fillna('Not Available', inplace=True)
df['poster_path'].fillna('No Image Available', inplace=True)
df['backdrop_path'].fillna('No Image Available', inplace=True)

# Verify no more NaN values
nan_values_after = df.isna().sum()
print(nan_values_after)

df.head()

df.describe()

# Check null values that are left
pd.isnull(df).sum()[pd.isnull(df).sum() > 0]


df.to_csv("horror_movies_data_protocol_end.csv")

# Calculate median values
budget_median = df['budget'].median()
revenue_median = df['revenue'].median()

# Impute missing values with median
df['budget'].fillna(budget_median, inplace=True)
df['revenue'].fillna(revenue_median, inplace=True)


pd.isnull(df).sum()[pd.isnull(df).sum() > 0]


df


# Extracting year from the release_date column and creating a new column "release_year"
df['release_year'] = pd.to_datetime(df['release_date']).dt.year

# Display the first few rows to verify the new column
df



# import sweetviz as sv
# sweet_report = sv.analyze(df)
# sweet_report.show_html('horror_movies_sweetwiz_report.html')

df.columns

df.describe()

df.to_csv("horror_movies_correlations.csv")

df.info()

df['original_language'].value_counts()



language_counts = df['original_language'].value_counts()

# Creating a seaborn bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x=language_counts.index, y=language_counts.values)
plt.title('Frequency of Each Language in the Dataset')
plt.xlabel('Original Language')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()

df.info()


# Column	Number of Unique Values
# Unnamed: 0	        32,540
# original_title	    30,296
# title	                29,563
# original_language	    97
# overview        	    31,020
# tagline	            12,514
# release_date	        10,999
# poster_path	        28,048
# popularity	        7,250
# vote_count	        1,105
# vote_average	        92
# budget	            783
# revenue	            1,427
# runtime	            212
# status	            4
# adult	                1
# backdrop_path	        13,536
# genre_names	        772
# collection	        815
# collection_name	    815



df

# Show missing values in dataset 


# Create a boolean DataFrame where True indicates a missing value
missing = df.isnull()

# Use Seaborn to plot a heatmap to visualize the missing data
sns.heatmap(missing, cbar=False, cmap='viridis', yticklabels=False)


df['budget'].isnull()


pd.isnull(df).sum()[pd.isnull(df).sum() > 0]


df.info()

# Identifying categorical columns to be one-hot encoded
categorical_columns = ['title', 'original_language', 'tagline', 'status','genre_names']

# One-hot encoding the identified categorical columns
df = pd.get_dummies(df, columns=categorical_columns)

# Display the first few rows to verify the encoding


df

df.loc[(df['popularity'] >= 50)] 

df['popularity'] = np.log1p(df['popularity'])




# Selecting all numerical features except 'popularity'
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.drop('popularity')
X = df[numeric_features]
y = df['popularity']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing a dictionary of models
models = {
    "Linear Regression": LinearRegression(n_jobs=-1),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Support Vector Regressor": SVR(),
    "K-Neighbors Regressor": KNeighborsRegressor(n_jobs=-1),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Elastic Net Regression": ElasticNet(),
    "XGBoost Regressor": xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
}

# Dictionary to store MSE scores and predictions for visualization
mse_scores = {}
predictions = {}

# Training, evaluating each model, and collecting predictions
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_scores[name] = mse
    predictions[name] = y_pred

# Choosing a model's prediction for visualization, e.g., the best performing model
selected_model_name = min(mse_scores, key=mse_scores.get)
selected_y_pred = predictions[selected_model_name]

selected_model_name


print("MSE Score :")

for model_name, score in mse_scores.items():
    print(f"- {model_name}: {score}")







# Plotting the scatterplot for actual vs. predicted values
plt.figure(figsize=(10, 7), )
plt.scatter(y_test, selected_y_pred, alpha=0.5, s=5)
plt.title(f'Actual vs. Predicted Values - {selected_model_name}')
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2) # Diagonal line for reference
plt.show()


feature_names = X.columns


# Get feature importances
feature_importances = model.feature_importances_

# Sort the feature importances in descending order
indices = np.argsort(feature_importances)[::-1]

# Print the rankings
print("Feature ranking:")
for f in range(X.shape[1]):
    print(f"{f + 1}. feature {feature_names[indices[f]]} ({feature_importances[indices[f]]})")

# Identifying the highest and lowest importance features
highest_importance_feature = feature_names[indices[0]]
lowest_importance_feature = feature_names[indices[-1]]

print(f"Highest importance feature: {highest_importance_feature}")
print(f"Lowest importance feature: {lowest_importance_feature}")


# Plotting the histogram for the 'popularity' column
plt.figure(figsize=(10, 6))  # Set the figure size
plt.hist(df['popularity'], bins=30, color='skyblue', edgecolor='black')  # Histogram with 30 bins
plt.title('Histogram of Popularity')  # Title of the histogram
plt.xlabel('Popularity')  # X-axis label
plt.ylabel('Frequency')  # Y-axis label
plt.grid(axis='y', alpha=0.75)  # Add grid lines on the y-axis for better readability

plt.show()  # Display the histogram



# # Assuming your dataset is loaded into a DataFrame called df
# # Prepare your features (X) and target variable (y)


# # Define the parameter grid to search
# param_grid = {
#     'n_estimators': [100, 200, 300, 500],
#     'learning_rate': [0.01,0.001,0.0001, 0.1 ],
#     'max_depth': [3, 4, 5, 8, 10, 12, 14, 16, 18, 20],
#     'min_samples_split': [1 , 2, 3],
#     'min_samples_leaf': [1, 2],
# }

# # Initialize the GridSearchCV object
# grid_search = GridSearchCV(
#     estimator=GradientBoostingRegressor(),
#     param_grid=param_grid,
#     cv=3,  # Number of folds in cross-validation
#     scoring='neg_mean_squared_error',  # Can change based on your objective
#     n_jobs=-1,  # Use all available CPUs
#     verbose=2,  # Higher number gives more messages about the process
# )

# # Fit GridSearchCV
# grid_search.fit(X, y)

# # Best parameters and best score
# print("Best parameters:", grid_search.best_params_)
# print("Best score (negative MSE):", grid_search.best_score_)

# # You can also inspect the full set of results
# results = pd.DataFrame(grid_search.cv_results_)


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



scaler = StandardScaler()
# Fit the scaler to the data and transform the data
scaled_data_train = scaler.fit_transform(X_train)
scaled_x_test = scaler.transform(X_test)

# Use the optimal hyperparameters to train the model
optimal_params = {
    'max_iter': 200,  # Number of boosting iterations
    'learning_rate': 0.1,  # Learning rate
    'max_depth': 3,  # Maximum tree depth
    'min_samples_leaf': 20,  # Minimum number of samples per leaf
    'l2_regularization': 0.0  # L2 regularization
}

model = HistGradientBoostingRegressor(**optimal_params, random_state=42)
model.fit(scaled_data_train, y_train)

# Predict and calculate the MSE on the test set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(f"MSE on test set: {mse}")


print("Saving the trained model to /usr/src/app/trained_model.pkl")

# Assuming 'model' is your trained model variable
model_filename = './trained_model.pkl'

# Save the model to disk
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)

print(f"Model saved to {model_filename}")




print("Saving the Scaler object  to /usr/src/app/scaler_object.pkl")

# Assuming 'model' is your trained model variable
scaler_object_filename = './scaler_object.pkl'

# Save the model to disk
with open(scaler_object_filename, 'wb') as file:
    pickle.dump(scaler, file)

print(f"Scaler saved to {scaler_object_filename}")

