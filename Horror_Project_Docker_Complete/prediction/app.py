from flask import Flask, render_template, request
import numpy as np
import pickle

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mt
import seaborn as sns
import logging
from flask import Flask, request, jsonify


"""
try:
    block
except Exception(e):
    logger.exception(e)
    raise(e)
"""

import matplotlib.pyplot as plt

app = Flask(__name__)

# Load your model
model = pickle.load(open('./training/trained_model.pkl', 'rb'))
scaler = pickle.load(open('./training/scaler_object.pkl', 'rb'))


def preprocess_input(form_data):
    # List of features expected by the model
    expected_features = ['id', 'original_title', 'title', 'original_language', 'overview']
    
    # Initialize a list to hold processed data
    processed_data = []

    ######################################################################## START - Preprocess ######################################################################


    # processed_data = np.array([[8922, "Jeepers Creepers", "Jeepers Creepers" , "en","A brother and sister driving home through an isolated countryside from college encounter a flesh-eating creature in the midst of its ritualistic eating spree."]])

   

    processed_data = pd.DataFrame(form_data)


    # Importing necessary libraries



    print("Matplotlib version:", mt.__version__)
    print("Seaborn version:", sns.__version__)
    print("NumPy version:", np.__version__)

    # Load the dataset
    # file_path = './horror_movies.csv'  # Update with the correct path
    # processed_data = pd.read_csv(file_path)
    processed_data.head()

    del  processed_data['id']

    processed_data['budget'].isnull()


    processed_data.info()

    # import sweetviz as sv
    # sweet_report = sv.analyze(processed_data)
    # sweet_report.show_html('horror_movies_sweetwiz_report.html')

    pd.isnull(processed_data).sum()[pd.isnull(processed_data).sum() > 0]


    categorical_columns = ['original_title', 'title', 'original_language', 'tagline', 'status', 'genre_names', 'collection_name']

    processed_data['original_language'].value_counts()

    language_counts = processed_data['original_language'].value_counts()

    # Creating a seaborn bar plot
    # plt.figure(figsize=(10, 6))
    # sns.barplot(x=language_counts.index, y=language_counts.values)
    # plt.title('Frequency of Each Language in the Dataset')
    # plt.xlabel('Original Language')
    # plt.ylabel('Frequency')
    # plt.xticks(rotation=45)
    # plt.show()

    # TODO: change 1 in languages to Other category unitl we get 10 categories. - DONE


    # Assuming processed_data is your DataFrame and 'original_language' is the column name
    top_9_languages = processed_data['original_language'].value_counts().nlargest(9).index
    processed_data['original_language'] = processed_data['original_language'].apply(lambda x: x if x in top_9_languages else 'Other')
    processed_data['original_language'].value_counts()


    language_counts = processed_data['original_language'].value_counts()

    # Creating a seaborn bar plot
    # plt.figure(figsize=(10, 6))
    # sns.barplot(x=language_counts.index, y=language_counts.values)
    # plt.title('Frequency of Each Language in the Dataset')
    # plt.xlabel('Original Language')
    # plt.ylabel('Frequency')
    # plt.xticks(rotation=45)
    # plt.show()

    # Check the types of the rest fo the categories
    processed_data.columns





    date_columns = ['release_date']
    integer_columns = ['vote_count', 'budget', 'revenue', 'runtime', 'collection']

    # Applying changes
    for col in categorical_columns:
        if col in processed_data.columns:
            processed_data[col] = processed_data[col].astype('category')

    for col in date_columns:
        if col in processed_data.columns:
            processed_data[col] = pd.to_datetime(processed_data[col], errors='coerce')  # Coerce invalid dates to NaT

    for col in integer_columns:
        if col in processed_data.columns:
            if processed_data[col].isna().any():
                processed_data[col] = processed_data[col].astype('Int64')  # Capital 'I' allows for NaNs
            else:
                processed_data[col] = processed_data[col].astype('int')

    # Final data types after conversion
    final_dtypes = processed_data.dtypes

    # Displaying the initial and final data types
    # print("Initial Data Types:\n", initial_dtypes)
    print("\nFinal Data Types:\n", final_dtypes)

    processed_data['original_title'] = processed_data['original_title'].astype(str)
    processed_data['original_title'] = processed_data['original_title'].astype(str)

    processed_data.info()

    processed_data.shape

    processed_data.describe()

    pd.isnull(processed_data).sum()[pd.isnull(processed_data).sum() > 0]



    # import sweetviz as sv
    # sweet_report = sv.analyze(processed_data)
    # sweet_report.show_html('horror_movies_sweetwiz_report.html')




    processed_data.info()




    # Assuming processed_data is your DataFrame
    # Identify numeric columns
    numeric_columns = processed_data.select_dtypes(include=['int64', 'float64']).columns

    # Create boxplots for each numeric column
    # plt.figure(figsize=(15, 5 * len(numeric_columns)))
    # for i, col in enumerate(numeric_columns, 1):
    #     plt.subplot(len(numeric_columns), 1, i)
    #     sns.boxplot(x=processed_data[col])
    #     plt.title(f'Boxplot for {col}')

    # plt.tight_layout()
    # plt.show()

    processed_data['popularity'].count()


    processed_data.columns

    processed_data.info()


    # Selecting numerical columns for correlation analysis
    numerical_data = processed_data.select_dtypes(include=['float64', 'int64'])
    numerical_data


    pd.isnull(processed_data).sum()[pd.isnull(processed_data).sum() > 0]


    # Calculating correlations
    correlation_matrix = numerical_data.corr()
    correlation_matrix



    # Set up the matplotlib figure
    # fig, ax = plt.subplots(figsize=(10, 8))

    # # Draw the heatmap using the matshow function
    # cax = ax.matshow(correlation_matrix, cmap='coolwarm')
    # fig.colorbar(cax)

    # # Set ticks and labels for axes
    # ax.set_xticks(np.arange(len(correlation_matrix.columns)))
    # ax.set_yticks(np.arange(len(correlation_matrix.columns)))
    # ax.set_xticklabels(correlation_matrix.columns, rotation=90)
    # ax.set_yticklabels(correlation_matrix.columns)

    # # Loop over data dimensions and create text annotations.
    # for i in range(len(correlation_matrix.columns)):
    #     for j in range(len(correlation_matrix.columns)):
    #         ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
    #                 ha="center", va="center", color="w")

    # # Show the plot
    # plt.show()



    ## There is a hight coreelation between vote-count and revenue , revenue and budget 
    ## It seems like the more budge its put for the production of the horror film - the better is revenue for that film. 

    processed_data['popularity'].value_counts()

    processed_data['revenue'].value_counts() # We see that the most of the values in revenue is 0 - we going to replace them with NuN

    processed_data['revenue'] = processed_data['revenue'].replace(0, np.nan)


    processed_data['revenue'].value_counts()

    processed_data['budget'].value_counts() # We see that the most of the values in budget is 0 - we going to replace them with NuN

    processed_data['budget'] = processed_data['budget'].replace(0, np.nan)


    processed_data['budget'].value_counts()

    pd.isnull(processed_data).sum()[pd.isnull(processed_data).sum() > 0]




    processed_data['popularity'].value_counts()

    processed_data['revenue'].value_counts()

    processed_data.shape


    # Add new categories for categorical columns
    if processed_data['tagline'].dtype.name == 'category':
        processed_data['tagline'] = processed_data['tagline'].cat.add_categories('Unknown')
    if processed_data['collection_name'].dtype.name == 'category':
        processed_data['collection_name'] = processed_data['collection_name'].cat.add_categories('Unknown')

    # Fill NaN values for categorical columns
    processed_data['tagline'].fillna('Unknown', inplace=True)
    processed_data['collection_name'].fillna('Unknown', inplace=True)

    # Fill NaN values for 'collection' (treated as integer type)
    processed_data['collection'].fillna(-1, inplace=True)

    # Fill NaN values for other columns
    processed_data['overview'].fillna('Not Available', inplace=True)
    processed_data['poster_path'].fillna('No Image Available', inplace=True)
    processed_data['backdrop_path'].fillna('No Image Available', inplace=True)

    # Verify no more NaN values
    nan_values_after = processed_data.isna().sum()
    print(nan_values_after)

    processed_data.head()

    processed_data.describe()

    # Check null values that are left
    pd.isnull(processed_data).sum()[pd.isnull(processed_data).sum() > 0]



    # Calculate median values
    budget_median = processed_data['budget'].median()
    revenue_median = processed_data['revenue'].median()

    # Impute missing values with median
    processed_data['budget'].fillna(budget_median, inplace=True)
    processed_data['revenue'].fillna(revenue_median, inplace=True)


    pd.isnull(processed_data).sum()[pd.isnull(processed_data).sum() > 0]


    processed_data


    # Extracting year from the release_date column and creating a new column "release_year"
    processed_data['release_year'] = pd.to_datetime(processed_data['release_date']).dt.year

    # Display the first few rows to verify the new column
    processed_data



    # import sweetviz as sv
    # sweet_report = sv.analyze(processed_data)
    # sweet_report.show_html('horror_movies_sweetwiz_report.html')

    processed_data.columns

    processed_data.describe()


    processed_data.info()

    processed_data['original_language'].value_counts()



    language_counts = processed_data['original_language'].value_counts()

    # Creating a seaborn bar plot
    # plt.figure(figsize=(10, 6))
    # sns.barplot(x=language_counts.index, y=language_counts.values)
    # plt.title('Frequency of Each Language in the Dataset')
    # plt.xlabel('Original Language')
    # plt.ylabel('Frequency')
    # plt.xticks(rotation=45)
    # plt.show()

    processed_data.info()


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



    processed_data

    # Show missing values in dataset 


    # Create a boolean DataFrame where True indicates a missing value
    missing = processed_data.isnull()

    # Use Seaborn to plot a heatmap to visualize the missing data
    # sns.heatmap(missing, cbar=False, cmap='viridis', yticklabels=False)


    processed_data['budget'].isnull()


    pd.isnull(processed_data).sum()[pd.isnull(processed_data).sum() > 0]


    processed_data.info()

    # Identifying categorical columns to be one-hot encoded
    categorical_columns = ['title', 'original_language', 'tagline', 'status','genre_names']

    # One-hot encoding the identified categorical columns
    processed_data = pd.get_dummies(processed_data, columns=categorical_columns)

    # Display the first few rows to verify the encoding


    processed_data

    processed_data.loc[(processed_data['popularity'] >= 50)] 

    processed_data['popularity'] = np.log1p(processed_data['popularity'])

    numeric_features = processed_data.select_dtypes(include=['int64', 'float64']).columns.drop('popularity')
    processed_data = processed_data[numeric_features]
    
    
    
    # processed_data.head()
    # processed_data.columns
    


    ######################################################################## END - Preprocess ########################################################################

    
    # Example preprocessing for demonstration
    
    return np.array(processed_data).reshape(1, -1)
    # return processed_data


def make_prediction(processed_data):
    
    # TODO: Add scaling of the processed_data before predict method 
    
  

    # Use the loaded model to make predictions
    processed_data =  scaler.transform(processed_data)
    
    prediction = model.predict(processed_data)
    
    prediction = np.expm1(prediction)
    
    return prediction


    # return processed_data[0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # form_data = request.form.to_dict()


    # form_data = {
    # 'id': [24038],
    # 'original_title': ["976-EVIL"],
    # 'title': ["976-EVIL"],
    # 'original_language': ["en"],
    # 'overview': ["People who dial 976-EVIL receive supernatural powers and turn into satanic killers. When Spike dialed 976-EVIL, he knew it was an expensive toll call, but he didn't know that he'd have to pay for it with his soul."],
    # 'tagline': ["Now, horror has a brand new number."],
    # 'release_date': ["09/12/1988"],
    # 'poster_path': ["/mWSpulrInC2DjbXhysjrxtvJFfL.jpg"],
    # 'popularity': [7.324],
    # 'vote_count': ["89"],
    # 'vote_average': [5.1],
    # 'budget': [0],
    # 'revenue': [2955917],
    # 'runtime': [92],
    # 'status': ["Released"],
    # 'adult': ["FALSE"],
    # 'backdrop_path': ["/mk1KeasAfwUGfFfzlbUVKINxC7Q.jpg"],
    # 'genre_names': ["Horror"],
    # 'collection': [135501],
    # 'collection_name': ["976-EVIL Collection"]
    # }
    
    
    
    # data = request.form.to_dict()
    # form_data = {k: [v] for k, v in data.items()}  # Convert to the correct format
    
    
     # Convert form data to a dictionary
    data = request.form.to_dict()
    
    # Convert dictionary to DataFrame
    data_df = pd.DataFrame([data])
    
    # Ensure numerical fields are of the correct data type
    num_fields = ['popularity', 'vote_average', 'budget', 'revenue', 'runtime', 'id', 'vote_count']
    for field in num_fields:
        data_df[field] = pd.to_numeric(data_df[field], errors='coerce')
    
    
    processed_data = preprocess_input(data_df)
  
    prediction_result = make_prediction(processed_data)
    # return render_template('index.html', prediction_text=f'Prediction Result: {prediction_result}')
    return jsonify({'prediction': prediction_result.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
