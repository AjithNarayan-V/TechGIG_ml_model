import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle

# Load the dataset into a DataFrame
data = pd.read_csv(r"C:\Users\NANDHINI\Downloads\Doceree-HCP_Train.csv",engine='python',encoding='latin1')

#Shuffling the dataset
data= shuffle(data, random_state=42)
data.reset_index(drop=True, inplace=True)

# Select the relevant columns and create the feature matrix X
data.drop(['ID', 'BIDREQUESTIP', 'PLATFORMTYPE', 'CHANNELTYPE', 'URL','PLATFORMTYPE','PLATFORM_ID'], axis=1, inplace=True)

# Encode the categorical features into numerical values
label_encoder = LabelEncoder()
data['DEVICETYPE'] = label_encoder.fit_transform(data['DEVICETYPE'])
data['USERAGENT'] = label_encoder.fit_transform(data['USERAGENT'])
#data['PLATFORM_ID'] = label_encoder.fit_transform(data['PLATFORM_ID'])
data['USERPLATFORMUID'] = label_encoder.fit_transform(data['USERPLATFORMUID'])
data['USERCITY'] = label_encoder.fit_transform(data['USERCITY'])
data['KEYWORDS'] = label_encoder.fit_transform(data['KEYWORDS'])
data['TAXONOMY'] = label_encoder.fit_transform(data['TAXONOMY'])

# Specify the columns with missing values
columns_with_missing_values = ['IS_HCP', 'TAXONOMY','DEVICETYPE','USERAGENT','USERPLATFORMUID','USERCITY','USERZIPCODE','KEYWORDS']

# Create an instance of SimpleImputer with strategy='mean'
imputer = SimpleImputer(strategy='most_frequent')

# Fit the imputer on the columns with missing values
imputer.fit(data[columns_with_missing_values])

# Transform the data by replacing missing values with the mean value
data[columns_with_missing_values] = imputer.transform(data[columns_with_missing_values])

# Verify if there are any missing values after imputation
print("NaN values in present in dataset:\n",data.isnull().sum())

# Create the target variable y for 'Taxonomy' and 'IS_HCP'
X = data.drop(['TAXONOMY','IS_HCP'], axis=1)
y_taxonomy = data['TAXONOMY']
y_is_hcp = data['IS_HCP']

# Split the data into training and testing sets
X_train, X_test, y_taxonomy_train, y_taxonomy_test, y_is_hcp_train, y_is_hcp_test = train_test_split(X, y_taxonomy, y_is_hcp, test_size=0.2, random_state=42)

# Create an instance of the Support Vector Regression model
rf_taxonomy = RandomForestClassifier()
rf_is_hcp = RandomForestClassifier()

# Train the model for 'Taxonomy'
rf_taxonomy.fit(X_train, y_taxonomy_train)

# Train the model for 'IS_HCP'
rf_is_hcp.fit(X_train, y_is_hcp_train)

# Predict 'Taxonomy' and 'IS_HCP' values for the test set
y_taxonomy_pred = rf_taxonomy.predict(X_test)
y_is_hcp_pred = rf_is_hcp.predict(X_test)

#Printing accuracy Score
accuracy_taxonomy_pred = accuracy_score(y_taxonomy_test, y_taxonomy_pred)
accuracy_hcp_pred = accuracy_score(y_is_hcp_test, y_is_hcp_pred)
print('accuracy_taxonomy_pred:' ,accuracy_taxonomy_pred*100)
print('accuracy_hcp_pred:' ,accuracy_hcp_pred*100)


#Loading testing dataset
x_test = pd.read_csv(r"C:\Users\NANDHINI\Downloads\Doceree-HCP_Test.csv",engine='python',encoding='latin1')
ID=x_test['ID']
x_test.drop([ 'ID','BIDREQUESTIP', 'PLATFORMTYPE', 'CHANNELTYPE', 'URL','PLATFORM_ID'], axis=1, inplace=True)

# Encode the categorical features into numerical values
x_test['DEVICETYPE'] = label_encoder.fit_transform(x_test['DEVICETYPE'])
x_test['USERAGENT'] = label_encoder.fit_transform(x_test['USERAGENT'])
#x_test['PLATFORM_ID'] = label_encoder.fit_transform(x_test['PLATFORM_ID'])
x_test['USERPLATFORMUID'] = label_encoder.fit_transform(x_test['USERPLATFORMUID'])
x_test['USERCITY'] = label_encoder.fit_transform(x_test['USERCITY'])
x_test['KEYWORDS'] = label_encoder.fit_transform(x_test['KEYWORDS'])


# Specify the columns with missing values
columns_with_missing_values = ['DEVICETYPE','USERAGENT','USERPLATFORMUID','USERCITY','USERZIPCODE','KEYWORDS']

# Create an instance of SimpleImputer with strategy='mean'
imputer = SimpleImputer(strategy='most_frequent')

# Fit the imputer on the columns with missing values
imputer.fit(x_test[columns_with_missing_values])

# Transform the data by replacing missing values with the mean value
x_test[columns_with_missing_values] = imputer.transform(x_test[columns_with_missing_values])

# Verify if there are any missing values after imputation
print("NaN values in 'IS_HCP','TAXANOMY':\n",x_test.isnull().sum())

#Prediction for Test data
predictions_IS_HCP = rf_is_hcp.predict(x_test)
predictions_IS_TAXANOMY = rf_taxonomy.predict(x_test)

output_data = pd.DataFrame({
     'ID': ID, 
    'IS_HCP': predictions_IS_HCP, 
})
# Replace 'scoring_solution.csv' with your desired file name
output_data.to_csv('scoring_solution1.csv', index=False)