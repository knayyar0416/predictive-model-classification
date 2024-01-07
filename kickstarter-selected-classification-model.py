# Import libraries
import pandas as pd
import numpy as np

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(color_codes=True)
custom_colors = ["#ebdc78", "#63bff0", "#1984c5", "#54bebe", "#df979e", "#d7658b", "#ffd3b6", "#ee4035"]

# Import dataset
kickstarter_df = pd.read_excel('kickstarter.xlsx') # 15474 rows x 45 columns

# ==============================================================================    
# Data preprocessing and exploration
# ==============================================================================
# Remove the 'currency' column, since it is the same as 'country' column
kickstarter_df = kickstarter_df.drop(columns=['currency'], axis=1) # 15474 rows x 44 columns
# Replace non-US countries in country column with 'non-US'
kickstarter_df.loc[kickstarter_df['country'] != 'US', 'country'] = 'Non-US' # 15474 rows x 44 columns

# Drop name_len and blurb_len, since we already have clean versions!
kickstarter_df = kickstarter_df.drop(columns=['name_len', 'blurb_len'], axis=1) # 15474 rows x 42 columns
# Replace null values in name_len_clean with 0
kickstarter_df['name_len_clean'] = kickstarter_df['name_len_clean'].fillna(0)
# Replace null values in blurb_len_clean with 0
kickstarter_df['blurb_len_clean'] = kickstarter_df['blurb_len_clean'].fillna(0)

# Drop pledged, since we already have usd_pledged
kickstarter_df = kickstarter_df.drop(columns=['pledged'], axis=1) # 15474 rows x 41 columns

# Create new column 'goal_usd' by multiplying 'goal' and 'static_usd_rate'
kickstarter_df['goal_usd'] = kickstarter_df['goal'] * kickstarter_df['static_usd_rate'] # 15474 rows x 42 columns
# Remove columns 'goal' and 'static_usd_rate'
kickstarter_df = kickstarter_df.drop(columns=['goal', 'static_usd_rate'], axis=1) # 15474 rows x 40 columns

# Remove irrelevant columns
irrelevant_columns = ['id', 'name', 'deadline_hr', 'created_at_hr', 'launched_at_hr']
kickstarter_df = kickstarter_df.drop(columns=irrelevant_columns, axis=1) # 15474 rows x 35 columns

# Check for missing values
kickstarter_df.isnull().sum() # 1392 missing values in 'category' column
# If category is missing, then replace with "No category"
kickstarter_df['category'] = kickstarter_df['category'].fillna('No category') # 15474 rows x 35 columns

# Drop 'canceled' and 'suspended' from state column
kickstarter_df = kickstarter_df[kickstarter_df['state'] != 'canceled'] # 13602 rows x 35 columns
kickstarter_df = kickstarter_df[kickstarter_df['state'] != 'suspended'] # 13435 rows x 35 columns
# Change 'successful' to 1 and 'failed' to 0
kickstarter_df['state'] = kickstarter_df['state'].replace(['successful', 'failed'], [1, 0])

# Remove original date columns
date_columns = ['deadline', 'created_at', 'launched_at']
kickstarter_df = kickstarter_df.drop(columns=date_columns, axis=1) # 13435 rows x 32 columns

# Remove weekday columns
weekday_columns = ['deadline_weekday', 'created_at_weekday', 'launched_at_weekday']
kickstarter_df = kickstarter_df.drop(columns=weekday_columns, axis=1) # 13435 rows x 29 columns

# ==============================================================================
# The classification task is assumed to be done at the time each project is launched. In other
# words, we execute the model to predict whether a new project is going to be successful or not, at the moment
# when the project owner submits the project. Therefore, the model should only use the predictors that are
# available at the moment when a new project is launched.

# Columns that are not available at the time of launching a project
columns_not_available = ['disable_communication', 'state_changed_at', 'staff_pick', 
                         'backers_count', 'usd_pledged', 'spotlight', 'state_changed_at_weekday', 
                         'state_changed_at_month', 'state_changed_at_day', 'state_changed_at_yr', 
                         'state_changed_at_hr', 'launch_to_state_change_days']
# Remove columns that are not available at the time of launching a project
kickstarter_df = kickstarter_df.drop(columns=columns_not_available, axis=1) # 13435 rows x 17 columns

# X predictors
X = kickstarter_df.loc[:,kickstarter_df.columns!='state'] # 13435 rows x 16 columns
# Target variable
y = kickstarter_df['state'] # 13435 rows x 1 column

# Dummify categorical variables
dummify_cols = ['category', 'country']
X = pd.get_dummies(X, columns=dummify_cols) # 13435 rows x 39 columns

# Correlation
c = X.corr()
# Display the correlation matrix
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(c)

# List of highly correlated features
corr_features = []
# Set a threshold for correlation
threshold = 0.8
# Iterate through the correlation matrix and identify highly correlated features
for i in range(len(c.columns)):
    for j in range(i):
        if abs(c.iloc[i, j]) >= threshold:
            colname = c.columns[i]
            corr_features.append(colname)
# Display the highly correlated features
print("Highly Correlated Features:")
print(corr_features)
correlated_features = ['created_at_yr', 'launched_at_yr', 'country_Non-US']

# Revise X dataset
X = X.drop(columns=correlated_features, axis=1) # 13435 rows x 36 columns

# ==============================================================================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix

# Validation set approach: Split the standardized dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 5)
# I tried different test_sizes but I found 0.33 to be the best

# Create standardized training and test sets
standardizer = StandardScaler()
X_train_std = standardizer.fit_transform(X_train)
X_test_std = standardizer.transform(X_test)

# ==============================================================================
# Selected Model: Gradient Boosting Algorithm
# ==============================================================================
from sklearn.ensemble import GradientBoostingClassifier
# Build the model
gbt = GradientBoostingClassifier(random_state=0)                           
model_gbt = gbt.fit(X_train, y_train)
# Make prediction and evaluate accuracy
y_test_pred_gbt = model_gbt.predict(X_test)
# Performance measures
accuracy_gbt = accuracy_score(y_test, y_test_pred_gbt)
precision_gbt = precision_score(y_test, y_test_pred_gbt)
recall_gbt = recall_score(y_test, y_test_pred_gbt)
f1_gbt = f1_score(y_test, y_test_pred_gbt)
auc_gbt = roc_auc_score(y_test, y_test_pred_gbt)
conf_matrix_gbt = confusion_matrix(y_test, y_test_pred_gbt)
# Print the results
print(f"Accuracy of Gradient Boosting Model is: {accuracy_gbt*100:.2f}%") # 75.30%
print(f"Precision of Gradient Boosting Model is: {precision_gbt*100:.2f}%") # 68.45%
print(f"Recall of Gradient Boosting Model is: {recall_gbt*100:.2f}%") # 53.26%
print(f"F1 Score of Gradient Boosting Model is: {f1_gbt*100:.2f}%") # 59.90%
print(f"AUC of Gradient Boosting Model is: {auc_gbt*100:.2f}%") # 70.12%
print("Confusion Matrix of Gradient Boosting Model is:")
print(conf_matrix_gbt)
# K-fold cross-validation with different number of samples required to split
from sklearn.model_selection import cross_val_score
for i in range (2,10):                                                                        
    model2 = GradientBoostingClassifier(random_state=0,min_samples_split=i,n_estimators=100)
    scores = cross_val_score(estimator=model2, X=X, y=y, cv=5)
    print(i,':',np.average(scores))

# ==============================================================================
# Prediction on kickstarter-testing-dataset.xlsx
# ==============================================================================
# Import test data
kickstarter_test_df = pd.read_excel("kickstarter-test-dataset.xlsx")
# Display the shape of the dataset
kickstarter_test_df.shape # 2000 rows x 45 columns
# Display the names of all columns
kickstarter_test_df.columns
# Check for data types
kickstarter_test_df.info()
# Display the first 5 rows of the dataset
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(kickstarter_test_df.head(5))
# Descriptive statistics for numerical features    
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print (kickstarter_test_df.describe())

# ==============================================================================    
# Pre-processing
# ==============================================================================
# Remove the 'currency' column, since it is the same as 'country' column
kickstarter_test_df = kickstarter_test_df.drop(columns=['currency'], axis=1) # 2000 rows x 44 columns
# Replace non-US countries in country column with 'non-US'
kickstarter_test_df.loc[kickstarter_test_df['country'] != 'US', 'country'] = 'Non-US' # 2000 rows x 44 columns
# Drop name_len and blurb_len, since we already have clean versions!
kickstarter_test_df = kickstarter_test_df.drop(columns=['name_len', 'blurb_len'], axis=1) # 2000 rows x 42 columns
# Drop pledged, since we already have usd_pledged
kickstarter_test_df = kickstarter_test_df.drop(columns=['pledged'], axis=1) # 2000 rows x 41 columns

# Create new column 'goal_usd' by multiplying 'goal' and 'static_usd_rate'
kickstarter_test_df['goal_usd'] = kickstarter_test_df['goal'] * kickstarter_test_df['static_usd_rate'] # 2000 rows x 42 columns
# Remove columns 'goal' and 'static_usd_rate'
kickstarter_test_df = kickstarter_test_df.drop(columns=['goal', 'static_usd_rate'], axis=1) # 2000 rows x 40 columns

# Remove irrelevant columns
irrelevant_columns = ['id', 'name', 'deadline_hr', 'created_at_hr', 'launched_at_hr']
kickstarter_test_df = kickstarter_test_df.drop(columns=irrelevant_columns, axis=1) # 2000 rows x 35 columns
# Check for duplicates
kickstarter_test_df[kickstarter_test_df.duplicated()].shape # (0, 35)
# Check for missing values
kickstarter_test_df.isnull().sum()
# If category is missing, then replace with "No category"
kickstarter_test_df['category'] = kickstarter_test_df['category'].fillna('No category') # 2000 rows x 35 columns

# Drop 'canceled' and 'suspended' from state column
kickstarter_test_df = kickstarter_test_df[kickstarter_test_df['state'] != 'canceled'] # 1772 rows x 35 columns
kickstarter_test_df = kickstarter_test_df[kickstarter_test_df['state'] != 'suspended'] # 1750 rows x 35 columns

# Remove original date columns
date_columns = ['deadline', 'created_at', 'launched_at']
kickstarter_test_df = kickstarter_test_df.drop(columns=date_columns, axis=1) # 1750 rows x 32 columns
# Remove weekday columns
weekday_columns = ['deadline_weekday', 'created_at_weekday', 'launched_at_weekday']
kickstarter_test_df = kickstarter_test_df.drop(columns=weekday_columns, axis=1) # 1750 rows x 29 columns

# Columns that are not available at the time of launching a project
columns_not_available = ['disable_communication', 'state_changed_at', 'staff_pick', 
                         'spotlight', 'state_changed_at_weekday', 'backers_count', 
                         'usd_pledged', 'state_changed_at_month', 'state_changed_at_day', 
                         'state_changed_at_yr', 'state_changed_at_hr', 'launch_to_state_change_days']
# Remove columns that are not available at the time of launching a project
kickstarter_test_df = kickstarter_test_df.drop(columns=columns_not_available, axis=1) # 1750 rows x 17 columns

# X predictors
X_grading = kickstarter_test_df.loc[:,kickstarter_test_df.columns!='state'] # 1750 rows x 16 columns
# Change 'successful' to 1 and 'failed' to 0
kickstarter_test_df['state'] = kickstarter_test_df['state'].replace(['successful', 'failed'], [1, 0])
# Target variable
y_grading = kickstarter_test_df['state'] # 1750 rows x 1 column

# Dummify categorical variables
dummify_cols = ['category', 'country']
X_grading = pd.get_dummies(X_grading, columns=dummify_cols) # 1750 rows x 39 columns

# Remove highly correlated features
correlated_features = ['created_at_yr', 'launched_at_yr', 'country_Non-US']
X_grading = X_grading.drop(columns=correlated_features, axis=1) # 1750 rows x 36 columns

# Create standardized training and test sets
standardizer = StandardScaler()
X_grading_std = standardizer.fit_transform(X_grading)

# ==============================================================================
# Prediction: Gradient Boosting Algorithm
# ==============================================================================
# Predict on the transformed test data
y_grading_test_pred_gbt = model_gbt.predict(X_grading)
# Create a new DataFrame with actual and predicted values
result_df_gbt = pd.DataFrame({
    'Actual State': kickstarter_test_df['state'],
    'Predicted State': np.where(y_grading_test_pred_gbt == 1, 'successful', 'failed')
})
# Compare actual and predicted values
print(result_df_gbt)
# Check when both actual and predicted values match the specified conditions
matching_conditions_gbt = (result_df_gbt['Actual State'] == 0) & (result_df_gbt['Predicted State'] == 'failed') | \
                            (result_df_gbt['Actual State'] == 1) & (result_df_gbt['Predicted State'] == 'successful')
# Display the DataFrame with matching conditions
matching_df_gbt = result_df_gbt[matching_conditions_gbt]
print(matching_df_gbt)
# Display the count of matching and non-matching instances
matching_count_gbt = matching_conditions_gbt.sum()
non_matching_count_gbt = len(result_df_gbt) - matching_count_gbt
print(f"Matching Instances: {matching_count_gbt}") # 1301
print(f"Non-Matching Instances: {non_matching_count_gbt}") # 449
# Check the accuracy of the model
accuracy_gbt_grading = accuracy_score(y_grading, y_grading_test_pred_gbt)
print(f"Accuracy of Gradient Boosting Model is: {accuracy_gbt_grading*100:.2f}%") # 74.34%

# ==============================================================================