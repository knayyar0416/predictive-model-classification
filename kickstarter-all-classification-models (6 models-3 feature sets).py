# Import libraries
import pandas as pd
import numpy as np

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(color_codes=True)
custom_colors = ["#ebdc78", "#63bff0", "#1984c5", "#54bebe", "#df979e", "#d7658b", "#ffd3b6", "#ee4035"]

# Import dataset
kickstarter_df = pd.read_excel('kickstarter.xlsx')
# Display the shape of the dataset
kickstarter_df.shape # (15474, 45)
# Display the names of all columns
kickstarter_df.columns
# Check for data types
kickstarter_df.info()
# Display the first 5 rows of the dataset
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(kickstarter_df.head(5))
# Descriptive statistics for numerical features    
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print (kickstarter_df.describe())

# ==============================================================================    
# Data preprocessing and exploration
# ==============================================================================
# Plot the distribution of country
country_counts = kickstarter_df['country'].value_counts()
# Sort the country_counts in descending order
country_counts = country_counts.sort_values(ascending=True)
# Plot the distribution of country with custom colors
plt.figure(figsize=(10, 8))
ax = country_counts.plot(kind='barh', color=custom_colors[1])
# Add count labels next to each bar
for index, value in enumerate(country_counts):
    ax.text(value, index, str(value), ha='left', va='center', fontsize=10, color='black')
plt.title('Distribution of Countries in Kickstarter Projects')
plt.xlabel('Count')
plt.ylabel('Country')
# Save the figure
plt.savefig('country_distribution.png', bbox_inches='tight', dpi=300)
plt.show()

# 11000 US, 1924 GB, 830 CA, 532 AU, 240 NL, 181 DE, 161 FR, 102 IT, 89 DK, 86 NZ and so on.

# ==============================================================================
# Plot a chart displaying country and currency
country_currency_counts = kickstarter_df[['country', 'currency']].value_counts().sort_values()
# Plot the figure
plt.figure(figsize=(10, 8))
ax = country_currency_counts.plot(kind='barh', color=custom_colors[1])
# Add count labels next to each bar
for index, value in enumerate(country_currency_counts):
    ax.text(value, index, f'{value} ({country_currency_counts.index[index][0]} - {country_currency_counts.index[index][1]})', ha='left', va='center', fontsize=10)
# Add title and axis names
plt.title('Distribution of Countries and Currencies in Kickstarter Projects')
plt.xlabel('Count')
plt.ylabel('Country and Currency')
# Save the figure
plt.savefig('country-currency_distribution.png', bbox_inches='tight', dpi=300)
plt.show()

# We can remove the 'currency' column, since it is the same as 'country' column
kickstarter_df = kickstarter_df.drop(columns=['currency'], axis=1) # 15474 rows x 44 columns
# Replace non-US countries in country column with 'non-US'
kickstarter_df.loc[kickstarter_df['country'] != 'US', 'country'] = 'Non-US' # 15474 rows x 44 columns
# Display unique values for the country column
kickstarter_df['country'].unique()

# ==============================================================================
# Relationship between name_len and name_len_clean, and blurb_len and blurb_len_clean
plt.figure(figsize=(12, 6))
# Scatter plot for name_len vs name_len_clean
plt.subplot(1, 2, 1)
plt.scatter(kickstarter_df['name_len'], kickstarter_df['name_len_clean'], alpha=0.5, c=custom_colors[1])
plt.title('Relationship between name_len and name_len_clean')
plt.xlabel('name_len')
plt.ylabel('name_len_clean')
# Scatter plot for blurb_len vs blurb_len_clean
plt.subplot(1, 2, 2)
plt.scatter(kickstarter_df['blurb_len'], kickstarter_df['blurb_len_clean'], alpha=0.5, c=custom_colors[1])
plt.title('Relationship between blurb_len and blurb_len_clean')
plt.xlabel('blurb_len')
plt.ylabel('blurb_len_clean')
# Print the layout
plt.tight_layout()
# Save the figure
plt.savefig('relationship-name.png', bbox_inches='tight', dpi=300)
plt.show()

# Drop name_len and blurb_len, since we already have clean versions!
kickstarter_df = kickstarter_df.drop(columns=['name_len', 'blurb_len'], axis=1) # 15474 rows x 42 columns
# Replace null values in name_len_clean with 0
kickstarter_df['name_len_clean'] = kickstarter_df['name_len_clean'].fillna(0)
# Replace null values in blurb_len_clean with 0
kickstarter_df['blurb_len_clean'] = kickstarter_df['blurb_len_clean'].fillna(0)

# ==============================================================================
# Unique values for pledged along with their counts
kickstarter_df['disable_communication'].value_counts()
# Note: Almost all projects are not disabled for communication, i.e. only 167 / 15474 = 1.08% projects are disabled.

# Check if projects with disable_communication=True have a state of 'suspended' or 'canceled'
kickstarter_df[kickstarter_df['disable_communication'] == True]['state'].value_counts() 
# All 167 have state='suspended'

# ==============================================================================
# Unique values for staff_pick along with their counts
kickstarter_df['staff_pick'].value_counts()
# Note: Only 1782 / 15474 = 11.5% of projects are staff picked.

# ==============================================================================
# Correlation between pledged and usd_pledged
kickstarter_df['pledged'].corr(kickstarter_df['usd_pledged']) # 0.989, very high
# Drop pledged, since we already have usd_pledged
kickstarter_df = kickstarter_df.drop(columns=['pledged'], axis=1) # 15474 rows x 41 columns

# ==============================================================================
# Create new column 'goal_usd' by multiplying 'goal' and 'static_usd_rate'
kickstarter_df['goal_usd'] = kickstarter_df['goal'] * kickstarter_df['static_usd_rate'] # 15474 rows x 42 columns
# Remove columns 'goal' and 'static_usd_rate'
kickstarter_df = kickstarter_df.drop(columns=['goal', 'static_usd_rate'], axis=1) # 15474 rows x 40 columns

# ==============================================================================
# Remove irrelevant columns
irrelevant_columns = ['id', 'name', 'deadline_hr', 'created_at_hr', 'launched_at_hr']
kickstarter_df = kickstarter_df.drop(columns=irrelevant_columns, axis=1) # 15474 rows x 35 columns

# Check for duplicates
kickstarter_df[kickstarter_df.duplicated()].shape # (0, 35)

# Check for missing values
kickstarter_df.isnull().sum() # 1392 missing values in 'category' column

# If category is missing, then replace with "No category"
kickstarter_df['category'] = kickstarter_df['category'].fillna('No category') # 15474 rows x 35 columns

# ==============================================================================
# Outlier detection
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
# Plotting the countplot
ax = sns.countplot(x='state', data=kickstarter_df, color=custom_colors[1])
plt.title('Countplot for State')
# Annotating each bar with its count (without decimal places)
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')
# Save the figure
plt.savefig('state.png', bbox_inches='tight', dpi=300)
plt.show()
# For failed: 8860, For successful: 4575, For canceled: 1872, For suspended: 167

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

# Check for new X dataset
X.columns
X.info()
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(X.head(5))

# Correlation
c = X.corr()
# Display the correlation matrix
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(c)
# Plotting the correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(c, cmap="rocket_r", annot=True, fmt=".0f")  # fmt=".0f" for no decimal places
plt.title('Correlation Matrix')
# Save the figure
plt.savefig('corr.png', bbox_inches='tight', dpi=300)
plt.show()

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
# Model 1: Logistic regression
# ==============================================================================
# Run the model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
model_lr = lr.fit(X_train_std,y_train)
# View results
model_lr.intercept_
model_lr.coef_
# Using the model to predict the results based on the test dataset
y_test_pred_lr = model_lr.predict(X_test_std)
# Using the model to predict the probability of being classified to each category
y_test_pred_prob = model_lr.predict_proba(X_test_std)[:,1]
# Performance measures
accuracy_lr = accuracy_score(y_test, y_test_pred_lr)
precision_lr = precision_score(y_test, y_test_pred_lr)
recall_lr = recall_score(y_test, y_test_pred_lr)
f1_lr = f1_score(y_test, y_test_pred_lr)
auc_lr = roc_auc_score(y_test, y_test_pred_lr)
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, y_test_pred_prob)
conf_matrix_lr = confusion_matrix(y_test, y_test_pred_lr)
# Print the results
print(f"Accuracy of Logistic Regression Model is: {accuracy_lr*100:.2f}%")  # 72.06%
print(f"Precision of Logistic Regression Model is: {precision_lr*100:.2f}%")  # 64.05%
print(f"Recall of Logistic Regression Model is: {recall_lr*100:.2f}%")  # 44.08%
print(f"F1 Score of Logistic Regression Model is: {f1_lr*100:.2f}%")  # 52.22%
print(f"AUC of Logistic Regression Model is: {auc_lr*100:.2f}%")  # 65.48%
print("Confusion Matrix of Logistic Regression Model is:")
print(conf_matrix_lr)

# ==============================================================================
# Model 2: K-Nearest Neighbors
# ==============================================================================
from sklearn.neighbors import KNeighborsClassifier
# The general rule of thumb to pick a starting value of k is the square root of the number of observations in the dataset
import math
k = int(math.sqrt(len(X_train_std))) # 94
# Build a model with k = 3 and using euclidean distance function
knn = KNeighborsClassifier(n_neighbors=k,p=2) # For p, 1: Manhattan, 2: Euclidean
model_knn = knn.fit(X_train_std,y_train)
# Using the model to predict the results based on the test dataset
y_test_pred_knn = model_knn.predict(X_test_std)
# Performance measures
accuracy_knn = accuracy_score(y_test, y_test_pred_knn)
precision_knn = precision_score(y_test, y_test_pred_knn)
recall_knn = recall_score(y_test, y_test_pred_knn)
f1_knn = f1_score(y_test, y_test_pred_knn)
auc_knn = roc_auc_score(y_test, y_test_pred_knn)
conf_matrix_knn = confusion_matrix(y_test, y_test_pred_knn)
# Print the results
print(f"Accuracy of K-Nearest Neighbors Model is: {accuracy_knn*100:.2f}%") # 70.64%
print(f"Precision of K-Nearest Neighbors Model is: {precision_knn*100:.2f}%") # 63.27%
print(f"Recall of K-Nearest Neighbors Model is: {recall_knn*100:.2f}%") # 36.33%
print(f"F1 Score of K-Nearest Neighbors Model is: {f1_knn*100:.2f}%") # 46.15%
print(f"AUC of K-Nearest Neighbors Model is: {auc_knn*100:.2f}%") # 62.57%
print("Confusion Matrix of K-Nearest Neighbors Model is:")
print(conf_matrix_knn)
# Choosing k
for i in range (30,50):
    knn = KNeighborsClassifier(n_neighbors=i)
    model = knn.fit(X_train_std,y_train)
    y_test_pred = model.predict(X_test_std)
    print("Accuracy score using k-NN with ",i," neighbors = "+str(accuracy_score(y_test, y_test_pred)))
   
# ==============================================================================
# Model 3: Classification tree
# ==============================================================================
from sklearn.tree import DecisionTreeClassifier
# Build a tree model with 3 layers
ct = DecisionTreeClassifier(max_depth=3) # 3 layers
model_ct = ct.fit(X_train, y_train)  
# Make prediction and evaluate accuracy
y_test_pred_ct = model_ct.predict(X_test)  
# Performance measures
accuracy_ct = accuracy_score(y_test, y_test_pred_ct)
precision_ct = precision_score(y_test, y_test_pred_ct)
recall_ct = recall_score(y_test, y_test_pred_ct)
f1_ct = f1_score(y_test, y_test_pred_ct)
auc_ct = roc_auc_score(y_test, y_test_pred_ct)
conf_matrix_ct = confusion_matrix(y_test, y_test_pred_ct)
# Print the results
print(f"Accuracy of Classification Tree is: {accuracy_ct*100:.2f}%") # 69.51%
print(f"Precision of Classification Tree is: {precision_ct*100:.2f}%") # 55.59%
print(f"Recall of Classification Tree is: {recall_ct*100:.2f}%") # 59.57%
print(f"F1 Score of Classification Tree is: {f1_ct*100:.2f}%") # 57.51%
print(f"AUC of Classification Tree is: {auc_ct*100:.2f}%") # 67.17%
print("Confusion Matrix of Classification Tree is:")
print(conf_matrix_ct)
# Print the tree
from sklearn import tree
plt.figure(figsize=(20,20))
features = X.columns
classes = ['1','0']
tree.plot_tree(model_ct,feature_names=features,class_names=classes,filled=True)
plt.savefig('tree.png', bbox_inches='tight', dpi=300)
plt.show()

# Pruning pre-model building using cross validation for trees with different depths
from sklearn.model_selection import cross_val_score
for i in range (2,21):                                                 
    model = DecisionTreeClassifier(max_depth=i)
    scores = cross_val_score(estimator=model, X=X, y=y, cv=5)
    print(i,':',np.average(scores))
# Highest accuracy is for max_depth=5 @ 70.12%

# ==============================================================================
# Model 4: Random Forest
# ==============================================================================
from sklearn.ensemble import RandomForestClassifier
# Build the model
randomforest = RandomForestClassifier(random_state=0)
model_rf = randomforest.fit(X_train, y_train)
# Print feature importance
pd.Series(model_rf.feature_importances_,index = X.columns).sort_values(ascending = False).plot(kind = 'bar', figsize = (14,6))
# Make prediction and evaluate accuracy
y_test_pred_rf = model_rf.predict(X_test)
# Performance measures
accuracy_rf = accuracy_score(y_test, y_test_pred_rf)
precision_rf = precision_score(y_test, y_test_pred_rf)
recall_rf = recall_score(y_test, y_test_pred_rf)
f1_rf = f1_score(y_test, y_test_pred_rf)
auc_rf = roc_auc_score(y_test, y_test_pred_rf)
conf_matrix_rf = confusion_matrix(y_test, y_test_pred_rf)
# Print the results
print(f"Accuracy of Random Forest Model is: {accuracy_rf*100:.2f}%") # 74.24%
print(f"Precision of Random Forest Model is: {precision_rf*100:.2f}%") # 67.40%
print(f"Recall of Random Forest Model is: {recall_rf*100:.2f}%") # 49.67%
print(f"F1 Score of Random Forest Model is: {f1_rf*100:.2f}%") # 57.20%
print(f"AUC of Random Forest Model is: {auc_rf*100:.2f}%") # 68.47%
print("Confusion Matrix of Random Forest Model is:")
print(conf_matrix_rf)
# K-fold cross validation for different numbers of features to consider at each split
for i in range (2,7):                                                                   
    model = RandomForestClassifier(random_state=0,max_features=i,n_estimators=100)
    scores = cross_val_score(estimator=model, X=X, y=y, cv=5)
    print(i,':',np.average(scores))
# Cross-validate internally using OOB observations
randomforest = RandomForestClassifier(random_state=0,oob_score=True)   
model = randomforest.fit(X, y)
model.oob_score_ # 73.22%

# ==============================================================================
# Model 5: Gradient Boosting Algorithm
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
for i in range (2,10):                                                                        
    model2 = GradientBoostingClassifier(random_state=0,min_samples_split=i,n_estimators=100)
    scores = cross_val_score(estimator=model2, X=X, y=y, cv=5)
    print(i,':',np.average(scores))
    
# ==============================================================================
# Model 6: Artificial Neural Network
# ==============================================================================
from sklearn.neural_network import MLPClassifier
# Build a model
mlp = MLPClassifier(hidden_layer_sizes=(11),max_iter=1000, random_state=0)
model_ann = mlp.fit(X_train_std,y_train)
# Make prediction and evaluate the performance
y_test_pred_ann = model_ann.predict(X_test_std)
# Performance measures
accuracy_ann = accuracy_score(y_test, y_test_pred_ann)
precision_ann = precision_score(y_test, y_test_pred_ann)
recall_ann = recall_score(y_test, y_test_pred_ann)
f1_ann = f1_score(y_test, y_test_pred_ann)
auc_ann = roc_auc_score(y_test, y_test_pred_ann)
conf_matrix_ann = confusion_matrix(y_test, y_test_pred_ann)
# Print the results
print(f"Accuracy of Artificial Neural Network Model is: {accuracy_ann*100:.2f}%") # 71.06%
print(f"Precision of Artificial Neural Network Model is: {precision_ann*100:.2f}%") # 61.59%
print(f"Recall of Artificial Neural Network Model is: {recall_ann*100:.2f}%") # 43.75%
print(f"F1 Score of Artificial Neural Network Model is: {f1_ann*100:.2f}%") # 51.16%
print(f"AUC of Artificial Neural Network Model is: {auc_ann*100:.2f}%") # 64.65%
print("Confusion Matrix of Artificial Neural Network Model is:")
print(conf_matrix_ann)
# Varying the number of hidden layers
mlp2 = MLPClassifier(hidden_layer_sizes=(11,11),max_iter=1000, random_state=0)
model2 = mlp2.fit(X_train_std,y_train)
y_test_pred_2 = model2.predict(X_test_std)
accuracy_score(y_test, y_test_pred_2) # 71.54%
# Cross-validate with different size of the hidden layer
for i in range (2,21):    
    model3 = MLPClassifier(hidden_layer_sizes=(i),max_iter=1000, random_state=0)
    scores = cross_val_score(estimator=model3, X=X, y=y, cv=5)
    print(i,':',np.average(scores))

# ==============================================================================
# ==============================================================================
# Feature selection using LASSO
# ==============================================================================
# ==============================================================================
from sklearn.linear_model import Lasso
# Standardize X
standardizer = StandardScaler()
X_std = standardizer.fit_transform(X)
# Run LASSO with alpha=0.01
ls = Lasso(alpha=0.01) # you can control the number of predictors through alpha
model = ls.fit(X_std,y)
# Create a DataFrame to store predictor names and their corresponding coefficients
coefficients_df = pd.DataFrame(list(zip(X.columns,model.coef_)), columns = ['predictor','coefficient'])
# Filter predictors with non-zero coefficients
selected_features = coefficients_df[coefficients_df['coefficient'] != 0] # 26
# Print the selected features
num_selected_features = len(selected_features)
print(f"\nNumber of selected variables: {num_selected_features}")
print(f"\n{selected_features}")

# ==============================================================================
# Rerun models with the selected features
# ==============================================================================
# Extract the selected predictors for training and testing
selected_features_names = selected_features['predictor'].tolist()
# Training and test sets
X_lasso_train = X_train[selected_features_names]
X_lasso_test = X_test[selected_features_names]
# Standardize the training and test sets
standardizer = StandardScaler()
X_lasso_train_std = standardizer.fit_transform(X_lasso_train)
X_lasso_test_std = standardizer.transform(X_lasso_test)

# ==============================================================================
# Model 1: Logistic regression (Lasso selected features)
# ==============================================================================
# Run the model
lr_lasso = LogisticRegression()
model_lr_lasso = lr_lasso.fit(X_lasso_train_std,y_train)
# View results
model_lr_lasso.intercept_
model_lr_lasso.coef_
# Using the model to predict the results based on the test dataset
y_test_pred_lr_lasso = model_lr_lasso.predict(X_lasso_test_std)
# Using the model to predict the probability of being classified to each category
y_test_pred_prob_lasso = model_lr_lasso.predict_proba(X_lasso_test_std)[:,1]
# Performance measures
accuracy_lr_lasso = accuracy_score(y_test, y_test_pred_lr_lasso)
precision_lr_lasso = precision_score(y_test, y_test_pred_lr_lasso)
recall_lr_lasso = recall_score(y_test, y_test_pred_lr_lasso)
f1_lr_lasso = f1_score(y_test, y_test_pred_lr_lasso)
auc_lr_lasso = roc_auc_score(y_test, y_test_pred_lr_lasso)
conf_matrix_lr_lasso = confusion_matrix(y_test, y_test_pred_lr_lasso)
# Print the results
print(f"Accuracy of Logistic Regression Model using LASSO selected predictors is: {accuracy_lr_lasso*100:.2f}%") # 72.26%
print(f"Precision of Logistic Regression Model using LASSO selected predictors is: {precision_lr_lasso*100:.2f}%") # 64.57%
print(f"Recall of Logistic Regression Model using LASSO selected predictors is: {recall_lr_lasso*100:.2f}%") # 44.14%
print(f"F1 Score of Logistic Regression Model using LASSO selected predictors is: {f1_lr_lasso*100:.2f}%") # 52.44%
print(f"AUC of Logistic Regression Model using LASSO selected predictors is: {auc_lr_lasso*100:.2f}%") # 65.65%
print("Confusion Matrix of Logistic Regression Model using LASSO selected predictors is:")
print(conf_matrix_lr_lasso)

# ==============================================================================
# Model 2: K-Nearest Neighbors (Lasso selected features)
# ==============================================================================
# Initialize the k-NN classifier
knn_lasso = KNeighborsClassifier(n_neighbors=k, p=2)
# Fit the k-NN classifier on the transformed training data
model_knn_lasso = knn_lasso.fit(X_lasso_train_std, y_train)
# Predict on the transformed test data
y_test_pred_knn = model_knn_lasso.predict(X_lasso_test_std)
# Get performance measure scores for KNN
accuracy_knn_lasso = accuracy_score(y_test, y_test_pred_knn)
precision_knn_lasso = precision_score(y_test, y_test_pred_knn)
recall_knn_lasso = recall_score(y_test, y_test_pred_knn)
f1_knn_lasso = f1_score(y_test, y_test_pred_knn)
auc_knn_lasso = roc_auc_score(y_test, y_test_pred_knn)
conf_matrix_knn_lasso = confusion_matrix(y_test, y_test_pred_knn)
# Print the results
print(f"Accuracy of K-Nearest Neighbors Model using LASSO selected predictors is: {accuracy_knn_lasso*100:.2f}%") # 71.24%
print(f"Precision of K-Nearest Neighbors Model using LASSO selected predictors is: {precision_knn_lasso*100:.2f}%") # 64.45%
print(f"Recall of K-Nearest Neighbors Model using LASSO selected predictors is: {recall_knn_lasso*100:.2f}%") # 37.89%
print(f"F1 Score of K-Nearest Neighbors Model using LASSO selected predictors is: {f1_knn_lasso*100:.2f}%") # 47.72%
print(f"AUC of K-Nearest Neighbors Model using LASSO selected predictors is: {auc_knn_lasso*100:.2f}%") # 63.41%
print("Confusion Matrix of K-Nearest Neighbors Model using LASSO selected predictors is:")
print(conf_matrix_knn_lasso)

# ==============================================================================
# Model 3: Classification tree (Lasso selected features)
# ==============================================================================
# Build a tree model with 3 layers
ct_lasso = DecisionTreeClassifier(max_depth=3) # 3 layers
# Fit the tree model on the transformed training data
model_ct_lasso = ct_lasso.fit(X_lasso_train, y_train)
# Predict on the transformed test data
y_test_pred_ct_lasso = model_ct_lasso.predict(X_lasso_test)
# Get performance measure scores for Classification Tree
accuracy_ct_lasso = accuracy_score(y_test, y_test_pred_ct_lasso)
precision_ct_lasso = precision_score(y_test, y_test_pred_ct_lasso)
recall_ct_lasso = recall_score(y_test, y_test_pred_ct_lasso)
f1_ct_lasso = f1_score(y_test, y_test_pred_ct_lasso)
auc_ct_lasso = roc_auc_score(y_test, y_test_pred_ct_lasso)
conf_matrix_ct_lasso = confusion_matrix(y_test, y_test_pred_ct_lasso)
# Print the results
print(f"Accuracy of Classification Tree Model using LASSO selected predictors is: {accuracy_ct_lasso*100:.2f}%") # 69.51%
print(f"Precision of Classification Tree Model using LASSO selected predictors is: {precision_ct_lasso*100:.2f}%") # 55.59%
print(f"Recall of Classification Tree Model using LASSO selected predictors is: {recall_ct_lasso*100:.2f}%") # 59.57%
print(f"F1 Score of Classification Tree Model using LASSO selected predictors is: {f1_ct_lasso*100:.2f}%") # 57.51%
print(f"AUC of Classification Tree Model using LASSO selected predictors is: {auc_ct_lasso*100:.2f}%") # 67.17%
print("Confusion Matrix of Classification Tree Model using LASSO selected predictors is:")
print(conf_matrix_ct_lasso)

# ==============================================================================
# Model 4: Random Forest (Lasso selected features)
# ==============================================================================
# Build the model
rf_lasso = RandomForestClassifier(random_state=0)
# Fit the model on the transformed training data
model_rf_lasso = rf_lasso.fit(X_lasso_train, y_train)
# Predict on the transformed test data
y_test_pred_rf_lasso = model_rf_lasso.predict(X_lasso_test)
# Get performance measure scores for Random Forest
accuracy_rf_lasso = accuracy_score(y_test, y_test_pred_rf_lasso)
precision_rf_lasso = precision_score(y_test, y_test_pred_rf_lasso)
recall_rf_lasso = recall_score(y_test, y_test_pred_rf_lasso)
f1_rf_lasso = f1_score(y_test, y_test_pred_rf_lasso)
auc_rf_lasso = roc_auc_score(y_test, y_test_pred_rf_lasso)
conf_matrix_rf_lasso = confusion_matrix(y_test, y_test_pred_rf_lasso)
# Print the results
print(f"Accuracy of Random Forest Model using LASSO selected predictors is: {accuracy_rf_lasso*100:.2f}%") # 72.73%
print(f"Precision of Random Forest Model using LASSO selected predictors is: {precision_rf_lasso*100:.2f}%") # 62.84%
print(f"Recall of Random Forest Model using LASSO selected predictors is: {recall_rf_lasso*100:.2f}%") # 52.08%
print(f"F1 Score of Random Forest Model using LASSO selected predictors is: {f1_rf_lasso*100:.2f}%") # 56.96%
print(f"AUC of Random Forest Model using LASSO selected predictors is: {auc_rf_lasso*100:.2f}%") # 67.88%
print("Confusion Matrix of Random Forest Model using LASSO selected predictors is:")
print(conf_matrix_rf_lasso)
# Cross-validate internally using OOB observations
randomforest = RandomForestClassifier(random_state=0,oob_score=True)   
model = randomforest.fit(X_std, y)
model.oob_score_ # 73.21%

# ==============================================================================
# Model 5: Gradient Boosting Algorithm (Lasso selected features)
# ==============================================================================
# Build the model
gbt_lasso = GradientBoostingClassifier(random_state=0)
# Fit the model on the transformed training data
model_gbt_lasso = gbt_lasso.fit(X_lasso_train, y_train)
# Predict on the transformed test data
y_test_pred_gbt_lasso = model_gbt_lasso.predict(X_lasso_test)
# Get performance measure scores for Gradient Boosting
accuracy_gbt_lasso = accuracy_score(y_test, y_test_pred_gbt_lasso)
precision_gbt_lasso = precision_score(y_test, y_test_pred_gbt_lasso)
recall_gbt_lasso = recall_score(y_test, y_test_pred_gbt_lasso)
f1_gbt_lasso = f1_score(y_test, y_test_pred_gbt_lasso)
auc_gbt_lasso = roc_auc_score(y_test, y_test_pred_gbt_lasso)
conf_matrix_gbt_lasso = confusion_matrix(y_test, y_test_pred_gbt_lasso)
# Print the results
print(f"Accuracy of Gradient Boosting Model using LASSO selected predictors is: {accuracy_gbt_lasso*100:.2f}%") # 74.49%
print(f"Precision of Gradient Boosting Model using LASSO selected predictors is: {precision_gbt_lasso*100:.2f}%") # 67.23%
print(f"Recall of Gradient Boosting Model using LASSO selected predictors is: {recall_gbt_lasso*100:.2f}%") # 51.43%
print(f"F1 Score of Gradient Boosting Model using LASSO selected predictors is: {f1_gbt_lasso*100:.2f}%") # 58.28%
print(f"AUC of Gradient Boosting Model using LASSO selected predictors is: {auc_gbt_lasso*100:.2f}%") # 69.07%
print("Confusion Matrix of Gradient Boosting Model using LASSO selected predictors is:")
print(conf_matrix_gbt_lasso)

# ==============================================================================
# Model 6: Artificial Neural Network (Lasso selected features)
# ==============================================================================
# Build a model
ann_lasso = MLPClassifier(hidden_layer_sizes=(11),max_iter=1000, random_state=0)
# Fit the model on the transformed training data
model_ann_lasso = ann_lasso.fit(X_lasso_train_std, y_train)
# Predict on the transformed test data
y_test_pred_ann_lasso = model_ann_lasso.predict(X_lasso_test_std)
# Get performance measure scores for Artificial Neural Network
accuracy_ann_lasso = accuracy_score(y_test, y_test_pred_ann_lasso)
precision_ann_lasso = precision_score(y_test, y_test_pred_ann_lasso)
recall_ann_lasso = recall_score(y_test, y_test_pred_ann_lasso)
f1_ann_lasso = f1_score(y_test, y_test_pred_ann_lasso)
auc_ann_lasso = roc_auc_score(y_test, y_test_pred_ann_lasso)
conf_matrix_ann_lasso = confusion_matrix(y_test, y_test_pred_ann_lasso)
# Print the results
print(f"Accuracy of Artificial Neural Network Model using LASSO selected predictors is: {accuracy_ann_lasso*100:.2f}%") # 72.10%
print(f"Precision of Artificial Neural Network Model using LASSO selected predictors is: {precision_ann_lasso*100:.2f}%") # 63.55%
print(f"Recall of Artificial Neural Network Model using LASSO selected predictors is: {recall_ann_lasso*100:.2f}%") # 45.64%
print(f"F1 Score of Artificial Neural Network Model using LASSO selected predictors is: {f1_ann_lasso*100:.2f}%") # 53.13%
print(f"AUC of Artificial Neural Network Model using LASSO selected predictors is: {auc_ann_lasso*100:.2f}%") # 65.88%
print("Confusion Matrix of Artificial Neural Network Model using LASSO selected predictors is:")
print(conf_matrix_ann_lasso)

# ==============================================================================
# ==============================================================================
# Dimensionality reduction using PCA
# ==============================================================================
# ==============================================================================    
from sklearn.decomposition import PCA
pca = PCA(n_components=16)
pca.fit(X_std)
# Explained Variance
pca.explained_variance_ratio_
pca.components_
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
plt.plot(cumulative_variance, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by all Components')
plt.show()
# Selecting the number of components
threshold = 0.95
n_components = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= threshold) + 1
# Plotting the scree plot
PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.grid(True)
plt.savefig('scree.png', bbox_inches='tight', dpi=300)
plt.show()
# Fit the model on new dataset and evaluate
pca = PCA(n_components=5)
X_pca_train = pca.fit_transform(X_train)
X_pca_test = pca.fit_transform(X_test)
# Create standardized training and test sets
standardizer = StandardScaler()
X_pca_train_std = standardizer.fit_transform(X_pca_train)
X_pca_test_std = standardizer.transform(X_pca_test)

# ==============================================================================
# Model 1: Logistic regression (PCA selected components)
# ==============================================================================
# Run the model
lr_pca = LogisticRegression()
model_lr_pca = lr_pca.fit(X_pca_train_std,y_train)
# View results
model_lr_pca.intercept_
model_lr_pca.coef_
# Using the model to predict the results based on the test dataset
y_test_pred_lr_pca = model_lr_pca.predict(X_pca_test_std)
# Using the model to predict the probability of being classified to each category
y_test_pred_prob_pca = model_lr_pca.predict_proba(X_pca_test_std)[:,1]
# Performance measures
accuracy_lr_pca = accuracy_score(y_test, y_test_pred_lr_pca)
precision_lr_pca = precision_score(y_test, y_test_pred_lr_pca)
recall_lr_pca = recall_score(y_test, y_test_pred_lr_pca)
f1_lr_pca = f1_score(y_test, y_test_pred_lr_pca)
auc_lr_pca = roc_auc_score(y_test, y_test_pred_lr_pca)
conf_matrix_lr_pca = confusion_matrix(y_test, y_test_pred_lr_pca)
# Print the results
print(f"Accuracy of Logistic Regression Model using PCA selected predictors is: {accuracy_lr_pca*100:.2f}%") # 65.49%
print(f"Precision of Logistic Regression Model using PCA selected predictors is: {precision_lr_pca*100:.2f}%") # 51.28%
print(f"Recall of Logistic Regression Model using PCA selected predictors is: {recall_lr_pca*100:.2f}%") # 7.81%
print(f"F1 Score of Logistic Regression Model using PCA selected predictors is: {f1_lr_pca*100:.2f}%") # 13.56%
print(f"AUC of Logistic Regression Model using PCA selected predictors is: {auc_lr_pca*100:.2f}%") # 51.94%
print("Confusion Matrix of Logistic Regression Model using PCA selected predictors is:")
print(conf_matrix_lr_pca)

# ==============================================================================
# Model 2: K-Nearest Neighbors (PCA selected components)
# ==============================================================================
# The general rule of thumb to pick a starting value of k is the square root of the number of observations in the dataset
import math
k = int(math.sqrt(len(X_pca_train_std))) # 94
# Initialize the k-NN classifier
knn_pca = KNeighborsClassifier(n_neighbors=k,p=2) # For p, 1: Manhattan, 2: Euclidean
# Fit the k-NN classifier on the transformed training data
model_knn_pca = knn_lasso.fit(X_pca_train_std, y_train)
# Using the model to predict the results based on the test dataset
y_test_pred_pca = model_knn_pca.predict(X_pca_test_std)
# Performance measures
accuracy_knn_pca = accuracy_score(y_test, y_test_pred_pca)
precision_knn_pca = precision_score(y_test, y_test_pred_pca)
recall_knn_pca = recall_score(y_test, y_test_pred_pca)
f1_knn_pca = f1_score(y_test, y_test_pred_pca)
auc_knn_pca = roc_auc_score(y_test, y_test_pred_pca)
conf_matrix_knn_pca = confusion_matrix(y_test, y_test_pred_pca)
# Print the results
print(f"Accuracy of K-Nearest Neighbors Model using PCA selected components is: {accuracy_knn_pca*100:.2f}%") # 64.75%
print(f"Precision of K-Nearest Neighbors Model using PCA selected components is: {precision_knn_pca*100:.2f}%") # 43.35%
print(f"Recall of K-Nearest Neighbors Model using PCA selected components is: {recall_knn_pca*100:.2f}%") # 5.73%
print(f"F1 Score of K-Nearest Neighbors Model using PCA selected components is: {f1_knn_pca*100:.2f}%") # 10.12%
print(f"AUC of K-Nearest Neighbors Model using PCA selected components is: {auc_knn_pca*100:.2f}%") # 50.88%
print("Confusion Matrix of K-Nearest Neighbors Model using PCA selected components is:")
# Choosing k
for i in range (30,50):
    knn = KNeighborsClassifier(n_neighbors=i)
    model = knn.fit(X_pca_train_std,y_train)
    y_test_pred = model.predict(X_pca_test_std)
    print("Accuracy score using k-NN with ",i," neighbors = "+str(accuracy_score(y_test, y_test_pred)))
    
# ==============================================================================
# Model 3: Classification tree (PCA selected components)
# ==============================================================================
# Build a tree model with 3 layers
ct_pca = DecisionTreeClassifier(max_depth=3) # 3 layers
# Fit the tree model on the transformed training data
model_ct_pca = ct_pca.fit(X_pca_train, y_train)
# Predict on the transformed test data
y_test_pred_ct_pca = model_ct_pca.predict(X_pca_test)
# Get performance measure scores for Classification Tree
accuracy_ct_pca = accuracy_score(y_test, y_test_pred_ct_pca)
precision_ct_pca = precision_score(y_test, y_test_pred_ct_pca)
recall_ct_pca = recall_score(y_test, y_test_pred_ct_pca)
f1_ct_pca = f1_score(y_test, y_test_pred_ct_pca)
auc_ct_pca = roc_auc_score(y_test, y_test_pred_ct_pca)
conf_matrix_ct_pca = confusion_matrix(y_test, y_test_pred_ct_pca)
# Print the results
print(f"Accuracy of Classification Tree Model using PCA selected components is: {accuracy_ct_pca*100:.2f}%") # 57.92%
print(f"Precision of Classification Tree Model using PCA selected components is: {precision_ct_pca*100:.2f}%") # 42.97%
print(f"Recall of Classification Tree Model using PCA selected components is: {recall_ct_pca*100:.2f}%") # 65.62%
print(f"F1 Score of Classification Tree Model using PCA selected components is: {f1_ct_pca*100:.2f}%") # 51.93%
print(f"AUC of Classification Tree Model using PCA selected components is: {auc_ct_pca*100:.2f}%") # 59.73%
print("Confusion Matrix of Classification Tree Model using PCA selected components is:")
print(conf_matrix_ct_pca)

# ==============================================================================
# Model 4: Random Forest (PCA selected components)
# ==============================================================================
# Build the model
rf_pca = RandomForestClassifier(random_state=0)
model_rf_pca = rf_pca.fit(X_pca_train, y_train)
# Print feature importance
pd.Series(model_rf_pca.feature_importances_,index = X.columns).sort_values(ascending = False).plot(kind = 'bar', figsize = (14,6))
# Make prediction and evaluate accuracy
y_test_pred_rf_pca = model_rf_pca.predict(X_pca_test)
# Performance measures
accuracy_rf_pca = accuracy_score(y_test, y_test_pred_rf_pca)
precision_rf_pca = precision_score(y_test, y_test_pred_rf_pca)
recall_rf_pca = recall_score(y_test, y_test_pred_rf_pca)
f1_rf_pca = f1_score(y_test, y_test_pred_rf_pca)
auc_rf_pca = roc_auc_score(y_test, y_test_pred_rf_pca)
conf_matrix_rf_pca = confusion_matrix(y_test, y_test_pred_rf_pca)
# Print the results
print(f"Accuracy of Random Forest Model using PCA selected components is: {accuracy_rf_pca*100:.2f}%") # 52.17%
print(f"Precision of Random Forest Model using PCA selected components is: {precision_rf_pca*100:.2f}%") # 40.48%
print(f"Recall of Random Forest Model using PCA selected components is: {recall_rf_pca*100:.2f}%") # 80.92%
print(f"F1 Score of Random Forest Model using PCA selected components is: {f1_rf_pca*100:.2f}%") # 53.96%
print(f"AUC of Random Forest Model using PCA selected components is: {auc_rf_pca*100:.2f}%") # 58.92%
print("Confusion Matrix of Random Forest Model using PCA selected components is:")
print(conf_matrix_rf_pca)
# K-fold cross-validation for different numbers of features to consider at each split
for i in range(2, 7):
    model = RandomForestClassifier(random_state=0, max_features=i, n_estimators=100)
    scores = cross_val_score(estimator=model, X=X_pca_train, y=y_train, cv=5)
    print(f"{i} components:", np.average(scores))
# Cross-validate internally using OOB observations
randomforest = RandomForestClassifier(random_state=0, oob_score=True)
model = randomforest.fit(X_pca_train, y_train)
oob_score_rf_pca = model.oob_score_
print(f"OOB Score of Random Forest Model using PCA selected components is: {oob_score_rf_pca*100:.2f}%") # 66.49%

# ==============================================================================
# Model 5: Gradient Boosting Algorithm (PCA selected components)
# ==============================================================================
# Build the model
gbt_pca = GradientBoostingClassifier(random_state=0)
# Fit the model on the transformed training data
model_gbt_pca = gbt_pca.fit(X_pca_train, y_train)
# Predict on the transformed test data
y_test_pred_gbt_pca = model_gbt_pca.predict(X_pca_test)
# Get performance measure scores for Gradient Boosting
accuracy_gbt_pca = accuracy_score(y_test, y_test_pred_gbt_pca)
precision_gbt_pca = precision_score(y_test, y_test_pred_gbt_pca)
recall_gbt_pca = recall_score(y_test, y_test_pred_gbt_pca)
f1_gbt_pca = f1_score(y_test, y_test_pred_gbt_pca)
auc_gbt_pca = roc_auc_score(y_test, y_test_pred_gbt_pca)
conf_matrix_gbt_pca = confusion_matrix(y_test, y_test_pred_gbt_pca)
# Print the results
print(f"Accuracy of Gradient Boosting Model using PCA selected components is: {accuracy_gbt_pca*100:.2f}%") # 50.38%
print(f"Precision of Gradient Boosting Model using PCA selected components is: {precision_gbt_pca*100:.2f}%") # 39.21%
print(f"Recall of Gradient Boosting Model using PCA selected components is: {recall_gbt_pca*100:.2f}%") # 78.52%
print(f"F1 Score of Gradient Boosting Model using PCA selected components is: {f1_gbt_pca*100:.2f}%") # 52.30%
print(f"AUC of Gradient Boosting Model using PCA selected components is: {auc_gbt_pca*100:.2f}%") # 56.99%
print("Confusion Matrix of Gradient Boosting Model using PCA selected components is:")
print(conf_matrix_gbt_pca)

# ==============================================================================
# Model 6: Artificial Neural Network (PCA selected components)
# ==============================================================================
# Build a model
ann_pca = MLPClassifier(hidden_layer_sizes=(11),max_iter=1000, random_state=0)
# Fit the model on the transformed training data
model_ann_pca = ann_pca.fit(X_pca_train_std, y_train)
# Predict on the transformed test data
y_test_pred_ann_pca = model_ann_pca.predict(X_pca_test_std)
# Get performance measure scores for Artificial Neural Network
accuracy_ann_pca = accuracy_score(y_test, y_test_pred_ann_pca)
precision_ann_pca = precision_score(y_test, y_test_pred_ann_pca)
recall_ann_pca = recall_score(y_test, y_test_pred_ann_pca)
f1_ann_pca = f1_score(y_test, y_test_pred_ann_pca)
auc_ann_pca = roc_auc_score(y_test, y_test_pred_ann_pca)
conf_matrix_ann_pca = confusion_matrix(y_test, y_test_pred_ann_pca)
# Print the results
print(f"Accuracy of Artificial Neural Network Model using PCA selected components is: {accuracy_ann_pca*100:.2f}%") # 65.22%
print(f"Precision of Artificial Neural Network Model using PCA selected components is: {precision_ann_pca*100:.2f}%") # 49.76%
print(f"Recall of Artificial Neural Network Model using PCA selected components is: {recall_ann_pca*100:.2f}%") # 39.78%
print(f"F1 Score of Artificial Neural Network Model using PCA selected components is: {f1_ann_pca*100:.2f}%") # 44.21%
print(f"AUC of Artificial Neural Network Model using PCA selected components is: {auc_ann_pca*100:.2f}%") # 59.24%
print("Confusion Matrix of Artificial Neural Network Model using PCA selected components is:")
print(conf_matrix_ann_pca)

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

# ==============================================================================
# Create new column 'goal_usd' by multiplying 'goal' and 'static_usd_rate'
kickstarter_test_df['goal_usd'] = kickstarter_test_df['goal'] * kickstarter_test_df['static_usd_rate'] # 2000 rows x 42 columns
# Remove columns 'goal' and 'static_usd_rate'
kickstarter_test_df = kickstarter_test_df.drop(columns=['goal', 'static_usd_rate'], axis=1) # 2000 rows x 40 columns

# ==============================================================================
# Remove irrelevant columns
irrelevant_columns = ['id', 'name', 'deadline_hr', 'created_at_hr', 'launched_at_hr']
kickstarter_test_df = kickstarter_test_df.drop(columns=irrelevant_columns, axis=1) # 2000 rows x 35 columns
# Check for duplicates
kickstarter_test_df[kickstarter_test_df.duplicated()].shape # (0, 35)
# Check for missing values
kickstarter_test_df.isnull().sum()
# If category is missing, then replace with "No category"
kickstarter_test_df['category'] = kickstarter_test_df['category'].fillna('No category') # 2000 rows x 35 columns

# ==============================================================================
# Drop 'canceled' and 'suspended' from state column
kickstarter_test_df = kickstarter_test_df[kickstarter_test_df['state'] != 'canceled'] # 1772 rows x 35 columns
kickstarter_test_df = kickstarter_test_df[kickstarter_test_df['state'] != 'suspended'] # 1750 rows x 35 columns

# ==============================================================================
# Remove original date columns
date_columns = ['deadline', 'created_at', 'launched_at']
kickstarter_test_df = kickstarter_test_df.drop(columns=date_columns, axis=1) # 1750 rows x 32 columns
# Remove weekday columns
weekday_columns = ['deadline_weekday', 'created_at_weekday', 'launched_at_weekday']
kickstarter_test_df = kickstarter_test_df.drop(columns=weekday_columns, axis=1) # 1750 rows x 29 columns

# ==============================================================================
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
# ==============================================================================
# Predictions on initials sets of models
# ==============================================================================
# ==============================================================================
# Since we got the highest accuracy for Gradient Boosting and Random Firest using 
# the initial set of features, we will use the original models for predictions.

# ==============================================================================
# Model 1: Logistic regression
# ==============================================================================
# Predict on the transformed test data
y_grading_test_pred_lr = model_lr.predict(X_grading_std)
# Create a new DataFrame with actual and predicted values
result_df = pd.DataFrame({
    'Actual State': kickstarter_test_df['state'],
    'Predicted State': np.where(y_grading_test_pred_lr == 1, 'successful', 'failed')
})
# Compare actual and predicted values
print(result_df)
# Check when both actual and predicted values match the specified conditions
matching_conditions = (result_df['Actual State'] == 0) & (result_df['Predicted State'] == 'failed') | \
                      (result_df['Actual State'] == 1) & (result_df['Predicted State'] == 'successful')
# Display the DataFrame with matching conditions
matching_df = result_df[matching_conditions]
print(matching_df)
# Display the count of matching and non-matching instances
matching_count = matching_conditions.sum()
non_matching_count = len(result_df) - matching_count
print(f"Matching Instances: {matching_count}") # 1267
print(f"Non-Matching Instances: {non_matching_count}") # 483
# Check the accuracy of the model
accuracy_lr_grading = accuracy_score(y_grading, y_grading_test_pred_lr)
print(f"Accuracy of Logistic Regression Model is: {accuracy_lr_grading*100:.2f}%") # 72.40%

# ==============================================================================
# Model 2: K-Nearest Neighbors
# ==============================================================================
# Predict on the transformed test data
y_grading_test_pred_knn = model_knn.predict(X_grading_std)
# Create a new DataFrame with actual and predicted values
result_df_knn = pd.DataFrame({
    'Actual State': kickstarter_test_df['state'],
    'Predicted State': np.where(y_grading_test_pred_knn == 1, 'successful', 'failed')
})
# Compare actual and predicted values
print(result_df_knn)
# Check when both actual and predicted values match the specified conditions
matching_conditions_knn = (result_df_knn['Actual State'] == 0) & (result_df_knn['Predicted State'] == 'failed') | \
                           (result_df_knn['Actual State'] == 1) & (result_df_knn['Predicted State'] == 'successful')
# Display the DataFrame with matching conditions
matching_df_knn = result_df_knn[matching_conditions_knn]
print(matching_df_knn)
# Display the count of matching and non-matching instances
matching_count_knn = matching_conditions_knn.sum()
non_matching_count_knn = len(result_df_knn) - matching_count_knn
print(f"Matching Instances: {matching_count_knn}") # 1248
print(f"Non-Matching Instances: {non_matching_count_knn}") # 502
# Check the accuracy of the model
accuracy_knn_grading = accuracy_score(y_grading, y_grading_test_pred_knn)
print(f"Accuracy of K-Nearest Neighbors Model is: {accuracy_knn_grading*100:.2f}%") # 71.31%

# ==============================================================================
# Model 3: Classification tree
# ==============================================================================
# Predict on the transformed test data
y_grading_test_pred_ct = model_ct.predict(X_grading)
# Create a new DataFrame with actual and predicted values
result_df_ct = pd.DataFrame({
    'Actual State': kickstarter_test_df['state'],
    'Predicted State': np.where(y_grading_test_pred_ct == 1, 'successful', 'failed')
})
# Compare actual and predicted values
print(result_df_ct)
# Check when both actual and predicted values match the specified conditions
matching_conditions_ct = (result_df_ct['Actual State'] == 0) & (result_df_ct['Predicted State'] == 'failed') | \
                            (result_df_ct['Actual State'] == 1) & (result_df_ct['Predicted State'] == 'successful')                            
# Display the DataFrame with matching conditions
matching_df_ct = result_df_ct[matching_conditions_ct]
print(matching_df_ct)
# Display the count of matching and non-matching instances
matching_count_ct = matching_conditions_ct.sum()
non_matching_count_ct = len(result_df_ct) - matching_count_ct
print(f"Matching Instances: {matching_count_ct}") # 1217
print(f"Non-Matching Instances: {non_matching_count_ct}") # 533
# Check the accuracy of the model
accuracy_ct_grading = accuracy_score(y_grading, y_grading_test_pred_ct)
print(f"Accuracy of Classification Tree Model is: {accuracy_ct_grading*100:.2f}%") # 69.54%

# ==============================================================================
# Model 4: Random Forest
# ==============================================================================
# Predict on the transformed test data
y_grading_test_pred_rf = model_rf.predict(X_grading)
# Create a new DataFrame with actual and predicted values
result_df_rf = pd.DataFrame({
    'Actual State': kickstarter_test_df['state'],
    'Predicted State': np.where(y_grading_test_pred_rf == 1, 'successful', 'failed')
})
# Compare actual and predicted values
print(result_df_rf)
# Check when both actual and predicted values match the specified conditions
matching_conditions_rf = (result_df_rf['Actual State'] == 0) & (result_df_rf['Predicted State'] == 'failed') | \
                            (result_df_rf['Actual State'] == 1) & (result_df_rf['Predicted State'] == 'successful')                    
# Display the DataFrame with matching conditions
matching_df_rf = result_df_rf[matching_conditions_rf]
print(matching_df_rf)
# Display the count of matching and non-matching instances
matching_count_rf = matching_conditions_rf.sum()
non_matching_count_rf = len(result_df_rf) - matching_count_rf
print(f"Matching Instances: {matching_count_rf}") # 1275
print(f"Non-Matching Instances: {non_matching_count_rf}") # 475
# Check the accuracy of the model
accuracy_rf_grading = accuracy_score(y_grading, y_grading_test_pred_rf)
print(f"Accuracy of Random Forest Model is: {accuracy_rf_grading*100:.2f}%") # 72.86%

# ==============================================================================
# Model 5: Gradient Boosting Algorithm
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
# Model 6: Artificial Neural Network
# ==============================================================================
# Predict on the transformed test data
y_grading_test_pred_ann = model_ann.predict(X_grading_std)
# Create a new DataFrame with actual and predicted values
result_df_ann = pd.DataFrame({
    'Actual State': kickstarter_test_df['state'],
    'Predicted State': np.where(y_grading_test_pred_ann == 1, 'successful', 'failed')
})
# Compare actual and predicted values
print(result_df_ann)
# Check when both actual and predicted values match the specified conditions
matching_conditions_ann = (result_df_ann['Actual State'] == 0) & (result_df_ann['Predicted State'] == 'failed') | \
                            (result_df_ann['Actual State'] == 1) & (result_df_ann['Predicted State'] == 'successful')                
# Display the DataFrame with matching conditions
matching_df_ann = result_df_ann[matching_conditions_ann]
print(matching_df_ann)
# Display the count of matching and non-matching instances
matching_count_ann = matching_conditions_ann.sum()
non_matching_count_ann = len(result_df_ann) - matching_count_ann
print(f"Matching Instances: {matching_count_ann}") # 1255
print(f"Non-Matching Instances: {non_matching_count_ann}") # 495
# Check the accuracy of the model
accuracy_ann_grading = accuracy_score(y_grading, y_grading_test_pred_ann)
print(f"Accuracy of Artificial Neural Network Model is: {accuracy_ann_grading*100:.2f}%") # 71.71%

# ==============================================================================