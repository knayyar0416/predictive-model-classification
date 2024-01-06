# Classifying the kickstarter projects as success or failure
In this project, I built and tested 6 supervised machine learning algorithms including logistic regression, k-nearest neighbors, classification tree, random forest, gradient boosting and artificial neural network (ANN), to predict the success of kickstarter projects.

🌐 About Kickstarter:
Kickstarter is a platform where creators share their project visions with the communities that will come together to fund them. 

💼 Business Value:
For the Kickstarter's managament, predicting success means planning ahead. My model helps in predicting the success of projects, guiding staff picks, to select the projects worthy of the spotlight, which can increase the visibility and popularity of the platform.

🔄 Process Overview:
I followed these steps to build and test the models:
1. Data Preprocessing:
   I began with data exploration, identified the dominance of US (71%) in column 𝑐𝑜𝑢𝑛𝑡𝑟𝑦, so I replaced other countries with label 'Non-US'. Then, I dropped 𝑛𝑎𝑚𝑒_𝑙𝑒𝑛 and 𝑏𝑙𝑢𝑟𝑏_𝑙𝑒𝑛, keeping
   the cleaned versions, handled a strong correlation between pledged and 𝑢𝑠𝑑_𝑝𝑙𝑒𝑑𝑔𝑒𝑑, created a new column 𝑔𝑜𝑎𝑙_𝑢𝑠𝑑 by multiplying 𝑔𝑜𝑎𝑙 and 𝑠𝑡𝑎𝑡𝑖𝑐_𝑢𝑠𝑑_𝑟𝑎𝑡𝑒, addressed missing values in 𝑐𝑎𝑡𝑒𝑔𝑜𝑟𝑦, and excluded observations with 𝑠𝑡𝑎𝑡𝑒 other than 'successful' or 'failure'.
2. Model Preparation and Feature Engineering:
   - I excluded irrelevant features such as 𝑖𝑑 and 𝑛𝑎𝑚𝑒, hourly details, original date columns, and weekday columns.
   - The goal of this project is to classify a new project as successful or not, based on the information available at the moment when the project owner submits the project. So, the model should only use the predictors that are available at that time. Hence, I removed 12 columns not available at project submission, including 𝑝𝑙𝑒𝑑𝑔𝑒𝑑, 𝑢𝑠𝑑_𝑝𝑙𝑒𝑑𝑔𝑒𝑑, 𝑑𝑖𝑠𝑎𝑏𝑙𝑒_𝑐𝑜𝑚𝑚𝑢𝑛𝑖𝑐𝑎𝑡𝑖𝑜𝑛, 𝑠𝑡𝑎𝑡𝑒_𝑐h𝑎𝑛𝑔𝑒𝑑_𝑎𝑡, 𝑠𝑡𝑎𝑓𝑓_𝑝𝑖𝑐𝑘 and 𝑠𝑝𝑜𝑡𝑙𝑖𝑔h𝑡.
   - After separating the target 'state', I created dummies from 17 features, resulting in 39 predictors, and eliminated 3 having a correlation of 0.80 or higher.
4. Model Building:
   After splitting the dataset, I trained six classification models, and chose accuracy as the primary performance metric to predict true success and failure. The Gradient Boosting (GBT) Algorithm emerged as the top performer with the highest accuracy at 75.30%. 
💡 What is GBT? It generates a large number of trees, and through its sequential tree growth (every time learning from the tree one before it), it places greater emphasis on observations with large errors, making it well-suited for this context.

🎉 Conclusion:
I applied the GBT model to predict the state of projects in kickstarter_grading_df, and achieved an accuracy of 74.34%, confirming its effectiveness as the best model.
