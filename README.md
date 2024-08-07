I've been working on a project to predict heart disease using a dataset of heart failure patients. Here's a breakdown of what I did:

Data Exploration and Cleaning:

I started by importing the heart disease data from a CSV file using pandas.
I took a look at the data using describe() to understand the distribution of features like age, cholesterol, and blood pressure.
I visualized the distribution of categorical features like chest pain type and exercise angina using seaborn's countplot function. This helped me identify patterns and potential relationships with heart disease.
To handle missing values, I replaced zeros in the cholesterol column with NaNs (Not a Number) and then filled those NaNs with the median value in the column. For the resting blood pressure data, I replaced values outside a reasonable range (below 40 or above 140) with the median value as well.

Feature Engineering:

I converted the categorical features (text labels) into numerical values using scikit-learn's LabelEncoder. This is necessary for most machine learning algorithms that work with numbers.

Model Building and Evaluation:

I split the data into training and testing sets using train_test_split. The training data is used to fit the model, and the testing data is used to evaluate how well the model generalizes to unseen data.
I trained a Logistic Regression model, which is a good starting point for classification problems. I evaluated its performance using the classification_report function, which provides metrics like precision, recall, and F1-score for each class (with or without heart disease).
I compared the Logistic Regression model with other algorithms like KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier, GaussianNB, and SVC (Support Vector Classifier). I used a linear kernel for SVC.
I used scikit-learn's accuracy_score function to compare the accuracy of these models on the testing data. Random Forest achieved the highest accuracy of 96%.

Visualization:

To visualize the performance of the Random Forest model, I plotted the ROC curve using roc_curve and plot from matplotlib. The ROC curve shows the trade-off between true positive rate (correctly identifying people with heart disease) and false positive rate (incorrectly identifying healthy people as having heart disease).
I also created a decision tree for the DecisionTreeClassifier model using scikit-learn's export_text function and plotted it using plot_tree from the same library. This visualization helps to understand the decision-making process of the model.

Comparison and Conclusion:

Finally, I summarized the accuracy of all the models in a pandas DataFrame and created a bar chart using seaborn to see which model performed best.
Overall, this project explored using machine learning to predict heart disease based on various patient characteristics. The Random Forest Classifier achieved the highest accuracy of 96% on the test data. However, it's important to note that this is just a starting point, and further exploration and fine-tuning of models could potentially improve the results.
