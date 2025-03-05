# Microsoft-Classifying-Cybersecurity-Incidents 

OBJECTIVE:

      The goal of this project is to build a machine learning classification model  that can accurately predict the triage grade of cybersecurity incidents. Utilizing the comprehensive GUIDE dataset, my goal is to create a classification model that categorizes incidents as true positive (TP), benign positive (BP), or false positive (FP) based on historical evidence and customer responses

PROBLEM STATEMENT:

      I trained the model using the train.csv dataset and provide evaluation metrics—macro-F1 score, precision, and recall—based on the model's performance on the test.csv dataset. This ensures that the model is not only well-trained but also generalizes effectively to unseen data, making it reliable for real-world applications.

SCOPE: 
    
    The solution developed in this project can be implemented in various business scenarios, particularly in the field of cybersecurity, in Security Operation Centers (SOCs),Incident Response Automation, Threat Intelligence, Enterprise Security Management. 

INITIAL STEPS: 

    Due to my given dataset is large, and to prevent the problem arises, when we run on local, i used stratified sampling based on the target class and select around 5 lakh data for every class. So by doing this you will get around 15 lakh data. So train your model in train dataset and do the predictions in your test data set.

STEPS

1)DATA EXPLORATION AND UNDERSTANDING:

    By loading the train.csv dataset and perform an initial inspection to understand the structure of the data, including the number of features, types of variables (categorical, numerical), and the distribution of the target variable (IncidentGrade)(TP, BP, FP). Use visualizations and statistical summaries to identify patterns, correlations, and potential anomalies in the data. Pay special attention to class imbalances, as they may require specific handling strategies later on. In this i mainly performed Histogram, Boxplot, Piechart, and Heatmap etc. These are done in the format of univariate analysis, bivariate analysis, and also performed as whole, to understand the data initially before starting the data cleaning and preprocessing.

2) DATA CLEANING AND PREPROCESSING:
   
     i)HANDLING MISSING VALUES:

             For majority of features in my dataset, i treated as categorical columns and  uses approaches like mode imputation and also filled the missing values with  "unknown".The Tool used widely in handling missing values is Pandas. Initially the columns with majority of null values is removed.

     ii)FEATURE IMPORTANCE:

             In this, I used Chi-Square test to find the most important features related to the target.I removed the features which are less importance.

     iii)DATA TRANSFORMATION:

             Due to considering  all columns as categorical columns, i performed Target Encoding for all features. Reason to select this that unique values are not in order form, if it is in order form i would have gone for label encoding.

3)MODEL DEVELOPMENT:

     i)MODEL SELECTION: 
     
             Here target is multi classification, so i selected models such as Decision tree, Random Forest, XG Boost and AdaBoost. Main reason for this model selection that they can handle non linear models and can handle large datsets. 

    ii)MODEL SPLIT:
              
              Dataset is usually split by train test split as 80/30 or 80/20 in these cases. 

    iii)MODEL TRAINING:
    
              First done trial and error method to choose Best hyperparameter for each and every model. After Hyperparameter selection it is trained in models. 

4)MODEL EVALUATION:

    i) EVALUATION METRICS: 
              
              Accuracy for Overall performance of the model. Precision, Recall, F1-Score for Metrics to understand how well the model handles each class, especially for imbalanced data. Confusion Matrix for Display confusion matrices for each model to assess how well they perform across different IncidentGrade(Target) classes.This Evaluation metrics are done for each and every model to compare and choose the best model. After selecting the best model, i performed hyperparameter tuning for increasing model performance.

    ii)MODEL COMPARISON:
              
              1)Accuracy: Random Forest has the highest accuracy (0.89), followed by Decision Tree and XGBoost.
              
              2)Precision, Recall, and F1-Score: Random Forest and Decision Tree excelled in precision, recall, and F1-score across most classes, with Random Forest slightly outperforming. XGBoost performed well for class 1, while AdaBoost struggled, particularly with class 1 and recall for class 0.
              
              3)Confusion Matrix: Decision Tree and Random Forest show balanced performance, with Random Forest having slightly more accurate predictions, especially for class 0 and 2. XGBoost struggles with misclassifications for class 1, particularly with false positives. AdaBoost underperforms overall, especially for class 1, showing a high number of false positives and misclassifications for class 0.

              4)Conclusion: Based on the accuracy score, classification report, and confusion matrix, the Random Forest model is the best among the three. It has the highest accuracy and overall better performance across all metrics.

    iii)BEST MODEL: 

             Since Random Forest seems to be the best model, it would be the ideal candidate for hyperparameter tuning to further improve its performance. 

5)FINAL EVALUATION ON TEST DATA:

      Once the model is finalized and optimized, I evaluated it on the test.csv dataset. Random_forest is best model chosen from train dataset, when it is tested on unseen test data it gives a accuracy of 0.86. Precision,recall,f1-score too comes with good numbers in confusion matrix: class0,class1,class2 correctly identifies their respective classes in large number.

6)CONCLUSION: 

    i)SUMMARY OF RESULTS:
                  
                  The Random classifier performed better than other models with the good accuracy of 89%, when compare to another models.The model can predict the Cybersecurity Incident classes with a reasonable degree of accuracy. When done Evaluation on unseen test data it gives accuracy of 86%,and good result for precision, recall and F1 score, and overall prediction among three classess in confusion matrix seams to be good.

    ii)LIMITATIONS:
    
                  The challenges faced in this project are: Some Evaluation metrics performed good and shows better result and some result seems to be average. 

    iii)FUTURE WORK:
                  
                  IMPROVEMENT AREAS:Incorporating additional features, Testing other advanced models (e.g., deep learning or neural networks) for better performance.
 
