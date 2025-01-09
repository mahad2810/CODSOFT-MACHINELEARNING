# CODSOFT-MACHINELEARNING




# Machine Learning Internship Projects

This repository contains the work completed during my machine learning internship. Each task showcases a different aspect of machine learning, from text classification and hyperparameter tuning to churn prediction and spam detection. Below is a detailed description of each task.

---

## **Task 1: Genre Classification Using Logistic Regression**

### **Overview**  
In this task, I built a text classification pipeline to predict the genre of a given description using logistic regression with hyperparameter tuning.

### **Steps**  
1. **Dataset Preparation**:  
   - Loaded training and testing data from text files.
   - Handled missing values and cleaned data.

2. **Feature Engineering**:  
   - Applied TF-IDF vectorization with n-grams (1 to 3) and a maximum of 10,000 features.

3. **Class Imbalance Handling**:  
   - Computed class weights to balance the training dataset.

4. **Model Training and Tuning**:  
   - Used Logistic Regression with `GridSearchCV` to find the best hyperparameters.

5. **Validation and Prediction**:  
   - Evaluated the model on a validation set using accuracy and classification reports.
   - Predicted genres for test data and saved results to a CSV file.

---

## **Task 3: Customer Churn Prediction**

### **Overview**  
This task focused on predicting customer churn using multiple machine learning models.

### **Steps**  
1. **Data Preparation**:  
   - Cleaned the dataset and removed irrelevant columns.
   - Encoded categorical variables using label encoding and one-hot encoding.
   - Standardized numerical features for consistent scaling.

2. **Model Training**:  
   - Built three models: Logistic Regression, Random Forest, and Gradient Boosting.
   - Compared performance based on accuracy and feature importance.

3. **Feature Importance**:  
   - Extracted and analyzed feature importance from the Random Forest model.

4. **External Prediction**:  
   - Implemented a function to predict churn for new customer data dynamically.

---

## **Task 4: Spam Classification**

### **Overview**  
Developed a system to classify SMS messages as spam or ham (not spam).

### **Steps**  
1. **Model Training and Saving**:  
   - Trained a spam classifier using a suitable machine learning model and TF-IDF vectorizer.
   - Saved the model and vectorizer using the `pickle` library for later use.

2. **Real-Time Prediction**:  
   - Created a function to load the saved model and vectorizer to classify user-input messages dynamically.

3. **Interactive Testing**:  
   - Allowed users to input SMS messages and receive spam/ham predictions in real-time.

---

## **Key Skills and Tools**

- **Machine Learning**: Logistic Regression, Random Forest, Gradient Boosting, TF-IDF, Hyperparameter Tuning.
- **Python Libraries**: Pandas, NumPy, Scikit-learn, Pickle.
- **Data Preprocessing**: Feature scaling, encoding, handling missing data.
- **Evaluation**: Accuracy, Classification Reports, Feature Importance.

---

Here’s the revised **How to Use** section based on your updated instructions:

---

## **How to Use**

1. Clone this repository and navigate to it:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Open Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

4. In Jupyter Notebook, open and run the respective files for each task:
   - **Task 1:** Open and run `task1.ipynb`  
   - **Task 3:** Open and run `task3.ipynb`  
   - **Task 4:** Open and run `task4.ipynb`  

--- 
