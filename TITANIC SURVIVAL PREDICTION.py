#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Step 1: Load the dataset
import pandas as pd
import seaborn as sns


# In[3]:


# Load the Titanic dataset from Seaborn
data = pd.read_csv("F:\codsoft\dataset\Titanic-Dataset.csv")


# In[5]:


# Display the first few rows of the dataset
print(data.head())


# In[17]:


data.columns


# In[18]:


# Step 2: Explore the data
print(data.info())


# In[19]:


data.isna().sum()


# In[20]:


print(data.describe())


# In[21]:


print(data.isnull().sum())


# In[22]:


# Step 3: Preprocess the data
# Fill missing values
data['Age'].fillna(data['Age'].median(), inplace=True)


# In[11]:


sns.heatmap(data.corr(),cmap='YlGnBu',annot=True,fmt='.2f')


# In[23]:


data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)


# In[24]:


data['Fare'].fillna(data['Fare'].median(), inplace=True)


# In[25]:


# Drop columns that won't be used in the model
data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)


# In[26]:


# Encode categorical variables
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)


# In[27]:


# Step 4: Split the data
from sklearn.model_selection import train_test_split


# In[28]:


# Define features and target variable
X = data.drop(columns=['Survived'])
y = data['Survived']


# In[29]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[30]:


# Step 5: Build the model
from sklearn.ensemble import RandomForestClassifier


# In[31]:


# Create a RandomForestClassifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)


# In[32]:


# Train the model
model.fit(X_train, y_train)


# In[33]:


# Step 6: Evaluate the model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[34]:


# Make predictions on the test set
y_pred = model.predict(X_test)


# In[35]:


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')


# In[36]:


# Display confusion matrix
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))


# In[37]:


# Display classification report
print('Classification Report:')
print(classification_report(y_test, y_pred))


# In[ ]:




