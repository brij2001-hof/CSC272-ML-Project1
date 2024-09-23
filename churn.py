import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler

print("Running dataset1")


# Load the data
df = pd.read_csv('churn.csv')
df = df.dropna()
df = df.drop(['RowNumber','CustomerId','Surname'], axis=1)
#feature importance


#print(df.head())
df.columns = df.columns.str.replace("'", "") #there were apostrophes in the column names
#encode categorical data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

#all categorical columns
cat_cols = [col for col in df.columns if df[col].dtype == 'object']

#apply label encoder to categorical columns
for col in cat_cols:
    df[col] = le.fit_transform(df[col])
print(df.head())
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
X = df.drop('Exited', axis=1)
y = df['Exited']
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()
#split the data into train test split
from sklearn.model_selection import train_test_split
X = df.drop('Exited', axis=1)
y = df['Exited']
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)



# Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)
# X_test = scaler.transform(X_test)

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'Decision Tree': DecisionTreeClassifier(),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Perceptron': Perceptron(),
        'Logistic Regression': LogisticRegression()
    }
    
    plt.figure(figsize=(15, 15))
    
    for i, (name, model) in enumerate(models.items(), 1):
        # model.fit(X_train, y_train)
        # y_pred = model.predict(X_test)

        #F1 score
        # from sklearn.metrics import f1_score
        # f1 = f1_score(y_test, y_pred, average='weighted')
        # print(f"{name} F1 Score: {f1:.4f}")
        
        # learning curve
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_train, y_train, cv=5, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 10))
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        plt.subplot(2, 2, i)
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")
        plt.title(f"{name} Learning Curve")
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.legend(loc="best")
        plt.grid()
    
    plt.tight_layout()
    plt.show()
    return plt


#train_and_evaluate_models(X_train, X_test, y_train, y_test)
train_and_evaluate_models(X, X, y, y)

