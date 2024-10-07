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
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('churn.csv')
df = df.dropna()
df = df.drop(['RowNumber','CustomerId','Surname'], axis=1)
#feature importance


#print(df.head())
df.columns = df.columns.str.replace("'", "") #there were apostrophes in the column names
#smote to balance the data

#apply label encoder to categorical columns
df=pd.get_dummies(df)
print(df.head())

X = df.drop('Exited', axis=1)
y = df['Exited']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,stratify=y)

print((X_train.columns))
print(y_train.value_counts())


# Scaling
scaler = RobustScaler()
X_train  = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the models and their hyperparameters for GridSearchCV
models = {
    'Decision Tree': {
        'model': DecisionTreeClassifier(random_state=42),
        'params': {
            'max_depth': range(1, 15),
            'min_samples_split': [2,3,4],
            'min_samples_leaf': [1,2,3],
            'class_weight': [{0: 1.0, 1: 1.0}, 'balanced', {0: 0.8888888888888888, 1: 0.1111111111111111}, 
                             {0: 0.7777777777777778, 1: 0.2222222222222222}, 
                             {0: 0.6666666666666667, 1: 0.3333333333333333}, 
                            {0: 0.4444444444444444, 1: 0.5555555555555556}, 
                            {0: 0.33333333333333337, 1: 0.6666666666666666}, 
                            {0: 0.22222222222222232, 1: 0.7777777777777777}, 
                            {0: 0.11111111111111116, 1: 0.8888888888888888}],
        }
    },
    'K-Nearest Neighbors': {
        'model': KNeighborsClassifier(n_jobs=-1),
        'params': {
            'n_neighbors': range(4, 11),
            'weights': ['uniform', 'distance'],
        }
    },
    'Logistic Regression': {
        'model': LogisticRegression(max_iter=1000,random_state=42),
        'params': {
            'class_weight': [{0: 1.0, 1: 1.0}, 'balanced', {0: 0.4444444444444444, 1: 0.5555555555555556}, 
                            {0: 0.33333333333333337, 1: 0.6666666666666666}],
            'C': np.linspace(0.01, 3, 10),
            'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
            'penalty': ['l1', 'l2', 'elasticnet'],
        }
    },
    'Perceptron': {
        'model': Perceptron(random_state=42),
        'params': {
                'class_weight': [{0:1.0, 1:1.0}]+['balanced']+[{0:1.0-x, 1:x} for x in np.linspace(0.0,1,10)],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'alpha' : np.linspace(0.00001,0.01,5),
                'eta0' : np.linspace(0.01,2,10),
            }
    }
}

# Perform GridSearchCV for each model
best_params = {}
scores = {}
for name, config in models.items():
    grid_search = GridSearchCV(config['model'], config['params'], cv=5, scoring='f1_weighted', n_jobs=-1,verbose=3)
    grid_search.fit(X_train, y_train)
    best_params[name] = grid_search.best_params_
    scores[name] = grid_search.best_score_
    print(f"Best parameters for {name}= {grid_search.best_params_}")
    print(f"Best score for {name}: {grid_search.best_score_}")
# Display the best 3 hyperparameters for each model
for name, params in best_params.items():
    print(f"\n{name} best 3 hyperparameters:")
    for param, value in list(params.items()):
        print(f"{param}: {value}")

# Display the best scores for each model
print("\nBest scores for each model:")
for name, score in scores.items():
    print(f"{name}: {score}")

#save everything to a text file
with open('churn_grid_search_results.txt', 'w') as f:
    for name, params in best_params.items():
        f.write(f"{name} best 3 hyperparameters:\n")
        for param, value in list(params.items()):
            f.write(f"{param}: {value}\n")
    f.write("\nBest scores for each model:\n")
    for name, score in scores.items():
        f.write(f"{name}: {score}\n")

