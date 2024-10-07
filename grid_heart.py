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
from sklearn.model_selection import train_test_split
df = pd.read_csv('heart.csv')

from sklearn.model_selection import GridSearchCV

# Prepare the data
X = df.drop('target', axis=1)
X = pd.get_dummies(X)
y = df['target']
print(y.value_counts())

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the models and their hyperparameters for GridSearchCV
models = {
    # 'Decision Tree': {
    #     'model': DecisionTreeClassifier(),
    #     'params': {
    #         'max_depth': range(1, 20),
    #         'min_samples_split': range(2, 11),
    #         'min_samples_leaf': range(1, 11)
    #     }
    # },
    # 'K-Nearest Neighbors': {
    #     'model': KNeighborsClassifier(),
    #     'params': {
    #         'n_neighbors': range(1, 20),
    #         'weights': ['uniform', 'distance'],
    #         'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    #     }
    # },
    # 'Logistic Regression': {
    #     'model': LogisticRegression(max_iter=1000),
    #     'params': {
    #         'C': np.logspace(-4, 4, 20),
    #         'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
    #     }
    # },
    'Perceptron': {
        'model': Perceptron(tol=1e-3),
        'params': {
                'class_weight': ['balanced']+[{0:1.0-x, 1:x} for x in np.linspace(0.0,1,10)],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'eta0': [0.1, 0.5, 0.9, 1.0],
            }
    }
}

# Perform GridSearchCV for each model
best_params = {}
scores = {}
for name, config in models.items():
    grid_search = GridSearchCV(config['model'], config['params'], cv=5, scoring='f1_weighted', n_jobs=-1,verbose=2)
    grid_search.fit(X_train, y_train)
    # best_params[name] = grid_search.best_params_
    # scores[name] = grid_search.best_score_
    # print(f"Best parameters for {name}: {grid_search.best_params_}")
    # print(f"Best score for {name}: {grid_search.best_score_}")
    # #sort the mean test scores with their params
    # sorted_mean_test_scores = sorted(zip(grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['params']), key=lambda x: x[0], reverse=True)
    # print(sorted_mean_test_scores)
    # #calculate hyper parameter importance and how it affects the score
    # #plot the score against the hyper parameter
    # plt.figure()
    # plt.plot(sorted_mean_test_scores)
    # plt.xlabel('Hyperparameter')
    # plt.ylabel('Mean Test Score')
    # plt.title(f'Hyperparameter Importance for {name}')
    # plt.show()
    # if name == 'Perceptron':
    #     #print all mean test scores
    #     print(grid_search.cv_results_['mean_test_score'])
    #     print(grid_search.cv_results_['params'])
#validation curve
for name, config in models.items():
    from sklearn.model_selection import validation_curve
    plt.figure()
    train_scores, test_scores = validation_curve(config['model'], X_train, y_train, param_name='eta0', param_range=np.logspace(-5, 1, 20), cv=5, scoring='f1_weighted')
    plt.plot(np.logspace(-5, 0, 20), train_scores.mean(axis=1), label='Training score')
    plt.plot(np.logspace(-5, 0, 20), test_scores.mean(axis=1), label='Cross-validation score')
    plt.xlabel('eta0')
    plt.ylabel('F1 Score')
    plt.title(f'Validation Curve for {name}')
    plt.legend()
    plt.show()
    print(test_scores)



# Display the best 3 hyperparameters for each model
for name, params in best_params.items():
    print(f"\n{name} best 3 hyperparameters:")
    for param, value in list(params.items())[:3]:
        print(f"{param}: {value}")

# Display the best scores for each model
print("\nBest scores for each model:")
for name, score in scores.items():
    print(f"{name}: {score}")


