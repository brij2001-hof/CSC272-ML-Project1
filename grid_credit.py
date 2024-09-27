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
df = pd.read_csv('credit.csv')
df.columns = df.columns.str.replace("'", "")
from sklearn.model_selection import GridSearchCV

# Prepare the data
X = df.drop('class', axis=1)
X = pd.get_dummies(X)
y = df['class']
y=y.replace({'bad':0, 'good':1})
print(y.value_counts())
# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the models and their hyperparameters for GridSearchCV
# models = {
#     'Decision Tree': {
#         'model': DecisionTreeClassifier(class_weight={0:7,1:3}),
#         'params': {
#             'max_depth': range(1, 20),
#             'min_samples_split': range(2, 11),
#             'min_samples_leaf': range(1, 11)
#         }
#     },
#     'K-Nearest Neighbors': {
#         'model': KNeighborsClassifier(),
#         'params': {
#             'n_neighbors': range(1, 20),
#             'weights': ['uniform', 'distance'],
#             'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
#         }
#     },
#     'Logistic Regression': {
#         'model': LogisticRegression(max_iter=1000),
#         'params': {
#             'C': np.logspace(-10, 10, 20),
#             'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
#         }
#     },
#     'Perceptron': {
#         'model': Perceptron(max_iter=1000, tol=1e-3),
#         'params': {
#             'alpha': np.logspace(-5, 0, 20),
#             'eta0': [0.1, 0.5, 1.0],
#             'penalty': ['l2', 'l1', 'elasticnet'],
#             'class_weight': [{0:i, 1:j} for i in range(1, 11) for j in range(1, 11)]
#         }
#     }
# }

# # Perform GridSearchCV for each model
# best_params = {}
# scores = {}
# for name, config in models.items():
#     grid_search = GridSearchCV(config['model'], config['params'], cv=5, scoring='f1_weighted', n_jobs=-1)
#     grid_search.fit(X_train, y_train)
#     best_params[name] = grid_search.best_params_
#     scores[name] = grid_search.best_score_
#     print(f"Best parameters for {name}: {grid_search.best_params_}")
#     print(f"Best score for {name}: {grid_search.best_score_}")
# # Display the best 3 hyperparameters for each model
# for name, params in best_params.items():
#     print(f"\n{name} best 3 hyperparameters:")
#     for param, value in list(params.items())[:3]:
#         print(f"{param}: {value}")

# # Display the best scores for each model
# print("\nBest scores for each model:")
# for name, score in scores.items():
#     print(f"{name}: {score}")

# Display the best scores for each model

def evaluate_parameters(X_train, X_test, y_train, y_test):
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.model_selection import validation_curve
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    models = {
        # 'Decision Tree': {
        #     'pipeline': Pipeline([
        #         ('clf', DecisionTreeClassifier(class_weight={0:7,1:3}))
        #     ]),
        #     'params': {
        #         'clf__max_depth': range(1, 20),
        #         'clf__min_samples_split': range(2, 11),
        #         'clf__min_samples_leaf': range(1, 11),
        #         'clf__ccp_alpha': np.linspace(0, 0.02, 20)
        #     }
        # },
        # 'K-Nearest Neighbors': {
        #     'pipeline': Pipeline([
        #         ('clf', KNeighborsClassifier())
        #     ]),
        #     'params': {
        #         'clf__n_neighbors': range(1, 20),
        #         'clf__weights': ['uniform', 'distance'],
        #         'clf__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        #     }
        # },
        'Logistic Regression': {
            'pipeline': Pipeline([
                ('clf', LogisticRegression(max_iter=100))
            ]),
            'params': {
                # 'clf__C': np.logspace(-4, 4, 20),
                # 'clf__solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
                # 'clf__max_iter': [1000, 2000, 3000],
                'clf__class_weight': [{0:i, 1:j} for i in range(1, 11) for j in range(1, 11)]
            }
        },
        'Perceptron': {
            'pipeline': Pipeline([
                ('clf', Perceptron(class_weight={0:7,1:3}))
            ]),
            'params': {
            #     'clf__alpha': np.logspace(-4, 0, 20),
            #     'clf__eta0': [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
                'clf__penalty': ['l2', 'l1', 'elasticnet'],
                'clf__class_weight': [{0:i, 1:j} for i in range(1, 11) for j in range(1, 11)]
            }
        }
    }

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # Calculate the total number of subplots
    total_plots = sum(len(config['params']) for config in models.values())
    rows = int(np.ceil(np.sqrt(total_plots)))
    cols = int(np.ceil(total_plots / rows))

    # Create a single figure for all plots
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows), layout='constrained')
    axes = axes.flatten()  # Flatten the 2D array of axes for easier indexing

    plot_index = 0
    for name, config in models.items():
        pipeline = config['pipeline']
        params = config['params']
        
        for param_name, param_range in params.items():
            train_scores, test_scores = validation_curve(
                pipeline, X_train, y_train, param_name=param_name, param_range=param_range,
                cv=kfold, scoring="f1", n_jobs=-1
            )
            
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)
            if param_name == 'clf__class_weight':
                temp=[]
                for i in range(len(param_range)):
                    temp.append("{i},{j}".format(i=param_range[i][0],j=param_range[i][1]))
                param_range=temp
            ax = axes[plot_index]
            ax.set_title(f"{name} - {param_name}", fontsize=8)
            ax.set_xlabel(param_name, fontsize=8)
            ax.set_ylabel("Accuracy", fontsize=8)
            ax.set_ylim(0.0, 1.1)
            lw = 2
            ax.plot(param_range, train_mean, label="Training score", color="darkorange", lw=lw)
            ax.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.2, color="darkorange", lw=lw)
            ax.plot(param_range, test_mean, label="Cross-validation score", color="navy", lw=lw)
            ax.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.2, color="navy", lw=lw)
            ax.legend(loc="best", fontsize=6)
            ax.grid(True)
            ax.tick_params(axis='both', which='major', labelsize=6)
            
            plot_index += 1

    # Remove any unused subplots
    for i in range(plot_index, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

evaluate_parameters(X_train, X_test, y_train, y_test)

