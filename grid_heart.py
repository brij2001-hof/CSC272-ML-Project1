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

# # Define the models and their hyperparameters for GridSearchCV
# models = {
#     # 'Decision Tree': {
#     #     'model': DecisionTreeClassifier(),
#     #     'params': {
#     #         'max_depth': range(1, 20),
#     #         'min_samples_split': range(2, 11),
#     #         'min_samples_leaf': range(1, 11)
#     #     }
#     # },
#     # 'K-Nearest Neighbors': {
#     #     'model': KNeighborsClassifier(),
#     #     'params': {
#     #         'n_neighbors': range(1, 20),
#     #         'weights': ['uniform', 'distance'],
#     #         'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
#     #     }
#     # },
#     # 'Logistic Regression': {
#     #     'model': LogisticRegression(max_iter=1000),
#     #     'params': {
#     #         'C': np.logspace(-4, 4, 20),
#     #         'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
#     #     }
#     # },
#     'Perceptron': {
#         'model': Perceptron(tol=1e-3),
#         'params': {
#             'max_iter': range(500,2500,250),
#             'alpha': np.logspace(-5, 0, 20),
#             'eta0': np.logspace(-5, 1, 20),
#             'penalty': ['l2', 'l1', 'elasticnet']
#         }
#     }
# }

# # Perform GridSearchCV for each model
# best_params = {}
# scores = {}
# for name, config in models.items():
#     grid_search = GridSearchCV(config['model'], config['params'], cv=5, scoring='f1_weighted', n_jobs=-1)
#     grid_search.fit(X_train, y_train)
#     # best_params[name] = grid_search.best_params_
#     # scores[name] = grid_search.best_score_
#     # print(f"Best parameters for {name}: {grid_search.best_params_}")
#     # print(f"Best score for {name}: {grid_search.best_score_}")
#     # #sort the mean test scores with their params
#     # sorted_mean_test_scores = sorted(zip(grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['params']), key=lambda x: x[0], reverse=True)
#     # print(sorted_mean_test_scores)
#     # #calculate hyper parameter importance and how it affects the score
#     # #plot the score against the hyper parameter
#     # plt.figure()
#     # plt.plot(sorted_mean_test_scores)
#     # plt.xlabel('Hyperparameter')
#     # plt.ylabel('Mean Test Score')
#     # plt.title(f'Hyperparameter Importance for {name}')
#     # plt.show()
#     # if name == 'Perceptron':
#     #     #print all mean test scores
#     #     print(grid_search.cv_results_['mean_test_score'])
#     #     print(grid_search.cv_results_['params'])
# #validation curve
# for name, config in models.items():
#     from sklearn.model_selection import validation_curve
#     plt.figure()
#     train_scores, test_scores = validation_curve(config['model'], X_train, y_train, param_name='eta0', param_range=np.logspace(-5, 1, 20), cv=5, scoring='f1_weighted')
#     plt.plot(np.logspace(-5, 0, 20), train_scores.mean(axis=1), label='Training score')
#     plt.plot(np.logspace(-5, 0, 20), test_scores.mean(axis=1), label='Cross-validation score')
#     plt.xlabel('Alpha')
#     plt.ylabel('F1 Score')
#     plt.title(f'Validation Curve for {name}')
#     plt.legend()
#     plt.show()
#     print(test_scores)



# # Display the best 3 hyperparameters for each model
# for name, params in best_params.items():
#     print(f"\n{name} best 3 hyperparameters:")
#     for param, value in list(params.items())[:3]:
#         print(f"{param}: {value}")

# # Display the best scores for each model
# print("\nBest scores for each model:")
# for name, score in scores.items():
#     print(f"{name}: {score}")


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
        #         ('clf', DecisionTreeClassifier())
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
                ('clf', LogisticRegression())
            ]),
            'params': {
                 'clf__class_weight': ['balanced']+[{0:1.0-x, 1:x} for x in np.linspace(0.0,1,10)],

            }
        },
        'Perceptron': {
            'pipeline': Pipeline([
                ('clf', Perceptron())
            ]),
            'params': {
                'clf__class_weight': ['balanced']+[{0:1.0-x, 1:x} for x in np.linspace(0.0,1,10)],
                'clf__max_iter': [100,200,300,400,500,600,700,800,900,1000,1100],
            }
        }
    }

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    # Calculate the total number of subplots
    total_plots = sum(len(config['params']) for config in models.values())
    rows = int(np.ceil(np.sqrt(total_plots)))
    cols = int(np.ceil(total_plots / rows))

    # Create a single figure for all plots
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
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
            print('len train mean',len(train_mean))
            train_std = np.std(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)
            ax = axes[plot_index]
            if param_name == 'clf__class_weight':
                temp=[];c=0
                for i in range(len(param_range)):
                    print(len(param_range))
                    print(i)
                    print("ssss")
                    if param_range[i] == 'balanced':
                        temp.append("{i}".format(i=param_range[i]))
                    else:
                        temp.append("{i:.2f},{j:.2f}".format(i=param_range[i][0],j=param_range[i][1]))
            param_range=temp
            param_name='class_weight class0,class1'
            ax.set_title(f"{name} - {param_name}")
            ax.set_xlabel(param_name)
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0.0, 1.1)
            lw = 2
            print(param_name)
            ax.plot(param_range, train_mean, label="Training score", color="darkorange", lw=lw)
            ax.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.2, color="darkorange", lw=lw)
            ax.plot(param_range, test_mean, label="Cross-validation score", color="navy", lw=lw)
            ax.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.2, color="navy", lw=lw)
            ax.legend(loc="best")
            ax.grid(True)
            
            plot_index += 1

    # Remove any unused subplots
    for i in range(plot_index, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

evaluate_parameters(X_train, X_test, y_train, y_test)

