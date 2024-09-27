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
df = pd.read_csv('water_potability.csv')
print(df.head())
df.dropna(inplace=True)
#feature importance
# Feature importance
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Prepare the data
X = df.drop('Potability', axis=1)
y = df['Potability']
print(y.value_counts())

## Create and fit the random forest classifier
# rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# rf_classifier.fit(X, y)

# # Get feature importances
# importances = rf_classifier.feature_importances_
# feature_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)

# # Plot feature importances
# plt.figure(figsize=(10, 6))
# feature_importances.plot(kind='bar')
# plt.title('Feature Importances')
# plt.xlabel('Features')
# plt.ylabel('Importance')
# plt.tight_layout()
# plt.show()

# # Print feature importances
# print("Feature Importances:")
# for feature, importance in feature_importances.items():
#     print(f"{feature}: {importance:.4f}")


#split the data into train test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'Decision Tree': DecisionTreeClassifier(),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Perceptron': Perceptron(),
        'Logistic Regression': LogisticRegression()
    }
    
    plt.figure(figsize=(15, 15))
    
    for i, (name, model) in enumerate(models.items(), 1):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        #F1 score
        from sklearn.metrics import accuracy_score,classification_report
        f1 = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {f1:.4f}")
        print(classification_report(y_test, y_pred))
        
        # learning curve
        train_sizes, train_scores, test_scores = learning_curve(
             model, X_train, y_train, cv=5, n_jobs=-1)
        
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

def evaluate_parameters(X_train, X_test, y_train, y_test):
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.model_selection import validation_curve
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    models = {
        'Decision Tree': {
            'pipeline': Pipeline([
                ('clf', DecisionTreeClassifier())
            ]),
            'params': {
                'clf__max_depth': range(1, 20),
                'clf__min_samples_split': range(2, 11),
                'clf__min_samples_leaf': range(1, 11)
            }
        },
        'K-Nearest Neighbors': {
            'pipeline': Pipeline([
                ('clf', KNeighborsClassifier())
            ]),
            'params': {
                'clf__n_neighbors': range(1, 20),
                'clf__weights': ['uniform', 'distance'],
                'clf__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
            }
        },
        'Logistic Regression': {
            'pipeline': Pipeline([
                ('clf', LogisticRegression())
            ]),
            'params': {
                'clf__C': np.logspace(-4, 4, 20),
                'clf__solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
                'clf__max_iter': range(50,1500,50)
            }
        },
        'Perceptron': {
            'pipeline': Pipeline([
                ('clf', Perceptron())
            ]),
            'params': {
                'clf__alpha': np.logspace(-5, 0, 20),
                'clf__eta0': [0.1, 0.5, 1.0],
                'clf__penalty': ['l2', 'l1', 'elasticnet']
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
                cv=kfold, scoring="accuracy", n_jobs=-1
            )
            
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)
            
            ax = axes[plot_index]
            ax.set_title(f"{name} - {param_name}")
            ax.set_xlabel(param_name)
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0.0, 1.1)
            lw = 2
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


#evaluate_parameters(X_train, X_test, y_train, y_test)
train_and_evaluate_models(X_train, X_test, y_train, y_test)
#train_and_evaluate_models(X_train, X_test, y_train, y_test)

