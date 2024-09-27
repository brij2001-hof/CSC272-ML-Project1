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
df = pd.read_csv('Employee.csv')
print(df.head())
#feature importance
# Feature importance
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

#all categorical columns
cat_cols = [col for col in df.columns if df[col].dtype == 'object']

#apply label encoder to categorical columns
for col in cat_cols:
    df[col] = le.fit_transform(df[col])
# Prepare the data
X = df.drop('LeaveOrNot', axis=1)
y = df['LeaveOrNot']

## Create and fit the random forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X, y)

# Get feature importances
importances = rf_classifier.feature_importances_
feature_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
feature_importances.plot(kind='bar')
plt.title('Feature Importances')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.tight_layout()
plt.show()

# Print feature importances
print("Feature Importances:")
for feature, importance in feature_importances.items():
    print(f"{feature}: {importance:.4f}")

X.drop('EverBenched', axis=1, inplace=True)
#split the data into train test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'Decision Tree': DecisionTreeClassifier(max_depth=9, min_samples_split=2, min_samples_leaf=1),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=9, weights='uniform', algorithm='brute'),
        'Logistic Regression': LogisticRegression(C=1, solver='lbfgs', penalty='l2', max_iter=100, tol=0.0001),
        'Perceptron': Perceptron(alpha=0.001, max_iter=100, eta0=0.1, penalty='elasticnet', early_stopping=True)
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
             model, X_train, y_train)
        
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
        plt.text(0.5, 0.5, f"{name} (tuned) F1 Score: {f1:.4f}", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
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

    models = {
        'Decision Tree': {
            'model': DecisionTreeClassifier(),
            'params': {
                'max_depth': range(1, 20),
                'min_samples_split': range(2, 11),
                'min_samples_leaf': range(1, 11)
            }
        },
        'K-Nearest Neighbors': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': range(1, 20),
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
            }
        },
        'Logistic Regression': {
            'model': LogisticRegression(),
            'params': {
                'C': np.logspace(-4, 4, 20),
                'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
                'max_iter': [100, 200, 300, 400, 500]
            }
        },
        'Perceptron': {
            'model': Perceptron(),
            'params': {
                'alpha': np.logspace(-5, 0, 20),
                'max_iter': range(1000,5000),
                'eta0': [0.1, 0.5, 1.0],
                'penalty': ['l2', 'l1', 'elasticnet']
            }
        }
    }

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    for name, config in models.items():
        model = config['model']
        params = config['params']
        
        for param_name, param_range in params.items():
            train_scores, test_scores = validation_curve(
                model, X_train, y_train, param_name=param_name, param_range=param_range,
                cv=kfold, scoring="accuracy", n_jobs=-1
            )
            
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)
            
            plt.figure(figsize=(10, 6))
            plt.title(f"{name} - {param_name}")
            plt.xlabel(param_name)
            plt.ylabel("Score")
            plt.ylim(0.0, 1.1)
            lw = 1  # Thin lines
            plt.bar(param_range, train_mean, yerr=train_std, label="Training score", color="darkorange", alpha=0.6, lw=lw)
            plt.bar(param_range, test_mean, yerr=test_std, label="Cross-validation score", color="navy", alpha=0.6, lw=lw)
            plt.legend(loc="best")
            plt.show()
    
    
#train_and_evaluate_models(X_train, X_test, y_train, y_test)
evaluate_parameters(X_train, X_test, y_train, y_test)

#a function to find distribution of each column
import seaborn as sns
def find_distribution(df):
    plt.figure(figsize=(10, 6))
    for i, col in enumerate(df.columns, 1):
        plt.subplot(4, 4, i)
        sns.histplot(df[col], kde=True)
        plt.title(f"{col} Distribution")
        plt.xlabel(col)
        plt.ylabel("Frequency")    
    plt.show()

#find_distribution(df)

