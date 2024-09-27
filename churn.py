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
df=df.drop_duplicates()
#split the data into train test split
from sklearn.model_selection import train_test_split
X = df.drop('Exited', axis=1)
y = df['Exited']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Scaling
scaler = StandardScaler()
X_train  = scaler.fit_transform(X_train)
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
                'clf__min_samples_leaf': range(1, 11),
                'clf__ccp_alpha': np.linspace(0, 0.02, 20)
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
                 'clf__class_weight': ['balanced']+[{0:1.0-x, 1:x} for x in np.linspace(0.0,1,10)],

            }
        },
        'Perceptron': {
            'pipeline': Pipeline([
                ('clf', Perceptron())
            ]),
            'params': {
                'clf__class_weight': ['balanced']+[{0:x, 1:1.0-x} for x in np.linspace(0.0,1,10)],
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
                cv=kfold, scoring="f1_weighted", n_jobs=-1
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
for i in np.logspace(-5, 0, 20):
    #format to 8 decimal places
    print("{:.8f}".format(i))

evaluate_parameters(X_train, X_test, y_train, y_test)
#train_and_evaluate_models(X_train, X_test, y_train, y_test)
#train_and_evaluate_models(X, X, y, y)

