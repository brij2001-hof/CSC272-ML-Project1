import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score

def gridsearch_learning_curves(filename='backup_churn.csv'):
    import numpy as np
    np.random.seed(42)

    df = pd.read_csv('churn.csv')
    df = df.dropna()
    df = df.drop(['RowNumber','CustomerId','Surname'], axis=1)
    df.columns = df.columns.str.replace("'", "") #there were apostrophes in the column names

    #getting dummies for categorical columns
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
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': range(4, 11),
                'weights': ['uniform', 'distance'],
                'p': [1,2,3,4],
            }
        },
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42),
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
            f.write(f"\n\n{name} best 3 hyperparameters:\n")
            for param, value in list(params.items()):
                f.write(f"{param}: {value}\n")
        f.write("\nBest scores for each model:\n")
        for name, score in scores.items():
            f.write(f"{name}: {score}\n")

    figure=[]
    mean_fit_times = {}
    mean_train_scores = {}
    for name, config in models.items():
        model = config['model'].set_params(**best_params[name])
        train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(model, X_train, y_train,scoring='f1_weighted', 
                                                                cv=5,shuffle=True, n_jobs=-1,random_state=42,
                                                                train_sizes=np.linspace(0.1, 1.0, 100),
                                                                return_times=True)
        fig,ax=plt.subplots(figsize=(8, 6))
        mean_fit_times[name] = np.mean(fit_times)
        mean_train_scores[name] = np.mean(train_scores)
        ax.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
        ax.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-validation score')
        ax.legend(loc="best")
        ax.grid(True)
        ax.set_title(f'Learning Curve for {name}')
        ax.set_xlabel('Training examples')
        ax.set_ylabel('F1Score')
        figure.append(fig)

    for i,j in best_params.items():
            #round values
            for key,value in j.items():
                if key == 'class_weight':
                    if value == 'balanced':
                        j[key] = value
                    else:
                        for class_,weight in value.items():
                            value[class_]=round(weight,2)
                else:
                    if key == 'penalty' or key == 'solver' or key == 'algorithm' or key == 'metric' or key == 'weights':
                        j[key] = value
                    else:
                        j[key]=round(value,5)

    with open('churn_train_test_set_scores.txt', 'w') as f:
        for name,config in models.items():
            test_model = config['model'].set_params(**best_params[name])
            temp = []
            for key,value in best_params[name].items():
                temp.append(f"{key}={value}")
            test_model.fit(X_train,y_train)
            y_pred = test_model.predict(X_test)
            f1 = f1_score(y_test, y_pred,average='weighted')
            f1 = round(f1,3)
            #plot confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            fig,ax=plt.subplots(figsize=(8, 6))
            import seaborn as sns
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f"{name} - Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            figure.append(fig)

            print(f"\n\n\nModel: {name},\nBest Parameters: {temp}, \nTime to Train: {mean_fit_times[name]},\n Training Score: {mean_train_scores[name]},\nF1_weighted TEST Score: {f1}")
            f.write(f"\n\n\nModel: {name},\nBest Parameters: {temp}, \nTime to Train: {mean_fit_times[name]}, \nTraining Score: {mean_train_scores[name]}, \nF1_weighted TEST Score: {f1}")
        f.close()



    import plots_to_pdf
    plots_to_pdf.to_pdf(figure,filename)

if __name__ == "__main__":
    import datetime
    filename = datetime.datetime.now().strftime("churn_learning_curves_%Y-%m-%d_%H-%M-%S.pdf")
    gridsearch_learning_curves(filename=filename)
