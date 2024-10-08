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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score

def gridsearch_learning_curves(filename="backup_heart.pdf"):
    import numpy as np
    np.random.seed(42)

    df = pd.read_csv('heart.csv')

    cat_cols = ['sex','cp','fbs','restecg','exang','slope','thal']  
    for i in cat_cols:
        df[i] = df[i].astype('object')

    df = pd.get_dummies(df)

    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define the models and their hyperparameters for GridSearchCV
    models = {
        'Decision Tree': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {
                'max_depth': range(5, 10),
                # 'min_samples_split': np.linspace(0.001, 0.4, 10),
                # 'min_samples_leaf': range(1, 11),
                # 'ccp_alpha': np.linspace(0.0001, 0.01, 5),
                'max_features': np.linspace(0.1, 1, 10),
            }
        },
        'K-Nearest Neighbors': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': range(1, 20),
                # 'weights': ['uniform', 'distance'],
                # 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'p': [1,2,3,4,5,6,7]
            }
        },
        'Logistic Regression': {
            'model': LogisticRegression(solver='saga',penalty='elasticnet',random_state=42),
            'params': {
                'class_weight':[{0: 1.0, 1: 1.0}, 'balanced',{0: 0.5555555555555556, 1: 0.4444444444444444},
                                {0: 0.4444444444444444, 1: 0.5555555555555556}, 
                                {0: 0.33333333333333337, 1: 0.6666666666666666}],
                'l1_ratio': np.linspace(0.3,0.7,4),
                'C': np.linspace(0.0001,0.25,4),
                #'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                #'penalty': ['l1','l2','elasticnet'],
            }
        },
        'Perceptron': {
            'model': Perceptron(random_state=42),
            'params': {
                    'class_weight':[{0: 1.0, 1: 1.0}, 'balanced', 
                                    {0: 0.5555555555555556, 1: 0.4444444444444444}, 
                                    {0: 0.4444444444444444, 1: 0.5555555555555556}, 
                                    {0: 0.33333333333333337, 1: 0.6666666666666666}],
                'penalty': ['l1','l2','elasticnet'],
                'l1_ratio': np.linspace(0,1,10),
            }
        }
    }

    # Perform GridSearchCV for each model
    best_params = {}
    scores = {}
    for name, config in models.items():
        print([{0:1.0, 1:1.0}]+['balanced']+[{0:1.0-x, 1:x} for x in np.linspace(0.0,1,10)])
        grid_search = GridSearchCV(config['model'], config['params'], cv=5, scoring='recall_macro', n_jobs=-1,verbose=3)
        grid_search.fit(X_train, y_train)
        best_params[name] = grid_search.best_params_
        if name == 'Logistic Regression':
            best_params[name]['solver'] = 'saga'
            best_params[name]['penalty'] = 'elasticnet'
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
    with open('heart_grid_search_results.txt', 'w') as f:
        for name, params in best_params.items():
            f.write(f"\n{name} best 3 hyperparameters:\n")
            for param, value in list(params.items()):
                f.write(f"{param}= {value}\n")
        f.write("\nBest scores for each model:\n")
        for name, score in scores.items():
            f.write(f"{name}: {score}\n")

    #plotting the learning curve for each model with the best parameters
    figure=[]
    for name, config in models.items():
        model = config['model'].set_params(**best_params[name])
        train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train,scoring='recall_macro', 
                                                                cv=5,shuffle=True, n_jobs=-1,random_state=42,
                                                                train_sizes=np.linspace(0.1, 1.0, 100))
        fig,ax=plt.subplots(figsize=(8, 6))
        ax.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
        ax.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-validation score')
        ax.legend(loc="best")
        ax.grid(True)
        ax.set_title(f'Learning Curve for {name}')
        ax.set_xlabel('Training examples')
        ax.set_ylabel('Score')
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
                    if key == 'penalty' or key == 'solver' or key == 'algorithm' or key == 'metric':
                        j[key] = value
                    else:
                        j[key]=round(value,4)

    with open('heart_test_set_scores.txt', 'w') as f:
        for name,config in models.items():
            test_model = config['model'].set_params(**best_params[name])
            temp = []
            for key,value in best_params[name].items():
                temp.append(f"{key}={value}")
            #create a table with columns:MODEl, best parameter, time to train, recall,
            #use mathplotlib to create a table
            import time
            start_time = time.time()
            test_model.fit(X_train,y_train)
            total_time = time.time()-start_time
            total_time = round(total_time,3)
            y_pred = test_model.predict(X_test)
            recall = recall_score(y_test, y_pred,average='macro')
            recall = round(recall,3)
            print(f"\n\n\nModel: {name},\nBest Parameters: {temp}, \nTime to Train: {total_time}, \nRecall Score: {recall}")
            f.write(f"\n\n\nModel: {name},\nBest Parameters: {temp}, \nTime to Train: {total_time}, \nRecall Score: {recall}")


        

    
    import plots_to_pdf
    plots_to_pdf.to_pdf(figure,filename)
