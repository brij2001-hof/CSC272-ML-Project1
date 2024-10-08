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
from sklearn.model_selection import KFold, validation_curve
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
import plots_to_pdf
import seaborn as sns

#set random seed


def evaluate_parameters(filename):
    import numpy as np
    np.random.seed(42)
    print("Running dataset1")


    # Load the data
    df = pd.read_csv('heart.csv')
    print(df.head())
    print(df['target'].value_counts())
    # #correlation graph
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(df.corr(),annot=True, cmap='coolwarm')
    # plt.show()


    cat_cols = ['sex','cp','fbs','restecg','exang','slope','thal'] #for categorical columns
    for i in cat_cols:
        df[i] = df[i].astype('object')

    df = pd.get_dummies(df)
    print(len(df.columns))
    print(df.columns)

    X = df.drop('target', axis=1)
    y = df['target']
    print(y.value_counts())
    #split the data into train test split
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    print(X_train)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    models = {
        'Decision Tree': {
            'pipeline': Pipeline([
                ('clf', DecisionTreeClassifier(random_state=42))
            ]),
            'params': {
                'clf__max_depth': range(1, 20),
                'clf__criterion': ['gini', 'entropy','log_loss'],
                'clf__min_samples_split': range(2, 100),
                'clf__min_samples_leaf': range(1,100),
                'clf__ccp_alpha': np.linspace(0.0001, 0.02, 20),
                'clf__max_features': np.linspace(0.1, 1, 10),
            }
        },
        'K-Nearest Neighbors': {
            'pipeline': Pipeline([
                ('clf', KNeighborsClassifier(n_neighbors=5,algorithm='kd_tree'))
            ]),
            'params': {
                'clf__n_neighbors': range(1, 20),
                'clf__p': [1,2,3,4,5,6,7],
            }
        },
        'Logistic Regression': {
            'pipeline': Pipeline([
                ('clf', LogisticRegression(random_state=42,penalty='elasticnet',solver='saga',l1_ratio=0.5))
            ]),
            'params': {
                 'clf__class_weight': [{0:1.0, 1:1.0}]+['balanced']+[{0:1.0-x, 1:x} for x in np.linspace(0.0,1,10)],
                #  'clf__max_iter': [100,200,300,400,500,600,700,800,900,1000,1100],
                 'clf__C': np.linspace(0.0001,2,100),
                 'clf__l1_ratio': np.linspace(0,1,100),
            }
        },
        'Perceptron': {
            'pipeline': Pipeline([
                ('clf', Perceptron(random_state=42))
            ]),
            'params': {
                'clf__class_weight': [{0:1.0, 1:1.0}]+['balanced']+[{0:1.0-x, 1:x} for x in np.linspace(0.0,1,10)],
                'clf__penalty': ['l1', 'l2', 'elasticnet'],
                'clf__alpha': np.linspace(0,1,100),
                'clf__l1_ratio': np.linspace(0,1,100),
                'clf__eta0': np.linspace(0.0001, 1, 100),
                'clf__max_iter': [100,200,300,400,500,600,700,800,900,1000,1100],
            }
        }
    }

    from sklearn.model_selection import StratifiedKFold
    Skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    figures = []  # List to store individual figures
    t=[]
    for name, config in models.items():
        pipeline = config['pipeline']
        params = config['params']
        
        for param_name, param_range in params.items():
            pipeline = config['pipeline']
            train_scores, test_scores = validation_curve(
                pipeline, X_train, y_train, param_name=param_name, param_range=param_range,
                cv=Skfold, scoring="recall_macro", n_jobs=-1
            )
            pipeline = config['pipeline']
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)
            highest_test_score = np.max(test_mean)
            #get param value that corresponds to the highest test score
            highest_test_score_index = np.argmax(test_mean)
            highest_test_score_value = param_range[highest_test_score_index]
            t.append(f"{name} - {param_name}: {highest_test_score_value} - {highest_test_score:.4f}")
            #get param value that corresponds to the highest test score
            highest_test_score_index = np.argmax(test_mean)
            highest_test_score_value = param_range[highest_test_score_index]
            pipeline.set_params(**{param_name: highest_test_score_value})
            pipeline.fit(X_train,y_train)
            y_pred = pipeline.predict(X_test)
            from sklearn.metrics import recall_score
            recall = recall_score(y_test, y_pred, average='macro')
            print(f"{name} - {param_name}: {highest_test_score_value} - {highest_test_score:.4f} - {recall:.4f}")
            print(f"{name} - {param_name}: {highest_test_score_value} - {highest_test_score:.4f}")
            
            fig,ax=plt.subplots(figsize=(8, 6))
            if param_name == 'clf__penalty' or param_name == 'clf__solver':
                #bar chart
                ax.bar(param_range, test_mean, alpha=0.5, color='blue')
                ax.set_title(f"{name} - {param_name.replace('clf__', '').replace('_', ' ').title()}")
                ax.set_xlabel(param_name.replace('clf__', '').replace('_', ' ').title())
                ax.set_ylabel("Recall")
                ax.set_ylim(min(min(test_mean),min(train_mean))-0.08,max(max(test_mean),max(train_mean))+0.03)
                ax.grid(True)
                ax.annotate(f"Highest Test Score: {highest_test_score:.4f}, value: {highest_test_score_value}", xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top', color="red")
                figures.append(fig)  # Add the figure to the list
                continue
            if param_name == 'clf__class_weight':
                temp = []
                for pw in param_range:
                    if pw == 'balanced':
                        temp.append(str(pw))
                    else:
                        # Format as fraction
                        temp.append(r'$\frac{{{0:.2f}}}{{{1:.2f}}}$'.format(pw[0], pw[1]))
                param_labels = temp
                ax.set_title(f"{name} - Class Weight")
                ax.set_xlabel('Class Weight Class0,Class1')
                try:
                    param_value = f"0:{highest_test_score_value[0]:.2f} , 1:{highest_test_score_value[1]:.2f}"
                except:
                    param_value = str(highest_test_score_value)
            else:
                param_labels = param_range
                ax.set_title(f"{name} - {param_name.replace('clf__', '').replace('_', ' ').title()}")
                ax.set_xlabel(param_name.replace('clf__', '').replace('_', ' ').title())
                try:
                    #3 significant digits
                    if highest_test_score_value.round(3) == 0.000:
                        param_value = f'{highest_test_score_value:.5f}'
                    else:
                        param_value = f"{highest_test_score_value.round(3)}"

                except:
                    param_value = str(highest_test_score_value)
            ax.set_ylabel("Recall")
            ax.set_ylim(min(min(test_mean),min(train_mean))-0.1,max(max(test_mean),max(train_mean))+0.1)
            lw = 2
            ax.plot(param_labels, train_mean, label="Training score", color="darkorange", lw=lw)
            ax.fill_between(param_labels, train_mean - train_std, train_mean + train_std, alpha=0.2, color="darkorange", lw=lw)
            ax.plot(param_labels, test_mean, label="Cross-validation score", color="navy", lw=lw)
            ax.fill_between(param_labels, test_mean - test_std, test_mean + test_std, alpha=0.2, color="navy", lw=lw)
            ax.legend(loc="best")
            ax.grid(True)
            ax.annotate(f"Highest Validation Score: {highest_test_score:.4f}, value: {param_value}", xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top', color="red")            
            figures.append(fig)  # Add the figure to the list
            #plot confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            fig,ax=plt.subplots(figsize=(8, 6))
            import seaborn as sns
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f"{name} - {param_name.replace('clf__', '').replace('_', ' ').title()}({param_value}) - Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            figures.append(fig)

            
            

    # Save all figures to a single PDF, one plot per row (page)
    plots_to_pdf.to_pdf(figures, filename=filename)
    print(t)
    #ScrollableWindow(fig) 
    
    

#evaluate_parameters(filename)

