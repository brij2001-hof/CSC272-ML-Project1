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
df = pd.read_csv('weather.csv')
df = df.dropna()
#print(df.head())
df.columns = df.columns.str.replace("'", "") #there were apostrophes in the column names
df.drop(columns=['Date','Location'], inplace=True)
#encode categorical data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
#balance the dataset
#df = df.groupby('RainTomorrow').apply(lambda x: x.sample(n=df['RainTomorrow'].value_counts().min(), random_state=42))
#all categorical columns
cat_cols = [col for col in df.columns if df[col].dtype == 'object']

#apply label encoder to categorical columns
for col in cat_cols:
        df[col] = le.fit_transform(df[col])
#corr heatmap
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the correlation matrix
corr = df.corr()

# Create a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
#plt.show()

#split the data into train test split
from sklearn.model_selection import train_test_split
X = df.drop('RainTomorrow', axis=1)
y = df['RainTomorrow']
print(y.value_counts())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,stratify=y)
print(y_train.value_counts())
# exit()


# Scaling
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
        from sklearn.metrics import f1_score
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f"{name} F1 Score: {f1:.4f}")
        
        # learning curve
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_train, y_train, cv=5, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 100))
        
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
    from sklearn.model_selection import KFold, validation_curve
    from sklearn.pipeline import Pipeline
    import numpy as np
    import matplotlib.pyplot as plt
    import plots_to_pdf

    models = {
        'Decision Tree': {
            'pipeline': Pipeline([
                ('clf', DecisionTreeClassifier())
            ]),
            'params': {
                'clf__max_depth': range(1, 20),
                # 'clf__min_samples_split': range(2, 11),
                # 'clf__min_samples_leaf': range(1, 11),
                'clf__ccp_alpha': np.linspace(0, 0.02, 20)
            }
        },
        'K-Nearest Neighbors': {
            'pipeline': Pipeline([
                ('clf', KNeighborsClassifier())
            ]),
            'params': {
                'clf__n_neighbors': range(10, 30),
                # 'clf__weights': ['uniform', 'distance'],
                # 'clf__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                'clf__p': [1,2]
            }
        },
        'Logistic Regression': {
            'pipeline': Pipeline([
                ('clf', LogisticRegression(verbose=2))
            ]),
            'params': {
                 'clf__class_weight': ['balanced']+[{0:1.0-x, 1:x} for x in np.linspace(0.0,1,10)],
                 'clf__max_iter': [30,40,50,60,70,80,90,100,200,300,400,500],
                 'clf__solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
                 'clf__C': np.linspace(0.0001,2,10),
            }
        },
        'Perceptron': {
            'pipeline': Pipeline([
                ('clf', Perceptron(max_iter=1000,verbose=2,random_state=42,penalty='elasticnet'))
            ]),
            'params': {
                'clf__class_weight': ['balanced']+[{0:1.0-x, 1:x} for x in np.linspace(0.0,1,10)],
                'clf__n_iter_no_change': [1,2,3,4,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100],
                'clf__l1_ratio': np.linspace(0,0.3,100),
                # 'clf__tol' : np.linspace(0.0001,0.01,100),
                # 'clf__alpha' : np.linspace(0.00001,0.01,100),
                # 'clf__eta0' : np.linspace(0.00001,1,100),
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
            train_scores, test_scores = validation_curve(
                pipeline, X_train, y_train, param_name=param_name, param_range=param_range,
                cv=Skfold, scoring="f1_weighted", n_jobs=-1
            )
            
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
            from sklearn.metrics import f1_score
            f1 = f1_score(y_test, y_pred, average='weighted')
            print(f"{name} - {param_name}: {highest_test_score_value} - {highest_test_score:.4f} - {f1:.4f}")
            print(f"{name} - {param_name}: {highest_test_score_value} - {highest_test_score:.4f}")
            
            fig,ax=plt.subplots(figsize=(10, 6))
            if param_name == 'clf__penalty' or param_name == 'clf__solver':
                #bar chart
                ax.bar(param_range, test_mean, alpha=0.5, color='blue')
                ax.set_title(f"{name} - {param_name.replace('clf__', '').replace('_', ' ').title()}")
                ax.set_xlabel(param_name.replace('clf__', '').replace('_', ' ').title())
                ax.set_ylabel("F1 Weighted Score")
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
                    highest_test_score_value = f"0:{highest_test_score_value[0]:.2f} , 1:{highest_test_score_value[1]:.2f}"
                except:
                    highest_test_score_value = str(highest_test_score_value)
            else:
                param_labels = param_range
                ax.set_title(f"{name} - {param_name.replace('clf__', '').replace('_', ' ').title()}")
                ax.set_xlabel(param_name.replace('clf__', '').replace('_', ' ').title())
                try:
                    #3 significant digits
                    if highest_test_score_value.round(3) == 0.000:
                        highest_test_score_value = f'{highest_test_score_value:.5f}'
                    else:
                        highest_test_score_value = f"{highest_test_score_value.round(3)}"

                except:
                    highest_test_score_value = str(highest_test_score_value)
            ax.set_ylabel("F1 Weighted Score")
            ax.set_ylim(min(min(test_mean),min(train_mean))-0.1,max(max(test_mean),max(train_mean))+0.1)
            lw = 2
            ax.plot(param_labels, train_mean, label="Training score", color="darkorange", lw=lw)
            ax.fill_between(param_labels, train_mean - train_std, train_mean + train_std, alpha=0.2, color="darkorange", lw=lw)
            ax.plot(param_labels, test_mean, label="Cross-validation score", color="navy", lw=lw)
            ax.fill_between(param_labels, test_mean - test_std, test_mean + test_std, alpha=0.2, color="navy", lw=lw)
            ax.legend(loc="best")
            ax.grid(True)
            ax.annotate(f"Highest Test Score: {highest_test_score:.4f}, value: {highest_test_score_value}", xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top', color="red")            
            figures.append(fig)  # Add the figure to the list
            #plot confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            fig,ax=plt.subplots(figsize=(10, 6))
            import seaborn as sns
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f"{name} - Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            figures.append(fig)
            from scikitplot.classifiers import plot_precision_recall_curve_with_cv
            fig,ax=plt.subplots(figsize=(10, 6))
            try:
                plot_precision_recall_curve_with_cv(do_cv=False,clf=pipeline, X=X_test, y=y_test, ax=ax)
                figures.append(fig)            
            except:
                print(f"Error in {name} - Precision-Recall Curve")
            plt.show()
            
    # Save all figures to a single PDF, one plot per row (page)
    plots_to_pdf.to_pdf(figures, filename='heart.pdf')
    print(t)
    #plt.show()

evaluate_parameters(X_train, X_test, y_train, y_test)
#train_and_evaluate_models(X_train, X_test, y_train, y_test)

