import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,roc_curve,auc
from sklearn.pipeline import Pipeline
from sklearn.model_selection import validation_curve
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class ScrollableWindow(QtWidgets.QMainWindow):
    def __init__(self, fig):
        self.qapp = QtWidgets.QApplication([])

        QtWidgets.QMainWindow.__init__(self)
        self.widget = QtWidgets.QWidget()
        self.setCentralWidget(self.widget)
        self.widget.setLayout(QtWidgets.QVBoxLayout())
        self.widget.layout().setContentsMargins(0,0,0,0)
        self.widget.layout().setSpacing(0)

        self.fig = fig
        self.canvas = FigureCanvas(self.fig)
        self.canvas.draw()
        self.scroll = QtWidgets.QScrollArea(self.widget)
        self.scroll.setWidget(self.canvas)

        self.nav = NavigationToolbar(self.canvas, self.widget)
        self.widget.layout().addWidget(self.nav)
        self.widget.layout().addWidget(self.scroll)

        self.show()
        exit(self.qapp.exec_())
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler

print("Running dataset1")


# Load the data
df = pd.read_csv('credit.csv')

df.columns = df.columns.str.replace("'", "") 
print(df.columns)
df.drop(['id'],axis=1,inplace=True)
df.dropna(inplace=True)
print(df.head())
#feature importance
# Feature importance
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

#all categorical columns
cat_cols = [col for col in df.columns if df[col].dtype == 'object']

#apply label encoder to categorical columns
# for col in cat_cols:
#     df[col] = le.fit_transform(df[col])
# Prepare the data
#drop duplicates

X = df.drop('class', axis=1)
print(X.columns)
scaler = StandardScaler()
#X = scaler.fit_transform(X)
#X_test = scaler.transform(X)
y = df['class']
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
# knn = KNeighborsClassifier()
X = pd.get_dummies(X,columns=[c for c in cat_cols if c != 'class'])
print(X.columns)
y = le.fit_transform(y)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
#one hot encoding
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
# b = X_train
#X_train = pd.get_dummies(X_train,columns=[c for c in cat_cols if c != 'class'])
X_train = scaler.fit_transform(X_train)
#print(X_train.info())
#print(b.head())
#X_test = pd.get_dummies(X_test,columns=[c for c in cat_cols if c != 'class'])
X_test = scaler.transform(X_test)

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'Decision Tree': DecisionTreeClassifier(max_depth=9),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=9, weights='uniform', algorithm='brute'),
        'Logistic Regression': LogisticRegression(C=1, solver='lbfgs', penalty='l2', max_iter=100, tol=0.0001),
        'Perceptron': Perceptron(alpha=0.001, max_iter=100, eta0=0.1, penalty='elasticnet', early_stopping=True)
    }
    
    plt.figure(figsize=(15, 15),layout='') #avoid overlapping
    for i, (name, model) in enumerate(models.items(), 1):
        # learning curve
        train_sizes, train_scores, test_scores = learning_curve(
             model, X_train, y_train)
        
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)

        #F1 score
        from sklearn.metrics import accuracy_score,classification_report
        f1 = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {f1:.4f}")
        print(classification_report(y_test, y_pred))

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
        plt.text(0.5, 0.5, f"{name} (tuned) F1 Score: {f1:.2f}", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        plt.legend(loc="best")
        plt.grid()
    plt.tight_layout(h_pad=0.5,w_pad=0.5)
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
                ('clf', DecisionTreeClassifier(random_state=42))
            ]),
            'params': {
                'clf__max_depth': range(1, 20),
                'clf__min_samples_split': range(2, 11),
                'clf__min_samples_leaf': range(1, 11),
                'clf__ccp_alpha': np.linspace(0.0001, 0.02, 20)
            }
        },
        'K-Nearest Neighbors': {
            'pipeline': Pipeline([
                ('clf', KNeighborsClassifier(n_neighbors=15))
            ]),
            'params': {
                'clf__n_neighbors': range(1, 20),
                'clf__weights': ['uniform', 'distance'],
                'clf__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
            }
        },
        'Logistic Regression': {
            'pipeline': Pipeline([
                ('clf', LogisticRegression(random_state=42,penalty='l1',solver='liblinear'))
            ]),
            'params': {
                 'clf__class_weight': [{0:1.0, 1:1.0}]+['balanced']+[{0:1.0-x, 1:x} for x in np.linspace(0.0,1,10)],
                 'clf__max_iter': [100,200,300,400,500,600,700,800,900,1000,1100],
                 'clf__C': np.linspace(0.0001,2,100),
                 'clf__solver': ['liblinear', 'saga'],
                 #'clf__penalty': ['l1', 'l2', 'elasticnet'],
            }
        },
        'Perceptron': {
            'pipeline': Pipeline([
                ('clf', Perceptron(random_state=42))
            ]),
            'params': {
                'clf__class_weight': [{0:1.0, 1:1.0}]+['balanced']+[{0:1.0-x, 1:x} for x in np.linspace(0.0,1,10)],
               # 'clf__penalty': ['l1', 'l2', 'elasticnet'],
                #'clf__l1_ratio': np.linspace(0,0.3,100),
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
            train_scores, test_scores = validation_curve(
                pipeline, X_train, y_train, param_name=param_name, param_range=param_range,
                cv=Skfold, scoring="recall_micro", n_jobs=-1
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
            from sklearn.metrics import recall_score
            recall = recall_score(y_test, y_pred, average='binary')
            print(f"{name} - {param_name}: {highest_test_score_value} - {highest_test_score:.4f} - {recall:.4f}")
            print(f"{name} - {param_name}: {highest_test_score_value} - {highest_test_score:.4f}")
            
            fig,ax=plt.subplots(figsize=(10, 6))
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
            ax.set_ylabel("Recall")
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
                if name =='Perceptron':
                    from sklearn.calibration import CalibratedClassifierCV
                    per = pipeline
                    clf_isotonic = CalibratedClassifierCV(per, cv=10, method='isotonic')
                    clf_isotonic.fit(X_train,y_train)
                    plot_precision_recall_curve_with_cv(do_cv=False,clf=clf_isotonic, X=X_test, y=y_test, ax=ax)
                    
                else:
                    plot_precision_recall_curve_with_cv(do_cv=False,clf=pipeline, X=X_test, y=y_test, ax=ax)
                figures.append(fig)            
            except:
                print(f"Error in {name} - Precision-Recall Curve")
            

    # Save all figures to a single PDF, one plot per row (page)
    plots_to_pdf.to_pdf(figures, filename='churn.pdf')
    print(t)
    #ScrollableWindow(fig) 
    
    
#train_and_evaluate_models(X_train, X_test, y_train, y_test)
#train_and_evaluate_models(X_train, X_test, y_train, y_test)
# Get feature importances
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
importances = rf_classifier.feature_importances_
#feature_importances = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)

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

# param_range = np.arange(10, 10000, 10)
# from sklearn.model_selection import validation_curve
# train_scores, test_scores = validation_curve(
#     Perceptron(), X_train, y_train, param_name='max_iter', param_range=param_range, cv=5, scoring='accuracy', n_jobs=-1
# )

# train_mean = np.mean(train_scores, axis=1)
# train_std = np.std(train_scores, axis=1)
# test_mean = np.mean(test_scores, axis=1)
# test_std = np.std(test_scores, axis=1)

# plt.figure()
# plt.title("Validation Curve with Perceptron (alpha)")
# plt.xlabel("alpha")
# plt.ylabel("Score")
# plt.ylim(0.0, 1.1)
# plt.semilogx(param_range, train_mean, label="Training score", color="darkorange", lw=2)
# plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.2, color="darkorange", lw=2)
# plt.semilogx(param_range, test_mean, label="Cross-validation score", color="navy", lw=2)
# plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.2, color="navy", lw=2)
# plt.legend(loc="best")
# plt.grid(True)
# plt.show()
evaluate_parameters(X_train, X_test, y_train, y_test)
#train_and_evaluate_models(X_train, X_test, y_train, y_test)
#train_and_evaluate_models(X_train, X_test, y_train, y_test)




