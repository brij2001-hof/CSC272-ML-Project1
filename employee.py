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
df = pd.read_csv('Employee.csv')
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

X = df.drop('LeaveOrNot', axis=1)
scaler = StandardScaler()
#X = scaler.fit_transform(X)
#X_test = scaler.transform(X)
y = df['LeaveOrNot']
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# from sklearn.model_selection import KFold
# scores=[]
# kFold=KFold(n_splits=10,random_state=42,shuffle=True)
# for i,(train_index,test_index) in enumerate(kFold.split(X,y)):
#     print("Train Index: ", train_index, "\n")
#     print("Test Index: ", test_index)
    
#     X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
#     knn.fit(X_train,y_train)
#     scores.append(knn.score(X_test, y_test))

# print(scores)
# exit

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
#one hot encoding
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
b = X_train
X_train = pd.get_dummies(X_train,columns=cat_cols)
print(X_train.info())
print(b.head())
X_test = pd.get_dummies(X_test,columns=cat_cols)

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
                cv=kfold, scoring="f1_weighted", n_jobs=-1
            )
            
            train_mean = np.mean(train_scores, axis=1)
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
            ax.set_ylabel("F1 weighted")
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

    #plt.tight_layout()
   # plt.show()
    ScrollableWindow(fig)
for i in np.logspace(-4, 0, 20):
    print(i)
#train_and_evaluate_models(X_train, X_test, y_train, y_test)
#train_and_evaluate_models(X_train, X_test, y_train, y_test)
# Get feature importances
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
importances = rf_classifier.feature_importances_
feature_importances = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)

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

