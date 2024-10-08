import churn
import GRID_churn
import heart
import grid_heart
import datetime


heart_HyperTuning_filename = datetime.datetime.now().strftime("heart_HyperTuning_%Y-%m-%d_%H-%M-%S.pdf")
churn_HyperTuning_filename = datetime.datetime.now().strftime("churn_HyperTuning_%Y-%m-%d_%H-%M-%S.pdf")

'''
Evaluating hyperparameters for each model using validation curve.
Outputs (for each dataset):
        - .pdf | Plots containing validation curves for each hyper parameter tested for each model. Also confusion
'''
heart.evaluate_parameters(filename=heart_HyperTuning_filename)
churn.evaluate_parameters(filename=churn_HyperTuning_filename)



# heart_learning_curves_filename = datetime.datetime.now().strftime("heart_learning_curves_%Y-%m-%d_%H-%M-%S.pdf")
# churn_learning_curves_filename = datetime.datetime.now().strftime("churn_learning_curves_%Y-%m-%d_%H-%M-%S.pdf")



'''
Calculates best params using grid search.
Fits on train set, predicts on test set and,
Outputs (for each dataset):
        - .pdf | Plots containing learning curves, confusion matrix and precision-recall AOC(*only for heart dataset)
        - .txt | Parameters returned by grid search for each model and their cross-validation scores
        - .txt | Test set scores, params and time to train for each model


'''
grid_heart.gridsearch_learning_curves(filename=heart_learning_curves_filename) 
GRID_churn.gridsearch_learning_curves(filename=churn_learning_curves_filename)

