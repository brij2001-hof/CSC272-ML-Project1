import churn
import GRID_churn
import heart
import grid_heart
import datetime


heart_HyperTuning_filename = datetime.datetime.now().strftime("heart_HyperTuning_%Y-%m-%d_%H-%M-%S.pdf")
churn_HyperTuning_filename = datetime.datetime.now().strftime("churn_HyperTuning_%Y-%m-%d_%H-%M-%S.pdf")

heart.evaluate_parameters(filename=heart_HyperTuning_filename)
churn.evaluate_parameters(filename=churn_HyperTuning_filename)



heart_learning_curves_filename = datetime.datetime.now().strftime("heart_learning_curves_%Y-%m-%d_%H-%M-%S.pdf")
churn_learning_curves_filename = datetime.datetime.now().strftime("churn_learning_curves_%Y-%m-%d_%H-%M-%S.pdf")

grid_heart.gridsearch_learning_curves(filename=heart_learning_curves_filename) # 
GRID_churn.gridsearch_learning_curves(filename=churn_learning_curves_filename)

