from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
import xlearn as xl


ffm_model = xl.create_ffm()  # Use field-aware factorization machine
ffm_model.setTrain("./small_train.txt")   # Training data
ffm_model.setValidate("./small_test.txt")  # Validation data

param = {'task':'binary', 'lr':0.2, 'lambda':0.002}

# Train model
ffm_model.fit(param, "./model.out")
