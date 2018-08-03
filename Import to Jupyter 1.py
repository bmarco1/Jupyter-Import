
#Fetching the Data
#Creates a dataset directory in the workspace
import os
import tarfile
from six.moves import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

#Break---


#Loading the Data
#Returns a Pandas dataframe object

import pandas
def load_housing_data(hosuing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

#Break---

#Histogram plot
#Shows the number of instances that have a given range

%matplot inline
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()

#Break---

#Creating a test set
#Typically set aside 20% of the dataset

import numpy as np
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data)) #random number generator
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
#Use the function
>>>train_set, test_set = split_train_test(housing, 0.2)
>>>print (len(train_set), "train +", len(test_set), "test")
16512 train + 4128 test

#hash table, save the test set on the first run to be used in subsequent Runs

import hashlib

def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 *test_ratio

def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]
#Add an Index columsn, this will act as a unique identifier
housing_with_id =housing.reset_index()
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
#May also use a stable feature as the unique identifier, such as longitude/latitude
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(hosuing_with_id, 0.2, "id")

#scikit-learn provides functions to split a dataset
from sklearn.model_selection import train_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

#Break---

#Create an income category attribute
#This code divides the median income by 1.5 to limit the number of income categories

hosuing["income_cat"] = np.ceil(housing["median_income"]/ 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

#Break---

#Begin Stratified Sampling

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
#Use the function
>>>strat_test_set["income_cat"].value_counts() / len(strat_test_set)

#Break---

#Because the sample income distribution is identical to the full dataset, we cane remove the income_cat to restore the dataset to it's original state

for set_in (strat_train_set, strat_test_set):
    set_.drop ("income_cat", axis=1, inplace=True)

#Break---

#Visualixe the Data
#Create a copy of the training set, so we don't screw it all up :)

housing = strat_train_set.copy()

#Create a geographical scatter plot with long/lat. Use an alpha = 0.1 to make the highly dense population areas more noticeable

housing.plot(kind="scatter", x="longitude", y="latitiude", alpha=0.1)

#Investigate housing prices

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=housing["population"] / 100, label="population", figsize=(10,7), c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,)
plt.legend()

#Break---

#Correlation Coefficient

corr_matrix = housing.corr()

#Break---

# Another way to check for correlations

from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))

#Zoom in on the most important correlation from above

housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)

>>>corr_matrix = housing.corr()
>>>corr_matrix["median_house_value"].sort_values(ascending=False)

#Break---

#Attribute Combinations

housing["rooms_per_housholds"] = housing["total_rooms"/housing"households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"/housing"total_rooms"]
housing["population_per_housholds"] = housing["population"/housing"households"]

#Get back to a clean set of data before Data Cleaning

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
housing["total_bedrooms"].fillna.(median, inplace=True)    #????????? Out of Place

#Begin Data Cleaning, get rid of corresponding districts, get rid of whole attributes, set values
    #Option 1: housing.dropna(subset=["total_bedrooms"])
    #Option 2: housing.drop("total_bedrooms", axis=1)
    #Option 3: median=housing ["total_bedrooms"].median()
    # housing["total_bedrooms"].fillna(median, inplace = True)

#Use Sci-scikit to replace missing values in the test set, Option 3

from sklearn.preprocessing import Imputer
imputer = Imputer(strategy="median")                      #the imputer computes the median of each attribute, also known as an estimator
housing_num = housing.drop("ocean_proximity", axis=1)     #We need to drop ocean_proximity since the median can only be computed in numerical attributes
imputer.fit(housing_num)

x = imputer.transform(housing_num)                        #Use the "trained" imputer to transform the training set. Replacing the missing values
housing_tr = pd.DataFrame(X, colums=housing_num.columns)        #to return to Pandas DataFrame, Optional

#Break---

#to change ocean_proximity into a numeric value if we didn't use the Imputer

>>>housing_cat = housing["ocean_proximity"]
>>>housing_cat.head(10)

#Convert all tect categories to numbers using Pandas factorize()

>>>housing_cat_encoded, housing_categories = housing_cat.factorize()
>>>housing_cat_encoded[:10]
>>>housing_categories

#Break---

#Converet interger categorical values into one-hot vectors. Only one attribute will be 1 or 0, hot or cold

>>>from sklearn.preprocessing import OneHotEncoder     #For CategoricalEncoder p.65
>>>encoder = OneHotEncoder()
>>>housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
>>>housing_cat_1hot
>>>housing_cat_1hot.toarray()

>>>cat_encoder.categories_

#Break---

#Custom transformers
#Create the following: fit(), transform(), or fit_transform()
#Combines previously used attributes

from sklearn.base import BaseEstimator, TransformerMixin            #BaseEstimator allows for 2 extra methods (get_params(), and set_params())

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True):               #add_bedrooms_per_room acts as the hyperparameter, thus allowing us to determine if this attribute helps the machine learning algorithm
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, y=None):
        return self
    def transform(self, x, y=None):
        rooms_per_houshold = x[:, rooms_ix] / x[:, household_ix]
        population_per_houshold = x[:, population_ix] / x[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = x[:, bedrooms_ix] / x[:, rooms_ix]
            return np.c_[x, rooms_per_houshold, population_per_houshold, bedrooms_per_room]
        else:
            return np.c_[x, rooms_per_houshold, population_per_houshold]

attr_addr = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_addr.transform(housing.values)

#Break---

#Create a  pipeline for numerical attributes
#Pipeline constrictor takes a list of name/estimator pairs to define a sequence of steps. All but the last estimator must be transformers

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()) #transformer
])

housing_num_tr = num_pipeline.fit-transform(housing_num)

#Break---

#Custom tranormer that allows us to feed the PandasDataFrame containing non-numerical colums directly into the Pipeline

from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):       #selecs the desired attributes, dropping the rest, and converts the resulting dataframe to a NumPy arrray
    def __init__(self, attrbute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

#Break---

#Create a pipeline for catergorical attributes

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
])

cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('cat_encoder', CategoricalEncoder(encoding="onehot-dense")),
])

#Full pipeline, joining the two scripted above using FeatureUnion

from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline)
        ("cat_pipeline", cat_pipeline)
])

>>>housing_prepared = full_pipeline.fit_transform(housing)
>>>housing_prepared

#Break---

from sklearn.linear_model import Linear_Regression

lin_reg = Linear_Regression()
lin_reg.fit(housing_prepared, housing_labels)



>>>some_data = housing.iloc[:5]
>>>some_labels = housing_labels.iloc[:5]
>>>some_data_prepared = full_pipeline.transform(some_data)
>>>print("labels", list(some_labels))


>>>from sklearn.metrics import mean_squared_error
>>>housing_predictions = lin_reg.predict(housing_prepared)
>>>lin_mse = mean_squared_error(housing_labels, housing_predictions)
>>>lin_rmse = np.sqrt(lin_mse)
>>>lin_rmse   #prediction error

#Decission Tree Regressor

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

>>>housing_predictions = tree_reg.predict(housing_prepared)
>>>tree_mse = mean_squared_error(housing_labels, housing_predictions)
>>>tree_rmse = np.sqrt(tree_mse)
>>>tree_rmse


#K-Fold Cross Validation

from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

>>>def display_scores(scores):
        print("Scores:", scores)
        print("Mean:", mean())
        print("Standard Deviation:", scores.std())
>>>display_scores(tree_rmse_scores)


#Linear Regrssion Model

>>>lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
>>>lin_rmse_scores  = np.sqrt(-lin_scores)
>>>display_scores(lin_rmse_scores)


#RandomForestRegressor

>>>from sklearn.ensemble import RandomForestRegressor
>>>forest_reg = RandomForestRegressor()
>>>forest_reg.fit(housing_prepared, housing_labels)
>>>[...]
>>>forest_rmse
>>>display_scores(forest_rmse_scores)



#JobLib

from sklearn.externals import job.joblib

joblib.dump(my_model, "My_Model.pkl")

#Later use
my_model_loaded = joblib.load("my_model.pkl")

#Grid Search

from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features'; [2, 3, 4]},

]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(housing_preparedn housing_labels)

>>>feature_importances = grid_search.best_estimator_.feature_importance_
>>>feature_importances


>>>extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
>>>cat_encoder = cat_pipeline.named_steps["cat_encoder"]
>>>cat_one_hot_attribs = list(cat_encoder.categories_[0])
>>>attributes = num_attribs + extra_attribs + cat_one_hot_attribs
>>>sorted(zip(feature_importances, attributes), reverse=True)

final_model = grid_search.best_estimator_

x_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

x_test_prepared = full_pipeline.transform(x_test)

final_predictions = final_model.predict(x_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

#MNIST dataset

>>>from sklearn.datasets import fetch_mldata
>>>mnist = fetch_mldata('MNIST original')
>>>mnist

>>>X, y =mnist["data"], mnist["target"]
>>>X.shape
>>>y.shape

%matplot inline
import matplotlib
import matplotlib.pyplot as plt

some_digit = x[36000]
some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()

>>>y[36000]


import numpy as np

shuffle_index = np.random.permutation(60000)
x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]

#Binary Classifier

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

#Stochastic Gradient Descent

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(x_train, y_train_5)

>>>sgd_clf.predict([some_digit])

#Cross Validation

from sklearn.model_selection import StratfiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits =3, random_state = 42)

for train_index, test_index in skfolds.split(x_train, y_train_5):
    clone_clf = clone(sgd_clf)
    x_train_folds = x_train[train_index]
    y_train_folds = y_train_5[train_index]
    x_test_fold = x_train[test_index]
    y_test_fold = y_train_5[test_index]

    clone_clf.fit(x_train_folds, y_train_folds)
    y_pred = clone_clf.predict(x_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))


>>>from sklearn.model_selection import cross_val_score
>>>cross_val_score(sgd.clf, x_train, y_train_5, cv=3, scoring="accuracy")


from sklearn.Base import BaseEstimator

class Never5Classifier(BaseEstimator):
    def fit(self, x, y=None):
        pass
    def predict(self, x):
        return np.zeros((len(x), 1), dtype=bool)

>>>never_5_clf = Never5Classifier()0000
>>>cross_val_score(never_5_clf, x_train, y_train_5, cv=3, scoring="accuracy")

#Confusion Matrix

from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd.clf, x_train, y_train_5, cv=3)

>>>from sklearn.metrics import confusion_matrix
>>>confusion_matrix(y_train_5, y_train_pred)

>>>confusion_matrix(y_train_5, y_train_perfect_predictions)             #See 87 for TP/FP Formulas

>>>from sklearn.metrics import precision_score, recall_score
>>>precision_score(y_train_5, y_train_pred)

#F1-Scores

>>>from sklearn.metrics import f1_score
>>>f1_score(y_train_5, y_train_pred)

>>>y_score = sgd_clf.decision_function([some_digit])
>>>y_scores
>>>threshold = 0
>>>y_some_digit_pred = (y_scores > threshold)


#ROC Curve
from sklearn.metrics import roc_curve(fpr, tpr, label=None)

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
#FRP againdt the tpr
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0. 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

plot_roc_curve(fpr, tpr)
plt.show()

#ROC accuracy

>>>from sklearn.metrics import roc_auc_score
>>>roc_auc_score(y_train_5, y_scores)
