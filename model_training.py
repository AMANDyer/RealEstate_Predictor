import pandas as pd
import numpy as np
import xgboost
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,accuracy_score,precision_score,recall_score,f1_score
from xgboost import XGBClassifier,XGBRegressor
from sklearn.metrics import confusion_matrix, classification_report
import joblib


#importing dataframe
df=pd.read_csv(r"C:\Users\amand\real_estate_ML\estates\indian_real_estate.csv")
df.head()

#shape 64 rows,44 columns
print(f"shape of dataframe is {df.shape}")

#columns
print(f"information of columns  {df.info()}")

#to handle mixed feature floor
def extract_floor_info(value):
    try:
        parts = value.lower().split('out of')
        current = parts[0].strip()
        total = int(parts[1].strip())

        # Convert 'ground' to 0, else keep numeric
        if 'ground' in current:
            current_floor = 0
        else:
            current_floor = int(current)
        return pd.Series([current_floor, total])
    except:
        return pd.Series([None, None])  # Handle missing or malformed entries

df[['Current_Floor', 'Total_Floors']] = df['Floor'].apply(extract_floor_info)

print(f"missing values per column {df.isnull().sum()}")
# there is  missing values in current_floor,total_floor

si=SimpleImputer(strategy='mean')
df['Current_Floor']=si.fit_transform(df[['Current_Floor']])
df['Total_Floors']=si.fit_transform(df[['Total_Floors']])

#unnecessary features removal
df.drop(columns=['Posted On','Floor','Rent','property_id','description','feels_like','Log_Rent','City_Encoded','Is_Furnished'],inplace=True)

# creating a list of target columns of all models(investment_price_prediction,investment_label_prediction,rental_income_prediction)
target_columns=['Monthly_Rental_Income','Est_Property_Value','Investment_Label']


#separate features and targets
new_df=df.copy()
x=new_df.drop(columns=[*target_columns,'Rent_Yield_Estimate'])
y=df[target_columns]
y['Investment_Label'].value_counts()
x.columns

def feature_eng(x):
    #Numeric and continuous features
    numeric_f=x.select_dtypes(include=['int64','float64']).columns.tolist()
    categorical_f=x.select_dtypes(include=['object']).columns.tolist()

    #pipeline for numeric and categorical transformation
    num_trans=Pipeline(steps=[
        ("Scaling",StandardScaler())
    ])
    cat_trans=Pipeline(steps=[
        ("Encode",OneHotEncoder(drop='first',sparse_output=False,handle_unknown='ignore'))
    ])

    #performing feature eng.
    preprocessor=ColumnTransformer(transformers=[
        ('trns1',num_trans,numeric_f),
        ('trns2',cat_trans,categorical_f)
        ],remainder='passthrough')
    
    return preprocessor


#for Investment Price Prediction Model
x1=df[[*x.columns,'Monthly_Rental_Income']]
y1=y['Est_Property_Value']

x1.head()
#target column is highly right skewed
y1.skew()
y1_log=np.log1p(y1)
y1_log.skew()

x_train,x_test,y_train,y_test=train_test_split(x1,y1_log,test_size=0.2,random_state=42)

#for feature eng.
preprocessor=feature_eng(x1)
parameters_regressor={
    'Regressor__n_estimators': [200,250,300],               # Number of boosting rounds
    'Regressor__learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],              # Shrinkage rate
    'Regressor__max_depth': [3, 5, 7, 9, 12],                             # Tree depth
    'Regressor__min_child_weight': [1, 3, 5, 7,10,20,22],                          # Minimum sum of instance weight in a child
    'Regressor__gamma': [0, 0.1, 0.3, 0.5, 1],                            # Minimum loss reduction to make a split
    'Regressor__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],                    # Row sampling per tree
    'Regressor__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],             # Feature sampling per tree
    'Regressor__reg_alpha': [0.5,1,2],                             # L1 regularization
    'Regressor__reg_lambda': [0.5, 1.0, 1.5, 2.0],                        # L2 regularization
    'Regressor__booster': ['gbtree', 'dart'],                             # Booster type
    'Regressor__random_state': [42]                                       # For reproducibility
}

clf=Pipeline(steps=[
    ('preprocessing',preprocessor), 
    ('Regressor',XGBRegressor(tree_method='hist',eval_metric='rmse'))
])

model1= RandomizedSearchCV(
    estimator=clf,
    param_distributions=parameters_regressor,
    n_iter=10,                      # Number of random combinations to try
    scoring='neg_root_mean_squared_error',  # Or 'r2', 'neg_mean_absolute_error'
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=1                     
)

model=model1.fit(x_train,y_train) 

train_preds = model.predict(x_train)
test_preds = model.predict(x_test)



train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
test_rmse = np.sqrt(mean_squared_error(y_test,test_preds))

print(f"Train RMSE: {train_rmse}")
print(f"Test RMSE: {test_rmse}")

r2_score(y_test,test_preds)

print(f"Best Parameters are {model.best_params_}")

test_rmse = mean_absolute_error(y_test,test_preds)

print(test_rmse) 

#model 2(Rental_Income_predictor)

x2=x
y2=y['Monthly_Rental_Income']


#it is overly skewed
y2.skew()
y2_log=np.log1p(y2)
y2_log.skew()


x_train,x_test,y_train,y_test=train_test_split(x2,y2_log,test_size=0.2,random_state=42)

#for feature eng.
preprocessor=feature_eng(x2)

clf2=Pipeline(steps=[
    ('preprocessing',preprocessor), 
    ('Regressor',XGBRegressor(tree_method='hist',eval_metric='rmse'))
])

model2= RandomizedSearchCV(
    estimator=clf2,
    param_distributions=parameters_regressor,
    n_iter=10,                      # Number of random combinations to try
    scoring='neg_root_mean_squared_error',  # Or 'r2', 'neg_mean_absolute_error'
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=1                     
)

model2=model2.fit(x_train,y_train)   

train_preds = model2.predict(x_train)
test_preds = model2.predict(x_test)



train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
test_rmse = np.sqrt(mean_squared_error(y_test,test_preds))

print(f"Train RMSE: {train_rmse}")
print(f"Test RMSE: {test_rmse}")

r2_score(y_test,test_preds)

print(f"Best Parameters are {model2.best_params_}")

test_rmse = mean_absolute_error(y_test,test_preds)

print(test_rmse)

#model 3 (Investment_Label Predictor)

x3=df[[*x.columns,'Rent_Yield_Estimate','Est_Property_Value']]
y3=y['Investment_Label']

y3.value_counts()

x3.info()

#check if data is imbalanced or not
print(y3.value_counts())

x_train,x_test,y_train,y_test=train_test_split(x3,y3,test_size=0.2,random_state=42)

#for feature eng.
preprocessor=feature_eng(x3)

parameters_classifier = {
    'Classifier__n_estimators': [200, 250, 300],             # Number of boosting rounds
    'Classifier__learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],# Shrinkage rate
    'Classifier__max_depth': [3, 5, 7, 9, 12],               # Tree depth
    'Classifier__min_child_weight': [1, 3, 5, 7, 10, 20, 22],# Minimum sum of instance weight in a child
    'Classifier__gamma': [0, 0.1, 0.3, 0.5, 1],              # Minimum loss reduction to make a split
    'Classifier__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],      # Row sampling per tree
    'Classifier__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],# Feature sampling per tree
    'Classifier__reg_alpha': [0.5, 1, 2],                    # L1 regularization
    'Classifier__reg_lambda': [0.5, 1.0, 1.5, 2.0],          # L2 regularization
    'Classifier__booster': ['gbtree', 'dart'],              # Booster type
    'Classifier__random_state': [42]                         # For reproducibility
}

clf3=Pipeline(steps=[
    ('preprocessing',preprocessor), 
    ('Classifier',XGBClassifier(tree_method='hist',eval_metric='logloss'))
])


model3 = RandomizedSearchCV(
    estimator=clf3,
    param_distributions=parameters_classifier,
    n_iter=10,                  # Number of random combinations to try
    scoring='accuracy',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=1
)

model3=model3.fit(x_train,y_train) 



y_pred_train = model3.predict(x_train)  
y_pred_test = model3.predict(x_test)

print(f"Best Parameters are {model3.best_params_}")

print("Train Accuracy:", accuracy_score(y_train, y_pred_train))
print("Test Accuracy:", accuracy_score(y_test, y_pred_test))

print("Train F1 Score:", f1_score(y_train, y_pred_train, average='weighted'))
print("Test F1 Score:", f1_score(y_test, y_pred_test, average='weighted'))


print(confusion_matrix(y_test, y_pred_test)) 
print(classification_report(y_test, y_pred_test))


#exporting trained models to joblib file
joblib.dump(model,"investment_Price.joblib")
joblib.dump(model2,"rental_income.joblib")
joblib.dump(model3,"investment_label.joblib")

