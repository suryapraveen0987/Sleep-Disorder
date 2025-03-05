import streamlit as st
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import pickle as pk



# loading the dataset
df = pd.read_csv("Sleep_health_dataset.csv")

X_train = pd.read_csv("X_train_to_preprocess.csv")
y_train = pd.read_csv("y_train_to_preprocess.csv") 




###########################################################################



st.title("Sleep Disorder Prediction")




## Gender -> selectbox()
Gender_options = X_train["Gender"].unique().tolist()
Gender = st.selectbox("Choose your Gender :-", Gender_options, index=None)

## Age -> slider()
Age = st.slider("Set your Age:-",1,100,25, format="%d")

## Occupation -> selectbox()
Occupation_options = X_train["Occupation"].unique().tolist()
Occupation = st.selectbox("Choose your Occupation :-", Occupation_options, index=None)

## Sleep Duration -> slider()
Sleep_Duration = st.slider("Set your Sleep Duration :-",1.0,10.0,5.0, step=0.1, format="%.1f")

## Physical Activity Level -> slider()
Physical_Activity_Level = st.slider("Set your Physical Activity Level :-",0,100,40, step=5, format="%d")

## Stress Level -> slider()
Stress_Level = st.slider("Set your Stress Level :-",1,10,5, step=1, format="%d")

## BMI Categoryy -> selectbox()
BMI_Category_options = X_train["BMI Category"].unique().tolist()
BMI_Category = st.selectbox("Choose your BMI-Category :-",BMI_Category_options, index=None)

## BP(Upper, Lower values) -> slider()
Upper_BP = st.slider("Set your upper-BP :-",10,200,120, step=1, format="%d")
Lower_BP = st.slider("Set your lower-BP :-",10,200,80, step=1, format="%d")

## Heart Rate -> slider()
Heart_Rate = st.slider("Set your Heart-Rate :-",1,100,72, step=1, format="%d")

## Daily steps -> slider()
Daily_Steps = st.slider("Set your Daily-Steps :-",100,10000,3000, step=100, format="%d")




###########################################################################








# Feature Scaling 

def feature_scaling(test_data):
    scaler = RobustScaler()
    num_cols = X_train.select_dtypes(exclude="object").columns.tolist()
    scaler.fit(X_train[num_cols])
    test_data[num_cols] = scaler.transform(test_data[num_cols])
    return test_data



# Feature Encoding 

def feature_encoding(test_data):
    ohe = OneHotEncoder(sparse_output=False, drop="first", dtype="int64", handle_unknown='ignore')
    cat_cols = X_train.select_dtypes(include="object").columns.tolist()
    ohe.fit(X_train[cat_cols])
    test_encoded = ohe.transform(test_data[cat_cols])
    test_encoded_df = pd.DataFrame(test_encoded, columns=ohe.get_feature_names_out(cat_cols)).astype("int64")
    test_data.drop(cat_cols, axis=1, inplace=True)
    test_data = pd.concat([test_encoded_df, test_data], axis=1)
    return test_data



# Predictions 
    
def predict(test_data):
    model_path = open("GB_model.pkl",'rb')
    model = pk.load(model_path)
    predicted = model.predict(test_data)
    encoder = LabelEncoder()
    encoder.fit(y_train)
    output = encoder.inverse_transform(predicted)
    return output
    


    


# all functionalities in one function 

def model_pipeline():
    data_to_predict = [str(Gender), np.int64(Age), str(Occupation), np.float64(Sleep_Duration), np.int64(Physical_Activity_Level), np.int64(Stress_Level), str(BMI_Category), np.int64(Upper_BP), np.int64(Lower_BP), np.int64(Heart_Rate), np.int64(Daily_Steps)]
    test = pd.DataFrame(data=np.array(data_to_predict).reshape(1,-1),
                        columns=X_train.columns.tolist())
    for col in X_train.columns:
        test[col] = test[col].astype(X_train.dtypes[col])
    scaled_test = feature_scaling(test)
    encoded_scaled_test = feature_encoding(scaled_test)
    final_test_data = encoded_scaled_test
    output = predict(final_test_data)
    return output


left, middle, right = st.columns(3)

## submit and predict button
if middle.button("Predict", use_container_width=True):

    model_output = model_pipeline()

    st.subheader("Prediction :-")
    
    if model_output=="Sleep Apnea":
        st.write('Person is having "Sleep Apnea"')
    elif model_output=="Insomnia":
        st.write('Person is having "Insomnia"')
    else :
        st.write("Person is Not having any Sleep Disorder")