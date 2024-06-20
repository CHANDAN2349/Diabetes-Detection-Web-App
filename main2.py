import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import plotly.express as px
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots



st.set_page_config(layout="wide")

loaded_model=pickle.load(open('random_forest_model.sav','rb'))

# Function to check DataFrame properties
def check(df):
    l=[]
    columns=df.columns
    for col in columns:
        dtypes=df[col].dtypes
        nunique=df[col].nunique()
        duplicated=df.duplicated().sum()
        sum_null=df[col].isnull().sum()
        l.append([col,dtypes,nunique,duplicated,sum_null])
    df_check=pd.DataFrame(l)
    df_check.columns=['columns','Data Types','No of Unique Values','No of Duplicated Rows','No of Null Values']
    return df_check 


# Specify the folder path where your data resides
data_folder_path = "Data"

# Load data directly from the folder
for file_name in os.listdir(Path(data_folder_path)):
    if file_name.endswith(".csv"):
        data_file_path = os.path.join(data_folder_path, file_name)
        df = pd.read_csv(data_file_path)
        break  # Load the first CSV file found in the folder
    
    
def purchase_pred(input_data):
    input_data_as_array=np.asarray(input_data,dtype=float)
    input_data_reshaped=input_data_as_array.reshape(1,-1)
    prediction=loaded_model.predict(input_data_reshaped)
    print(prediction)
    
    if prediction[0] == 0:
        return "The person is not diabetic"
    else:
        return "The person is diabetic"





# Streamlit sidebar
with st.sidebar:
    st.image("https://d18yrmqbzi0q7a.cloudfront.net/wp-content/uploads/diabetes-symptoms-and-treatment.jpg")
    st.title("Main Menu")
    choice = st.radio("Navigation", ["EDA", "Plots", "ML"])
    st.info("Explore data, visualize insights, and predict diabetes risk  with user-friendly web app. Uncover trends, interpret predictive models.")
    


if choice == "EDA":
    st.title("Diabetes Risk Analysis")
    st.markdown("Diabetes risk prediction involves using machine learning models to analyze various health metrics and demographic information, such as glucose levels, BMI, age, and family history, to assess the likelihood of an individual developing diabetes. By training models on historical data and applying them to new cases, these predictions can help individuals and healthcare providers proactively manage and prevent diabetes-related complications.")
    st.markdown(
    "<div style='text-align:center'><img src='https://medeor.ae/wp-content/uploads/2022/11/Diabetes-Facts.jpg' width='500'></div>",
    unsafe_allow_html=True)
    st.divider()
    st.subheader("Exploratory Data Analysis")
    if st.checkbox("Display Data"):
        st.dataframe(df)
    
    if st.checkbox("Show Shape"):
        shape = f"There are {df.shape[0]} columns and {df.shape[1]} rows in the datset."
        st.write(shape)
    
    if st.checkbox("Check Data"):
        df_check = check(df)
        st.dataframe(df_check)
        
    if st.checkbox("Describe Statistics of Numerical Data"):
        des1 = df.describe().T
        st.dataframe(des1)
    
    if st.checkbox("Describe Categorical Columns"):
        categorical_df = df.select_dtypes(include=['object'])
        des2 = categorical_df.describe()
        st.dataframe(des2)
        
    if st.checkbox("Show columns"):
        info1=df.columns
        st.dataframe(info1)
    

if choice == "Plots":
    st.title("Data Visualization")

    
    # Plot 1: Scatter Plot of Glucose vs. BMI
    fig1 = px.scatter(df, x='Glucose', y='BMI', color='Outcome', title='Glucose vs. BMI')
    st.plotly_chart(fig1)
    
    # Plot 2: Histogram of Age
    fig2 = px.histogram(df, x='Age', title='Age Distribution')
    st.plotly_chart(fig2)
    
    # Plot 3: Box Plot of Blood Pressure
    fig3 = px.box(df, y='BloodPressure', title='Blood Pressure Distribution')
    st.plotly_chart(fig3)


    st.subheader("Pie Chart Representing the Distribution of Age Groups")
    age_bins = [20, 30, 40, 50, 60, 70, 80, 90]
    age_labels = ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89']
    df['AgeGroup'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)
    age_counts = df['AgeGroup'].value_counts().reset_index()
    age_counts.columns = ['AgeGroup', 'Count']
    fig4 = px.pie(age_counts, names='AgeGroup', values='Count', hole=0.4)
    st.plotly_chart(fig4)

    
    
if choice == "ML":
    def main():
        st.title("Predicting Diabetes Risk Web App")

        # Getting the input data
        pregnancies = st.text_input("Pregnancies")
        glucose = st.text_input("Glucose")
        blood_pressure = st.text_input("Blood Pressure")
        skin_thickness = st.text_input("Skin Thickness")
        insulin = st.text_input("Insulin")
        bmi = st.text_input("BMI")
        diabetes_pedigree_function = st.text_input("Diabetes Pedigree Function")
        age = st.text_input("Age")

        # Code for prediction
        result = ''
        if st.button("Predict"):
            input_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]
            result = purchase_pred(input_data)

        st.success(result)


    if __name__ == '__main__':
        main()
