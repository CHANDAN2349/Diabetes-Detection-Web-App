import streamlit as st
import pandas as pd
import plotly.express as px

def load_data():
    # Load your dataset or data here
    data =pd.read_csv(r"C:\Users\Chandan\Desktop\ml2\Data\diabetes.csv")
    df = pd.DataFrame(data)
    return df

def plot_data(df):
    # Plot 1: Scatter Plot of Glucose vs. BMI
    fig1 = px.scatter(df, x='Glucose', y='BMI', color='Outcome', title='Glucose vs. BMI')
    st.plotly_chart(fig1)
    
    # Plot 2: Histogram of Age
    fig2 = px.histogram(df, x='Age', title='Age Distribution')
    st.plotly_chart(fig2)
    
    # Plot 3: Box Plot of Blood Pressure
    fig3 = px.box(df, y='BloodPressure', title='Blood Pressure Distribution')
    st.plotly_chart(fig3)

if __name__ == '__main__':
    df = load_data()
    plot_data(df)
