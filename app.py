from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from matplotlib import pyplot as plt

import streamlit as st
import seaborn as sns
import pandas as pd
import joblib

st.set_page_config(page_title="Dashboard")
st.title("Breast Cancer Detection 🎀")

lr = joblib.load('model.pkl')

df = pd.read_csv("data.csv")
df = df.drop(['Unnamed: 32', 'id'], axis=1)
df.diagnosis = [1 if value == "M" else 0 for value in df.diagnosis]

y = df['diagnosis'] # target variable
X = df.drop(['diagnosis'], axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.30, random_state=42)

y_pred = lr.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.metric("Model Accuracy", "{:.2f}".format(accuracy), border=True)

fig, ax = plt.subplots()
ax = sns.countplot(x ='diagnosis', data = df)
ax.set(xlabel="Diagnosis", ylabel="Count")
st.pyplot(fig)