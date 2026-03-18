
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix

st.set_page_config(page_title="Universal Bank AI Marketing Dashboard", layout="wide")

st.title("🏦 Universal Bank Personal Loan AI Dashboard")

st.markdown("""
This dashboard helps marketing teams understand customer behaviour and identify high‑probability customers for personal loan campaigns.
""")

@st.cache_data
def load_data():
    return pd.read_csv("UniversalBank.csv")

df = load_data()

tabs = st.tabs(["📊 Overview","📈 Descriptive Analytics","🤖 Model Performance","🎯 Predict New Customers"])

# ---------------------------------------------------------
# OVERVIEW
# ---------------------------------------------------------

with tabs[0]:

    st.header("Dataset Overview")

    col1,col2,col3 = st.columns(3)

    col1.metric("Total Customers", df.shape[0])
    col2.metric("Features", df.shape[1])
    col3.metric("Loan Acceptance Rate", str(round(df["Personal Loan"].mean()*100,2))+"%")

    st.dataframe(df.head())

# ---------------------------------------------------------
# DESCRIPTIVE ANALYTICS
# ---------------------------------------------------------

with tabs[1]:

    st.header("Customer Insights")

    c1,c2 = st.columns(2)

    fig = px.histogram(df,x="Income",nbins=40,color_discrete_sequence=["#1f77b4"])
    fig.update_layout(title="Income Distribution")
    c1.plotly_chart(fig,use_container_width=True)
    c1.markdown("Higher income segments show greater financial capacity to accept personal loan offers.")

    fig = px.histogram(df,x="CCAvg",nbins=40,color_discrete_sequence=["#ff7f0e"])
    fig.update_layout(title="Credit Card Spending")
    c2.plotly_chart(fig,use_container_width=True)
    c2.markdown("Customers with higher monthly credit card spending demonstrate stronger purchasing behaviour.")

    c3,c4 = st.columns(2)

    fig = px.box(df,x="Education",y="Income",color="Education")
    fig.update_layout(title="Income by Education Level")
    c3.plotly_chart(fig,use_container_width=True)
    c3.markdown("Education level often correlates with income and financial product adoption.")

    fig = px.pie(df,names="Personal Loan",title="Loan Acceptance Distribution",color_discrete_sequence=["#2ca02c","#d62728"])
    c4.plotly_chart(fig,use_container_width=True)
    c4.markdown("Only a small portion of customers accepted loans previously, showing opportunity for targeted marketing.")

# ---------------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------------

with tabs[2]:

    st.header("Machine Learning Model Comparison")

    X = df.drop(columns=["Personal Loan","ID"])
    y = df["Personal Loan"]

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)

    models = {
        "Decision Tree":DecisionTreeClassifier(),
        "Random Forest":RandomForestClassifier(),
        "Gradient Boosting":GradientBoostingClassifier()
    }

    results=[]
    roc_fig=plt.figure()

    for name,model in models.items():

        model.fit(X_train,y_train)

        train_pred=model.predict(X_train)
        test_pred=model.predict(X_test)

        train_acc=accuracy_score(y_train,train_pred)
        test_acc=accuracy_score(y_test,test_pred)

        precision=precision_score(y_test,test_pred)
        recall=recall_score(y_test,test_pred)
        f1=f1_score(y_test,test_pred)

        results.append([name,train_acc,test_acc,precision,recall,f1])

        probs=model.predict_proba(X_test)[:,1]

        fpr,tpr,_=roc_curve(y_test,probs)
        roc_auc=auc(fpr,tpr)

        plt.plot(fpr,tpr,label=name+" (AUC="+str(round(roc_auc,3))+")")

        st.subheader(name+" Confusion Matrix")

        cm=confusion_matrix(y_test,test_pred)
        cm_percent=cm/cm.sum()

        fig,ax=plt.subplots()
        sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        st.write("Percentage Distribution")
        st.dataframe((cm_percent*100).round(2))

    results_df=pd.DataFrame(results,columns=["Model","Train Accuracy","Test Accuracy","Precision","Recall","F1 Score"])

    st.subheader("Model Comparison Table")
    st.dataframe(results_df)

    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()

    st.pyplot(roc_fig)

# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

with tabs[3]:

    st.header("Predict Personal Loan Acceptance")

    uploaded=st.file_uploader("Upload CSV file",type=["csv"])

    if uploaded:

        test=pd.read_csv(uploaded)

        model=RandomForestClassifier()
        model.fit(X,y)

        preds=model.predict(test)

        test["Predicted Personal Loan"]=preds

        st.dataframe(test.head())

        csv=test.to_csv(index=False).encode()

        st.download_button("Download Results",csv,"loan_predictions.csv","text/csv")
