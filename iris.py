# Reproduced from https://datasciencechalktalk.com/2019/10/22/building-machine-learning-apps-with-streamlit/

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
 
st.title('Iris')
 
df = pd.read_csv('https://raw.githubusercontent.com/bonchae/data/master/iris.csv')

if st.checkbox('Show dataframe'):
    st.write(df)
 
st.subheader('Scatter plot')
 
species = st.multiselect('Show iris per Name?', df['Name'].unique())
col1 = st.selectbox('Which feature on x?', df.columns[0:4])
col2 = st.selectbox('Which feature on y?', df.columns[0:4])
 
new_df = df[(df['Name'].isin(species))]
st.write(new_df)

# create figure using plotly express
fig = px.scatter(new_df, x =col1,y=col2, color='Name')

# Plot!
st.plotly_chart(fig)
 
st.subheader('Histogram')
 
feature = st.selectbox('Which feature?', df.columns[0:4])

# Filter dataframe
new_df2 = df[(df['Name'].isin(species))][feature]
fig2 = px.histogram(new_df, x=feature, color="Name", marginal="rug")
st.plotly_chart(fig2)
 
##########################################################
# Accepting user data for predicting its Member Type
def accept_user_data():
    sl = st.slider("Choose the sepal length: ", min_value=4.2,
                    max_value=7.9, value=5.0, step=0.1)
    sw = st.slider("Choose the sepal width: ", min_value=2.0,
                    max_value=4.5, value=4.0, step=0.1)
    pl = st.slider("Choose the petal length: ", min_value=1.0,
                    max_value=6.9, value=5.0, step=0.1)
    pw = st.slider("Choose the petal width: ", min_value=0.1,
                    max_value=2.5, value=1.0, step=0.1)

    user_prediction_data = [[sl, sw, pl, pw]]

    return user_prediction_data
##########################################################

st.subheader('Machine Learning models')
 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
 
features= df[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']].values
labels = df['Name'].values
 
X_train,X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=1)
 
alg = ['Decision Tree', 'Support Vector Machine','Logistic Regression']
classifier = st.selectbox('Which algorithm?', alg)
if classifier=='Decision Tree':
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    acc = dtc.score(X_test, y_test)
    st.write('Accuracy: ', acc)
    pred_dtc = dtc.predict(X_test)
    cm_dtc=confusion_matrix(y_test,pred_dtc)
    st.write('Confusion matrix: ', cm_dtc)
 
    try:
        st.subheader('Predict your own input')
        if(st.checkbox("Want to predict on your own Input? ")):
            user_prediction_data = accept_user_data()
            st.write("You entered : ", user_prediction_data)  
            pred = dtc.predict(user_prediction_data)
            st.write("The Predicted Class is: ", pred)
            pred_prob = dtc.predict_proba(user_prediction_data)
            st.write("The Probability is: ", pred_prob)
    except:
    	pass


elif classifier == 'Support Vector Machine':
    svm=SVC()
    svm.fit(X_train, y_train)
    acc = svm.score(X_test, y_test)
    st.write('Accuracy: ', acc)
    pred_svm = svm.predict(X_test)
    cm=confusion_matrix(y_test,pred_svm)
    st.write('Confusion matrix: ', cm)
    try:
        st.subheader('Predict your own input')
        if(st.checkbox("Want to predict on your own Input? ")):
            user_prediction_data = accept_user_data()
            st.write("You entered : ", user_prediction_data)  
            pred = svm.predict(user_prediction_data)
            st.write("The Predicted Class is: ", pred)
    except:
    	pass

elif classifier == 'Logistic Regression':
    lr = LogisticRegression(solver='lbfgs', max_iter=500, multi_class='auto')
    lr.fit(X_train, y_train)
    acc = lr.score(X_test, y_test)
    st.write('Accuracy: ', acc)
    pred_lr = lr.predict(X_test)
    cm=confusion_matrix(y_test, pred_lr)
    st.write('Confusion matrix: ', cm)

    try:
        st.subheader('Predict your own input')
        if(st.checkbox("Want to predict on your own Input? ")):
            user_prediction_data = accept_user_data()
            st.write("You entered : ", user_prediction_data)  
            pred = lr.predict(user_prediction_data)
            st.write("The Predicted Class is: ", pred)
            pred_prob = lr.predict_proba(user_prediction_data)
            st.write("The Probability is: ", pred_prob)
    except:
    	pass