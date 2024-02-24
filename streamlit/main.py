import streamlit as st
import pandas as pd 
import numpy as np
import pickle
import base64

@st.cache(suppress_st_warning=True)

def get_fvalue(val):
    feature_dict = {"No":1, "Yes":2}
    for key, value in feature_dict.items():
        if val==key:
            return value
    
def get_value(val, my_dict):
    for key,value in my_dict.items():
        if val==key:
            return value
    
app_mode = st.sidebar.selectbox('Select Page',['Home','Prediction']) # single page

if app_mode=="Home":
    st.title("LOAN PREDICTION")
    st.write('App realised by: Izzarief Zahari')
    st.image('original.jpg')

if app_mode=='Prediction':
    st.title("LOAN PREDICTION")
    st.write('App realised by: Izzarief Zahari')
    st.image('original.jpg')
    st.subheader('You need to fill all necessary information in order to get a reply to your loan request')
    st.sidebar.header("Informations about the client")
    gender_dict = {"Male":1,"Female":2}
    feature_dict = {"No":1,"Yes":2}
    edu = {'Graduate':1,'Not Graduate':2}
    prop = {'Rural':1,'Urban':2,'Semiurban':3}
    ApplicantIncome = st.sidebar.slider('ApplicantIncome',0,10000,0,)
    CoapplicantIncome = st.sidebar.slider('CoapplicantIncome',0,10000,0,)
    LoanAmount = st.sidebar.slider('LoanAmount in K$',9.0,700.0,200.0)
    Loan_Amount_Term = st.sidebar.radio('Loan_Amount_Term',options=['12.0','36.0','60.0','84.0','120.0','180.0','240.0','300.0','360.0','480.0'])
    Credit_History = st.sidebar.radio('Credit_History',(0.0,1.0))
    Gender = st.sidebar.radio('Gender',tuple(gender_dict.keys()))
    Married = st.sidebar.radio('Married',tuple(feature_dict.keys()))
    Self_Employed = st.sidebar.radio('Self Employed',tuple(feature_dict.keys()))
    Dependents = st.sidebar.radio('Dependents',options=['0','1','2','3+'])
    Education = st.sidebar.radio('Education',tuple(edu.keys()))
    Property_Area = st.sidebar.radio('Property_Area',tuple(prop.keys()))

    term0,term1,term2,term3,term4,term5,term6,term7,term8,term9 = 0,0,0,0,0,0,0,0,0,0
    if Loan_Amount_Term == '12.0':
        term0 = 1
    elif Loan_Amount_Term == '36.0':
        term1 = 1
    elif Loan_Amount_Term == '60.0':
        term2 = 1
    elif Loan_Amount_Term == '84.0':
        term3 = 1
    elif Loan_Amount_Term == '120.0':
        term4 = 1
    elif Loan_Amount_Term == '180.0':
        term5 = 1
    elif Loan_Amount_Term == '240.0':
        term6 = 1
    elif Loan_Amount_Term == '300.0':
        term7 = 1
    elif Loan_Amount_Term == '360.0':
        term8 = 1
    else:
        term9 =1

    class_0,class_3,class_1,class_2 = 0,0,0,0
    if Dependents == '0':
        class_0 = 1
    elif Dependents == '1':
        class_1 = 1
    elif Dependents == '2':
        class_2 = 1
    else:
        class_3 = 1

    Rural,Urban,Semiurban=0,0,0
    if Property_Area == 'Urban':
        Urban = 1
    elif Property_Area == 'Semiurban':
        Semiurban = 1
    else:
        Rural = 1

    data1 = {
        'Gender': Gender,
        'Married': Married,
        'Education': Education,
        'Self Employed':Self_Employed,
        'ApplicantIncome':ApplicantIncome,
        'CoapplicantIncome':CoapplicantIncome,
        'LoanAmount':LoanAmount,
        'Credit_History':Credit_History,
        'Dependents':[class_0,class_1,class_2,class_3],
        'Loan_Amount_Term':[term0,term1,term2,term3,term4,term5,term6,term7,term8,term9],
        'Property_Area':[Rural,Urban,Semiurban]
    }

    feature_list=[
        get_value(Gender,gender_dict),
        get_fvalue(Married),
        get_value(Education,edu),
        get_fvalue(Self_Employed),
        ApplicantIncome,
        CoapplicantIncome,
        LoanAmount,
        Credit_History,
        data1['Dependents'][0],data1['Dependents'][1],data1['Dependents'][2],data1['Dependents'][3],
        data1['Loan_Amount_Term'][0],data1['Loan_Amount_Term'][1],data1['Loan_Amount_Term'][2],data1['Loan_Amount_Term'][3],data1['Loan_Amount_Term'][4],data1['Loan_Amount_Term'][5],data1['Loan_Amount_Term'][6],data1['Loan_Amount_Term'][7],data1['Loan_Amount_Term'][8],data1['Loan_Amount_Term'][9],
        data1['Property_Area'][0],data1['Property_Area'][1],data1['Property_Area'][2]
        ]

    single_sample = np.array(feature_list).reshape(1,-1)

    if st.button("Predict"):
        loaded_model = pickle.load(open('logisticRegressionModel.pkl','rb'))
        prediction = loaded_model.predict(single_sample)
        prediction_score = loaded_model.predict_proba(single_sample)
        if prediction[0] == 0:
            st.error('According to our calculations, your loan request is not approved')
            st.write(f'Your score is {prediction_score[0][1]}')
        elif prediction[0] == 1:
            st.success('Congratulations!! your loan request is approved')
            st.write(f'Your score is {prediction_score[0][1]}')