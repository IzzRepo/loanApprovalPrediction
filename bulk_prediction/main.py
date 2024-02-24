import os, sys
import pandas as pd 
import numpy as np
import pickle
import warnings
from absl import app

warnings.filterwarnings("ignore")

sdk_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"solution_sdk")
sys.path.append(sdk_path)

scaler_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"saved_model/scaler.pkl")
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"saved_model/logisticRegressionModel.pkl")
output = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"prediction_output")

import solution_sdk.components as comp

def main(_):

    df = pd.read_csv("test.csv")
    df = comp.impute_missing_values(df)
    df = comp.encode_categorical(df)
    X = df.drop(['Loan_ID'], axis=1)
    scaler = pickle.load(open(scaler_path,"rb"))
    scaled_X = scaler.transform(X)
    model = pickle.load(open(model_path,"rb"))
    y_pred = model.predict(scaled_X)
    y_pred_score = model.predict_proba(scaled_X)

    predict_proba_list = []
    for score in y_pred_score:
        predict_proba_list.append(score[1])

    df = pd.DataFrame({'Loan_ID':df['Loan_ID'].tolist(),
                       'Predict_Loan_Approval':y_pred,
                       'Probability_Loan_Approval':predict_proba_list})
    
    print("Prediction completed")
    
    df.to_csv(output + "/bulk_prediction_loan_approval.csv",index=False)
    print("Export to csv completed")

if __name__ == "__main__":
    app.run(main)