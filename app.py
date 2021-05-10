import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

model = pickle.load(open('model/model_encodings_saved.pkl', 'rb'))
label_encoder_dict = pickle.load(open('lableencoderdict.pkl','rb'))

@app.route('/')

def home():

    return ("Hi , this is an api for fraud detection")

@app.route('/predict_api',methods=['POST'])

def predict_api():

    '''

    For direct API calls through request

    '''

    data = request.get_json(force=True)
    print(data)
    encode_df = pd.DataFrame(data,index=['i',])
    test_cat_df = encode_df.select_dtypes(include=['object']).copy()
    test_num_df = encode_df.select_dtypes(include=["int64"]).copy()
    
    for column in test_cat_df.columns:
        label_encoder =  label_encoder_dict[column]
        test_cat_df[column] = label_encoder.transform(test_cat_df[column])
        
    final_test_df = pd.concat([test_num_df,test_cat_df], axis=1)
    df_list = final_test_df.values.tolist()
    print(df_list)

    prediction = model.predict(np.array(df_list)).tolist()
    prediction_confidence = model.predict_proba(np.array(df_list)).tolist()
    print(prediction_confidence)
    
    if prediction[0] == 1:
        confidence = prediction_confidence[0][1]
    else:
        confidence = prediction_confidence[0][0]

    return jsonify({'prediction': prediction[0], 'score': confidence})



if __name__ == "__main__":

    app.run(debug=True)