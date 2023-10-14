import pickle
import xgboost as xgb
from xgboost import XGBRegressor
from flask import Flask,request,render_template
app=Flask(__name__)
# filename='xgb_regressor'
loaded_model=pickle.load(open('xgb_regressor.pkl','rb'))
import numpy as np

@app.route('/')
def index():
    print(1)
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    
    test=[float (x) for x in request.form.values()]
    # test=request.form.values()
    print(test)
    test=[np.asarray(test)]
    # test_reshaped=test.reshape(1,-1)
    y_pred =loaded_model.predict(test)
    print(y_pred)
    
    return render_template("index.html",predict='{}' .format(y_pred))

if __name__ == '__main__':
    app.run(debug=True)