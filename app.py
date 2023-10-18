import pickle
from flask import Flask,request,render_template
import numpy as np
app=Flask(__name__)
# filename='xgb_regressor'
loaded_model=pickle.load(open('model_knn (1).sav','rb'))


@app.route('/')
def index():
    print(1)
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    
    test=[float (x) for x in request.form.values()]
    # test=request.form.values()
    print(test)
    test=np.asarray(test)
    test_reshaped=test.reshape(1,-1)
    y_pred =loaded_model.predict(test_reshaped)
    print(y_pred)
       
    return render_template("index.html",predict='{}' .format(int(y_pred)))

if __name__ == '__main__':
    app.run(debug=True)