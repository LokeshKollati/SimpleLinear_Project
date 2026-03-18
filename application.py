from flask import Flask, request, render_template
import pickle

application = Flask(__name__)
app=application

model = pickle.load(open('linear_reg_model.pkl', 'rb'))
Scalar=pickle.load(open('Scalar.pkl','rb'))
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method==['GET']:
        return render_template('index.html')
    else:
        years=request.form.get('years')
        years_scaled=Scalar.transform([[years]])
        result=model.predict(years_scaled)
        return render_template('index.html',result=result,years=years)
   

if __name__ == "__main__":
    app.run(debug=True)

