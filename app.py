from flask import Flask, request, render_template
# from flask_ngrok import run_with_ngrok
import numpy as np
import pickle


app = Flask(__name__)
# run_with_ngrok(app)

    
model_RF=pickle.load(open(r'C:\InhouseInternship\majorproject7\model_random.pkl', 'rb')) 
model_KNN=pickle.load(open(r'C:\InhouseInternship\majorproject7\model_knn.pkl', 'rb')) 
# model_K_SVM=pickle.load(open('Major_SVM_linear.pkl', 'rb')) 
model_DT=pickle.load(open(r'C:\InhouseInternship\majorproject7\model_dt.pkl', 'rb')) 
model_NB=pickle.load(open(r'C:\InhouseInternship\majorproject7\model_nb.pkl', 'rb')) 



@app.route('/')
def home():
  
    return render_template("index.html")
#------------------------------About us-------------------------------------------
@app.route('/aboutusnew')
def aboutusnew():
    return render_template('aboutusnew.html')
  #------------------------------Project-------------------------------------------
@app.route('/project')
def project():
    return render_template('project.html')

@app.route('/predict',methods=['GET'])

def predict():
    
     
    height = float(request.args.get('HEIGHT'))
    weight = float(request.args.get('WEIGHT'))   

 
    model = (request.args.get('Model'))

    if  model=="Random Forest Classifier":
      prediction = model_RF.predict([[height, weight]])

    elif model=="Decision Tree Classifier":
      prediction = model_DT.predict([[height, weight]])

    elif model=="KNN Classifier":
      prediction = model_KNN.predict([[height, weight]])

    # elif model=="SVM Classifier":
    #   prediction = model_K_SVM.predict([[height, weight]])

    else:
      prediction = model_NB.predict([[height, weight]])
    
    if prediction == [0]:
      return render_template('index.html', prediction_text='The Gender is Female', extra_text =" -- Prediction by " + model)
    
    else:
      return render_template('index.html', prediction_text='The Gender is Male', extra_text =" -- Prediction by " + model)

# app.run()
if __name__ == "__main__":
     app.run(debug=True)
