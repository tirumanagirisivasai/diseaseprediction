from flask import Flask, request, render_template
import pickle
from PIL import Image
import numpy as np
import cv2

app = Flask(__name__)

# Load the model1(done)
heart_stroke_model = pickle.load(open('heart_stroke.p', 'rb'))

# Load the model2(done)
#with open('models/brain_stroke.p', 'rb') as model_file:
#    brain_tumor_model = pickle.load(model_file)

# Load the model3(done)
brain_stroke_model = pickle.load(open('brain_stroke_prediction.p','rb'))

# Load the model4(done)
diabetes_model = pickle.load(open('diabetes.p', 'rb'))

# Load the model5(done)
breast_model = pickle.load(open('breast_cancer_prediction.p', 'rb'))

#load the model6
liver_model = pickle.load(open('liver_disease.p', 'rb'))

#app routing
@app.route('/')
def home():
    return render_template('index.html')

#heartstroke prerdiction
@app.route('/sample')
def heart():
    return render_template('sample.html')
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            data = (np.asarray([int(x) for x in request.form.values()])).reshape(1,-1)
            
            #data1 = (np.asarray([63,1,3,145,233,1,0,150,0,2.3,0,0,1])).reshape(1,-1)
            #return render_template('sample.html', prediction_text=data
            
            prediction = heart_stroke_model.predict(data)
            if prediction == 0:
                text = "Your Heart Condition Looks Great"
            else:
                text = "There are chances of getting a heart attack"

            return render_template('sample.html', prediction_text=text)
        except Exception as e:
            return render_template('sample.html', prediction_text=e)

#brain tumor model
@app.route('/tumor')
def tumorpage():
    return render_template('braintumorpage.html')

@app.route('/predict_tumor', methods=['POST'])
def predict_tumor():
    if 'mriscan' not in request.files:
        return render_template('braintumorpage.html', prediction_text="No image selected")
    
    image_file = request.files['mriscan']
   
    if image_file.filename == '':
        return render_template('braintumorpage.html', prediction_text="No image selected")
  
    try:
        photo = cv2.imread(image_file)
        photo = Image.fromarray(photo, 'RGB')
        photo = photo.resize((64,64))
        test = np.array(photo)
        test = test/255.0
        test = np.expand_dims(test,axis=0)
        prediction = brain_tumor_model.predict(test)
        answer = np.argmax(prediction)
       
        if answer==1:
            return render_template('braintumorpage.html', prediction_text="Brain tumor detected")
        
        else:
            return render_template('braintumorpage.html', prediction_text="No Brain tumor detected")
        
    except Exception as e:
        return render_template('braintumorpage.html', prediction_text=str(e))
    
#brain stroke
@app.route('/brainstroke')
def render():
    return render_template("brainstroke.html")

@app.route('/predict_brainstroke', methods=['POST'])
def predict_brainstroke():
    input_features = [int(x) for x in request.form.values()]
    
    predict_data = np.array(input_features)
    predict_data = predict_data.reshape(1,-1)
    
    predicted = brain_stroke_model.predict(predict_data)
    
    if predicted == 1:
        return render_template('brainstrokepage.html', predicted_text = "There is more chances of affected by brain stroke")
    else:
        return render_template('brainstrokepage.html', predicted_text = "There are less chances of affected by brain stroke")
    
#diabetes prediction
@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')

@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    input_variables = [int(x) for x in request.form.values()]
    predict_data = np.array(input_variables)
    predict_data = predict_data.reshape(1,-1)
    predicted = diabetes_model.predict(predict_data)
    
    if predicted==1:
        return render_template('diabetesepage.html', predicted_text = "This person has diabetes")
    else:
        return render_template('diabetesepage.html', predicted_text = "This person has diabetes")

#breast cancer
@app.route('/breastcancer')
def bcancer():
    return render_template('bcancer.html')

@app.route('/predict_bcancer', methods=['POST'])
def predict_bcancer():
    input_variables = [int(x) for x in request.form.values()]
    predict_data = np.array(input_variables)
    predict_data = predict_data.reshape(1,-1)
    predicted = breast_model.predict(predict_data)
    
    if predicted==1:
        return render_template('bcancer.html', predicted_text='Effected by Breast Cancer')
    else:
        return render_template('bcancer.html', predicted_text='Not effected by Breast Cancer')
    
#liver model
@app.route('/liver')
def liver():
    return render_template('liver_dis.html')

@app.route('/predict_liver', methods=["POST"])
def predict_liver():
    input_variables = [int(x) for x in request.form.values()]
    predict_data = np.array(input_variables)
    predict_data = predict_data.reshape(1,-1)
    predicted = breast_model.predict(predict_data)
    
    if predicted==1:
        return render_template('liver_dis.html', predicted_text='Effected by Liver Disease')
    else:
        return render_template('liver_dis.html', predicted_text='Not effected by Liver disease')
    

    

if __name__ == "__main__":
    app.run(debug=True)
