from flask import Flask, request, render_template,send_file, redirect, url_for, flash,session,jsonify
from pymongo import MongoClient
from PIL import Image
import pytesseract
import random
import pickle
import re
import io
import cv2

from groq import Groq
from drug_named_entity_recognition import find_drugs
from tensorflow.keras.models import load_model

from keras.utils import load_img, img_to_array
import numpy as np
from keras.applications.vgg16 import preprocess_input
from datetime import datetime
from fpdf import FPDF
import warnings
import logging




warnings.filterwarnings("ignore")


logging.basicConfig(level=logging.ERROR)

app = Flask(__name__)
app = Flask(__name__, static_url_path='/static' ,static_folder='/')
app.secret_key = 'cts-project-heart-pred'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
scan_model = load_model('models/final_model.h5')

with open('models/scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('models/rf_classifier.pkl', 'rb') as model_file:
    rf_classifier = pickle.load(model_file)


def get_db_connection():
    try:
        client = MongoClient("mongodb+srv://user:rND7ylTUdaOLKlg9@cluster0.pcr8z.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
        db = client['project']  
        print("Connected to MongoDB Atlas successfully!")
        return db
    except Exception as e:
        print(f"Error connecting to MongoDB Atlas: {e}")
        return None
#REPORT GENERATION
def generate_pdf(ultrasound_result, prescription_consistency, overstating_claim_result):

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()


    pdf.set_font('Arial', 'B', 16)
    pdf.cell(200, 10, txt="Medical Insurance Fraud Detection Report", ln=True, align='C')


    pdf.set_font('Arial', '', 12)
    pdf.ln(10)  # New line
    pdf.cell(200, 10, txt="Report Summary", ln=True, align='C')
    pdf.ln(5)

    
    
    if ultrasound_result.lower() == "yes":
        ultrasound_result = "Verified.Status:Passed"
    else:
        ultrasound_result = "Verified.Status:Not Passed.Reason: appendix not detected in the scan"
        
    if prescription_consistency.lower() == "yes":
        prescription_consistency = "Verified.Status:Passed"
    else:
        prescription_consistency = "Verified.Status:Not Passed.Reason: some medicines claimed shouldn't be used for appendix"
        
    if overstating_claim_result.lower() == "yes":
        overstating_claim_result = "Verified.Status:Passed"
    else:
        overstating_claim_result = "Verified.Status:Not Passed.Reason: flagged claim"

    # Patient Info Section
    pdf.cell(200, 10, txt="Patient Information:", ln=True)
    pdf.cell(200, 10, txt="Patient ID: P1234", ln=True)
    pdf.cell(200, 10, txt="Hospital: Apollo Hospital", ln=True)
    pdf.ln(5)

    # Ultrasound Scan
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(200, 10, txt="Ultrasound Scan Result", ln=True)
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 10, txt=f"Result: {ultrasound_result}")
    pdf.ln(5)

    # Prescription Bill
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(200, 10, txt="Prescription Bill Analysis", ln=True)
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 10, txt=f"Prescription Consistency: {prescription_consistency}")
    pdf.ln(5)

    # Overstating Claim
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(200, 10, txt="Overstating Claim Detection", ln=True)
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 10, txt=f"Claim Analysis: {overstating_claim_result}")
    pdf.ln(10)

    # Conclusion
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(200, 10, txt="Conclusion", ln=True)
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 10, txt="Based on the analysis, there is a potential fraud detected in the medical insurance claim.")
    pdf.ln(5)

    # Save the PDF to a byte stream
    pdf_file = "output/medical_fraud_report.pdf"
    pdf.output(pdf_file)
    
#ULTRASOUND SCAN
def predict_image(model, image_file):
    img = Image.open(io.BytesIO(image_file.read()))
    img = img.resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    prediction = model.predict(img_array)
    class_label = 'Class 1' if prediction[0] > 0.5 else 'Class 0'
    return class_label
#MEDICAL PRESCRIPTION CHECK
def extract_medicine_info(image,surgery_type="appendix"):
    client = Groq(api_key="{your_api_key}",)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sample_text = pytesseract.image_to_string(gray)
    print("sample text: ",sample_text)
    cleaned_sample_text = re.sub(r'[^a-zA-Z0-9\s]', '', sample_text)
    cleaned_sample_text = re.sub(r'\s+', ' ', cleaned_sample_text)
    print("cleaned sample_text: ",cleaned_sample_text)
    words = cleaned_sample_text.split(" ")
    ans=find_drugs(words)
    #print(ans)
    medicines_name=[]
    for x in ans:
        medicines_name.append(x[0]['name'])
    print(medicines_name)
    med_name_str = ", ".join(medicines_name)

    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {
                "role": "user",
                "content": "check if the given list of medicine " + med_name_str + 
                           " is given for common hospitalisation and post " + surgery_type +
                           " surgery purposes or for pain relief, infections, etc. Return only in one word as YES or NO."
            },
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )

    ans = ""
    for chunk in completion:
        if chunk.choices[0].delta.content:
            ans += chunk.choices[0].delta.content
    print("ans:\t",ans.strip())
    return ans.strip()
#OVERSTATING CLAIMS
def check_bill(img,medical_ness,scan_res):
    image = Image.open(io.BytesIO(img.read()))
    text = pytesseract.image_to_string(image)


    admission_date = None
    discharge_date = None
    patient_id = None

    # Regular expressions for dates and patient ID
    date_pattern = re.compile(r'(Date of Admission|Date of Discharge):\s*(\d{2}/\d{2}/\d{4})')
    patient_id_pattern = re.compile(r'Patient ID:\s*(\S+)')

    # Find all matches for dates
    for match in date_pattern.finditer(text):
        label, date_str = match.groups()
        try:
            if label == "Date of Admission":
                admission_date = datetime.strptime(date_str, '%m/%d/%Y')
            elif label == "Date of Discharge":
                discharge_date = datetime.strptime(date_str, '%m/%d/%Y')
        except ValueError:
            print(f"Error parsing {label}: {date_str}")
            
    if admission_date and discharge_date:
        days_in_hospital = (discharge_date - admission_date).days
    else:
        print("Could not find both admission and discharge dates.")


    # Find Patient ID
    patient_id_match = patient_id_pattern.search(text)
    if patient_id_match:
        patient_id = patient_id_match.group(1)

    # Print the extracted details
    if admission_date:
        print(f"Date of Admission: {admission_date.strftime('%m/%d/%Y')}")
    else:
        print("Date of Admission not found.")

    if discharge_date:
        print(f"Date of Discharge: {discharge_date.strftime('%m/%d/%Y')}")
    else:
        print("Date of Discharge not found.")

    if patient_id:
        print(f"Patient ID: {patient_id}")
    else:
        print("Patient ID not found.")
    # Regular expression to find "Amount Paid"
    amount_paid_pattern = re.compile(r'Amount Paid:\s*Rs\.?(\d+,?\d+\.?\d*)')

    # Search for "Amount Paid" using the regular expression
    match = amount_paid_pattern.search(text)

    if match:
        amount_paid = match.group(1)
        print(f"Amount Paid: Rs. {amount_paid}")
    else:
        print("Amount Paid not found.")
    input_features = np.array([int(amount_paid.replace(',', '')),medical_ness,scan_res,days_in_hospital,random.randint(1000, 9999)]).reshape(1, -1)
    
    # Scale the input features
    input_features_scaled = scaler.transform(input_features)
    
    # Make a prediction
    prediction = rf_classifier.predict(input_features_scaled)
    
    # Return the prediction result
    result = 'yes' if prediction[0] == 1 else 'no'
    return result


db = get_db_connection()
isLogin = False
@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    global isLogin
    # Get data from the submitted form
    name = request.form.get('name')
    user_id = request.form.get('user_id')
    password = request.form.get('password')
    print(user_id+" "+password)
    
    if db is not None:
        
        try:
            users_collection = db['insurer']
            
             # Replace 'users' with your collection name
            user = users_collection.find_one({'insurer_id': user_id, 'password': password})
            if user is not None:
                isLogin = True
                return render_template('add_clients.html', user=user)
            else:
                return render_template('login.html', error='Invalid username or password')
        except Exception as e:
            print(f"Error querying MongoDB: {e}")
    return render_template('login.html', error='Error querying MongoDB')

@app.route('/addclient', methods=['POST'])
def add_clients():
    client_name = request.form.get('client_name')
    claim_id = request.form.get('claim_id')
    policy_type = request.form.get('policy_type')
    client ={
        'client_id': random.randint(10000, 99999),
        'client_name': client_name,
        'claim_id': claim_id,
        'policy_type': policy_type
    }
    client_collection = db['client']
    client_collection.insert_one(client)
    return redirect('/existing_clients.html')
    

@app.route('/existing_clients.html', methods=['GET'])
def render_existing_clients():
    if isLogin:
        client_collection = db['client']
        clients = client_collection.find({}, {"client_id": 1, "client_name": 1, "claim_id": 1, "policy_type": 1, "_id": 0})
        names_locations = [{"name": client["client_name"], "id": client["client_id"]} for client in clients]
        return render_template('existing_clients.html',names_locations=names_locations)
    else:
        return render_template('login.html')
    
@app.route('/add_clients.html', methods=['GET'])
def render_add_clients():
    if isLogin:
        return render_template('add_clients.html')
    else:
        return render_template('login.html')
    
@app.route('/upload_document/<client_id>', methods=['GET'])
def upload_document(client_id):
    if isLogin:
        return render_template('upload_form.html', client_id=client_id)
    else:
        return render_template('login.html')

@app.route('/process-images', methods=['POST'])
def process_images():
    files = request.files.getlist('prescription')
    pres_results = {}
    surgery_type = request.form.get('surgery_type', 'appendix')
    
    for file in files:
        npimg = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        result = extract_medicine_info(image, surgery_type)
        pres_results[file.filename] = result

    scans= request.files.getlist('scan')
    scan_results = {}
    for scan in scans:
        result = predict_image(scan_model, scan)
        scan_results[scan.filename] = result
    
    bills = request.files.getlist('bill')
    bill_results = {}
    medical_ness = 1 if list(pres_results.values()).count('yes') > list(pres_results.values()).count('no') else 0
    scan_res = 1 if list(scan_results.values()).count('yes') > list(scan_results.values()).count('no') else 0
    for bill in bills:
        result = check_bill(bill,medical_ness=medical_ness,scan_res=scan_res)
        bill_results[bill.filename] = result

     # Determine overall results
    prescription_consistency = 'yes' if list(pres_results.values()).count('yes') > list(pres_results.values()).count('no') else 'no'
    ultrasound_result = 'yes' if list(scan_results.values()).count('yes') > list(scan_results.values()).count('no') else 'no'
    overstating_claim_result = 'yes' if list(bill_results.values()).count('yes') > list(bill_results.values()).count('no') else 'no'

    # Generate PDF with the determined results
    generate_pdf(prescription_consistency, ultrasound_result, overstating_claim_result)
    pdf_path = 'output\\medical_fraud_report.pdf'
    
    # Return the PDF file to the client
    return send_file(pdf_path, as_attachment=True, download_name='medical_fraud_report.pdf', mimetype='application/pdf')

if __name__ == '__main__':
    app.run(debug=True)