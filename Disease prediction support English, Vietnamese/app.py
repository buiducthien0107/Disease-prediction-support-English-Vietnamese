import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier

def load_resources(language):
    if language == 'English':
        model_path = 'model/model.cbm'
        data_path = 'datasets/dataset_diseases.csv'
        desc_path = 'datasets/symptom_Description.csv'
        prec_path = 'datasets/symptom_precaution.csv'
        symptom_columns = ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4']
        disease_column = 'Disease'
        translations = {
            'title': 'Disease Prediction Model',
            'description': "Enter symptoms and click 'Predict' for advice. Not a medical recommendation, consult a specialist.",
            'select_symptoms': 'Select Symptoms',
            'main_symptom': 'Select the main symptom',
            'second_symptom': 'Select a second symptom (if any)',
            'third_symptom': 'Select a third symptom (if any)',
            'fourth_symptom': 'Select a fourth symptom (if any)',
            'predict_button': 'Predict',
            'prediction': 'Prediction',
            'predicted_disease': 'Predicted disease',
            'description_label': 'Description',
            'recommendations': 'Recommendations',
            'precaution': 'Precaution'
        }
    else:  # Assuming 'Vietnamese'
        model_path = 'model/model_vn.cbm'
        data_path = 'datasets_vn/dataset_diseases_vn.csv'
        desc_path = 'datasets_vn/symptom_Description_vn.csv'
        prec_path = 'datasets_vn/symptom_precaution_vn.csv'
        symptom_columns = ['Triệu chứng 1', 'Triệu chứng 2', 'Triệu chứng 3', 'Triệu chứng 4']
        disease_column = 'Bệnh'
        translations = {
            'title': 'Dự Đoán Bệnh',
            'description': "Nhập triệu chứng và nhấn 'Dự đoán' để nhận lời khuyên. Không phải là khuyến nghị y tế, hãy tham khảo ý kiến chuyên gia.",
            'select_symptoms': 'Chọn triệu chứng',
            'main_symptom': 'Triệu chứng chính',
            'second_symptom': 'Triệu chứng thứ hai (nếu có)',
            'third_symptom': 'Triệu chứng thứ ba (nếu có)',
            'fourth_symptom': 'Triệu chứng thứ tư (nếu có)',
            'predict_button': 'Dự đoán',
            'prediction': 'Dự đoán bệnh',
            'predicted_disease': 'Bệnh dự đoán',
            'description_label': 'Mô tả',
            'recommendations': 'Khuyến cáo',
            'precaution': 'Biện pháp phòng ngừa'
        }

    model = CatBoostClassifier()
    model.load_model(model_path)
    
    try:
        data = pd.read_csv(data_path, sep=';', engine='python', on_bad_lines='skip')
        desc = pd.read_csv(desc_path, sep=',', engine='python', on_bad_lines='skip')
        prec = pd.read_csv(prec_path, sep=',', engine='python', on_bad_lines='skip')
        data = data.fillna('none')
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        return None, None, None, None, None, None, translations
    except pd.errors.ParserError as e:
        st.error(f"Parsing error: {e}. Please check the CSV file format.")
        return None, None, None, None, None, None, translations
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None, None, None, None, None, None, translations
    
    return model, data, desc, prec, symptom_columns, disease_column, translations

# Set page configuration
st.set_page_config(page_title="Disease Prediction Model", layout="centered")

# Load resources based on selected language
language = st.selectbox('Select Language', ['English', 'Vietnamese'])
model, data, desc, prec, symptom_columns, disease_column, translations = load_resources(language)

st.markdown("""
    <style>
    .title {
        color: #4a90e2;
        font-size: 28px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .description-text {
        font-size: 16px;
        text-align: center;
        margin-bottom: 20px;
    }
    .symptom-selection {
        margin-bottom: 15px;
    }
    .prediction {
        font-size: 18px;
        color: #ff5722;
        text-align: center;
        margin-top: 15px;
    }
    .result-section {
        margin-top: 20px;
        padding: 15px;
        background-color: #f9f9f9;
        border-radius: 8px;
    }
    .compact-box {
        margin-top: 0;
        margin-bottom: 0;
        padding: 5px;
    }
    .stSelectbox > div {
        padding: 5px !important;
    }
    .stButton button {
        padding: 8px 16px;
        font-size: 14px;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.markdown(f"<div class='title'>{translations['title']}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='description-text'>{translations['description']}</div>", unsafe_allow_html=True)

if data is not None:
    def predict_disease(input_data):
        prediction = model.predict([input_data])
        return prediction[0]

    # Symptom selection layout
    st.markdown(f"<div class='symptom-selection'><h4>{translations['select_symptoms']}</h4></div>", unsafe_allow_html=True)
    
    with st.form(key='symptom_form'):
        col1, col2 = st.columns([1, 1])
        with col1:
            symptom_1 = st.selectbox(translations['main_symptom'], data[symptom_columns[0]].unique(), key='symptom_1', label_visibility="collapsed")
            symptom_2 = st.selectbox(translations['second_symptom'], data[symptom_columns[1]].unique(), key='symptom_2', label_visibility="collapsed")
        with col2:
            symptom_3 = st.selectbox(translations['third_symptom'], data[symptom_columns[2]].unique(), key='symptom_3', label_visibility="collapsed")
            symptom_4 = st.selectbox(translations['fourth_symptom'], data[symptom_columns[3]].unique(), key='symptom_4', label_visibility="collapsed")
        
        # Predict button
        submit_button = st.form_submit_button(label=translations['predict_button'], help="Click to predict the disease based on your symptoms")

    if submit_button:
        prediction = predict_disease([symptom_1, symptom_2, symptom_3, symptom_4])

        # Display prediction results
        st.markdown(f"<div class='result-section'>", unsafe_allow_html=True)
        st.markdown(f"<div class='prediction'>{translations['prediction']}: {prediction}</div>", unsafe_allow_html=True)

        # Ensure prediction is a valid disease name and exists in the dataset
        prediction_str = str(prediction).strip().lower()
        desc[disease_column] = desc[disease_column].str.strip().str.lower()

        if prediction_str in desc[disease_column].values:
            selected_desc = desc[desc[disease_column] == prediction_str]['Description'].values
            if len(selected_desc) > 0:
                st.markdown(f"<div class='description'>{translations['predicted_disease']}: {prediction_str}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='description'>{translations['description_label']}: {selected_desc[0]}</div>", unsafe_allow_html=True)
                
                st.markdown(f"<div class='description'>{translations['recommendations']}:</div>", unsafe_allow_html=True)
                selected_prec = prec[prec[disease_column] == prediction_str].iloc[:, 1:].values
                for i, precaution in enumerate(selected_prec):
                    if precaution:
                        st.markdown(f"<div class='precaution'>{translations['precaution']} {i+1}: {precaution}</div>", unsafe_allow_html=True)
        st.markdown(f"</div>", unsafe_allow_html=True)
