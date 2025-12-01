import os
import keras
import joblib
import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# Streamlit Page Setup
st.set_page_config(
    page_title="Potato Disease Classifier",
    page_icon="🥔",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.big-font {
    font-size:20px !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.disease-header {
    color: #4CAF50;
    text-shadow: 2px 2px 4px #cccccc;
}
</style>
""", unsafe_allow_html=True)

# Potato Disease Classes
disease_names = ['Early Blight', 'Late Blight', 'Healthy']

# Load Models
@st.cache_resource(show_spinner=False)
def load_models():
    cnn_model = keras.models.load_model('potato_cnn_model.h5')  # Update with your CNN model path
    logistic_model = joblib.load('potato_logistic_regression_model.pkl')  # Update with your logistic regression model path
    # Dummy input to build model
    dummy_input = tf.zeros((1, 180, 180, 3))
    cnn_model(dummy_input)
    return cnn_model, logistic_model

cnn_model, logistic_model = load_models()

# Function to process the image and get predictions
def predict_image(model, image_data, model_type='cnn'):
    input_image = tf.keras.utils.load_img(image_data, target_size=(180, 180))
    input_image_array = tf.keras.utils.img_to_array(input_image) / 255.0
    input_image_batch = tf.expand_dims(input_image_array, 0)

    if model_type == 'cnn':
        predictions = model.predict(input_image_batch)
        result = tf.nn.softmax(predictions[0])
    else:
        # Extract features manually for logistic regression
        feature_extractor = tf.keras.Model(inputs=cnn_model.input, outputs=cnn_model.layers[-2].output)
        features = feature_extractor.predict(input_image_batch)
        predictions = model.predict_proba(features)
        result = tf.convert_to_tensor(predictions[0])

    result = tf.clip_by_value(result, clip_value_min=0, clip_value_max=1)
    predicted_class = np.argmax(result)
    confidence_score = np.max(result) * 100
    return disease_names[predicted_class], confidence_score, result

# Function to plot advanced predictions
def plot_advanced_predictions(cnn_pred, logistic_pred):
    plt.figure(figsize=(15, 6))
    
    # Line plot
    plt.subplot(1, 2, 1)
    plt.plot(disease_names, cnn_pred, marker='o', label='CNN Model', color='#1E90FF')
    plt.plot(disease_names, logistic_pred, marker='s', label='Logistic Regression', color='#32CD32')
    plt.title('Model Prediction Comparison', fontsize=15)
    plt.xlabel('Disease Types', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Heatmap
    plt.subplot(1, 2, 2)
    comparison_data = np.vstack([cnn_pred, logistic_pred])
    sns.heatmap(comparison_data, annot=True, cmap='YlGnBu',
                xticklabels=disease_names,
                yticklabels=['CNN', 'Logistic Regression'])
    plt.title('Prediction Heatmap', fontsize=15)

    plt.tight_layout()
    plt.savefig('advanced_predictions.png')
    plt.close()

# Streamlit Header
st.markdown("<h1 class='disease-header'>🥔 Potato Disease Classifier 🍃</h1>", unsafe_allow_html=True)

# Image Upload
uploaded_file = st.file_uploader('Upload a Potato Leaf Image', type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded_file, width=300, caption='Uploaded Potato Leaf Image')
    with col2:
        with st.spinner('🔍 Predicting... Please wait...'):
            # CNN Prediction
            cnn_disease, cnn_confidence, cnn_full_pred = predict_image(cnn_model, uploaded_file, 'cnn')

            # Logistic Regression Prediction
            logistic_disease, logistic_confidence, logistic_full_pred = predict_image(logistic_model, uploaded_file, 'logistic')

        # Display Predictions
        st.success('✅ Prediction Completed!')
        st.markdown(f"🔬 CNN Model Prediction:", unsafe_allow_html=True)
        st.markdown(f"<span class='big-font'>Disease: {cnn_disease}</span>", unsafe_allow_html=True)
        st.markdown(f"<span class='big-font'>Confidence: {cnn_confidence:.2f}%</span>", unsafe_allow_html=True)
        
        st.markdown(f"📊 Logistic Regression Prediction:", unsafe_allow_html=True)
        st.markdown(f"<span class='big-font'>Disease: {logistic_disease}</span>", unsafe_allow_html=True)
        st.markdown(f"<span class='big-font'>Confidence: {logistic_confidence:.2f}%</span>", unsafe_allow_html=True)

    # Plotting the predictions
    plot_advanced_predictions(cnn_full_pred, logistic_full_pred)
    st.image('advanced_predictions.png', caption='Detailed Model Predictions')

# Footer
st.markdown("---")
st.markdown("<div class='big-font' style='text-align:center;'>🌱 Powered by AI Potato Disease Detection 🌿</div>", unsafe_allow_html=True)      in single code 
