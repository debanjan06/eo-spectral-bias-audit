import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.models.multi_modal_cnn import MultiModalCNN

def load_model():
    model = MultiModalCNN(num_classes=3)
    # weights_only=True is used for security best practices
    model.load_state_dict(torch.load('models/best_baseline_model.pth', map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    return model

def main():
    st.set_page_config(page_title="AgriSight Diagnostic Dashboard", layout="wide")
    
    st.title("AgriSight: Multi-Modal Robustness Analysis")
    st.markdown("""
    This dashboard serves as a diagnostic tool to evaluate how late-fusion architectures 
    integrate spatial and meteorological data. It is specifically designed to surface 
    bias patterns in regional crop health classification.
    """)

    model = load_model()

    # Sidebar for meteorological parameter injection
    st.sidebar.header("Input Parameters")
    st.sidebar.subheader("Meteorological Context")
    temp = st.sidebar.slider("Maximum Temperature (C)", 10.0, 45.0, 28.0)
    rainfall = st.sidebar.slider("Daily Rainfall (mm)", 0.0, 50.0, 5.0)
    humidity = st.sidebar.slider("Relative Humidity (%)", 10.0, 90.0, 45.0)
    
    st.sidebar.subheader("Vegetation Indices")
    ndvi = st.sidebar.slider("NDVI", 0.0, 1.0, 0.4)
    savi = st.sidebar.slider("SAVI", 0.0, 1.0, 0.3)
    evi = st.sidebar.slider("EVI", 0.0, 1.0, 0.25)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Spatial Input (Satellite Patch)")
        # Simulating a Sentinel-2 4-channel patch
        dummy_image = np.random.rand(32, 32, 3)
        st.image(dummy_image, caption="Simulated RGB-NIR composite", use_container_width=True)
        st.info("Log: Using randomized spatial tensor to isolate tabular feature influence.")

    with col2:
        st.subheader("Diagnostic Prediction")
        
        # Prepare tensors
        spatial_tensor = torch.randn(1, 4, 32, 32)
        tabular_tensor = torch.tensor([[ndvi, savi, evi, temp, rainfall, humidity]], dtype=torch.float32)

        with torch.no_grad():
            outputs = model(spatial_tensor, tabular_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1).numpy()[0]
            prediction = np.argmax(probabilities)

        classes = ["Healthy", "Stressed", "Diseased"]
        
        # Displaying results in a professional metric format
        st.metric(label="Primary Classification", value=classes[prediction])
        
        # Probability Distribution Plot
        fig, ax = plt.subplots()
        ax.bar(classes, probabilities, color=['#2ecc71', '#f1c40f', '#e74c3c'])
        ax.set_ylabel('Confidence Level')
        ax.set_title('Class Probability Distribution')
        st.pyplot(fig)

    st.divider()
    st.subheader("Research Insight: Modal Over-reliance")
    st.write("""
    By adjusting the sliders in the sidebar, users can observe how the model reacts to 
    meteorological shifts. A consistent 'Healthy' output regardless of image quality 
    indicates the presence of Spectral Bias, where the model prioritizes tabular 
    priors over spatial ground truth.
    """)

if __name__ == "__main__":
    main()
