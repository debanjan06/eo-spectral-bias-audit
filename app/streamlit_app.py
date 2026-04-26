import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.models.multi_modal_cnn import MultiModalCNN

def load_model():
    model = MultiModalCNN(num_classes=3)
    model.load_state_dict(torch.load('models/best_baseline_model.pth', map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    return model

def main():
    st.set_page_config(page_title="AgriSight Diagnostic Dashboard", layout="wide")
    
    st.title("AgriSight: Multi-Modal Robustness Analysis")
    st.markdown("""
    ### Research Framework: Quantifying Spectral Bias
    This dashboard demonstrates how late-fusion architectures can develop an over-reliance on 
    meteorological priors. By isolating the tabular and spatial pathways, we can visualize 
    where the model fails to integrate visual evidence.
    """)

    model = load_model()

    # Experimental Controls
    st.sidebar.header("Input Parameters")
    st.sidebar.subheader("Meteorological Context")
    temp = st.sidebar.slider("Maximum Temperature (C)", 10.0, 45.0, 28.0)
    rainfall = st.sidebar.slider("Daily Rainfall (mm)", 0.0, 50.0, 5.0)
    humidity = st.sidebar.slider("Relative Humidity (%)", 10.0, 90.0, 45.0)
    
    st.sidebar.subheader("Vegetation Indices")
    ndvi = st.sidebar.slider("NDVI", 0.0, 1.0, 0.4)
    savi = st.sidebar.slider("SAVI", 0.0, 1.0, 0.3)
    evi = st.sidebar.slider("EVI", 0.0, 1.0, 0.25)

    # Prominent Research Notice
    st.warning("""
    **Diagnostic Mode Active:** The spatial input (satellite patch) is currently randomized 
    to isolate the influence of the weather sliders. If the model predicts 'Healthy' despite 
    static or random pixels, it confirms a mathematical dependency on weather priors.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Spatial Input (Isolator)")
        # Generating random pixels to prove the model ignores them
        dummy_image = np.random.rand(32, 32, 3)
        st.image(dummy_image, caption="Randomized RGB-NIR composite (Bias Test)", use_container_width=True)

    with col2:
        st.subheader("Model Inference")
        
        spatial_tensor = torch.randn(1, 4, 32, 32)
        tabular_tensor = torch.tensor([[ndvi, savi, evi, temp, rainfall, humidity]], dtype=torch.float32)

        with torch.no_grad():
            outputs = model(spatial_tensor, tabular_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1).numpy()[0]
            prediction = np.argmax(probabilities)

        classes = ["Healthy", "Stressed", "Diseased"]
        
        st.metric(label="System Prediction", value=classes[prediction])
        
        fig, ax = plt.subplots()
        ax.bar(classes, probabilities, color=['#2ecc71', '#f1c40f', '#e74c3c'])
        ax.set_ylabel('Probability')
        ax.set_title('Class Distribution')
        st.pyplot(fig)

    st.divider()
    st.subheader("Technical Conclusion")
    st.write("""
    The behavior observed above—where the prediction changes based on temperature or rainfall 
    sliders while the image remains random—is the definition of **Spectral Bias**. 
    In production environments, this would lead to a 'False Healthy' reading in regions like 
    Punjab or Australia where the weather profile is acceptable but the ground truth 
    captured by satellite imagery shows significant crop failure.
    """)

if __name__ == "__main__":
    main()
