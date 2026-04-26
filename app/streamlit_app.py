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
    ### Research Framework: Quantifying Spectral Bias
    This dashboard demonstrates how late-fusion architectures can develop an over-reliance on 
    meteorological priors. By strictly aligning input features with the training distribution, 
    we can visualize where the model fails to integrate visual evidence.
    """)

    model = load_model()

    # Experimental Controls: Aligned with [NDVI, EVI, SAVI, TMAX, TMIN, PRECIP]
    st.sidebar.header("Input Parameters")
    
    st.sidebar.subheader("Vegetation Indices")
    ndvi = st.sidebar.slider("NDVI (Normalized Difference)", 0.0, 1.0, 0.4)
    evi = st.sidebar.slider("EVI (Enhanced Vegetation)", 0.0, 1.0, 0.25)
    savi = st.sidebar.slider("SAVI (Soil Adjusted)", 0.0, 1.0, 0.3)
    
    st.sidebar.subheader("Meteorological Context")
    temp_max = st.sidebar.slider("Maximum Temperature (C)", 10.0, 45.0, 28.0)
    temp_min = st.sidebar.slider("Minimum Temperature (C)", 0.0, 30.0, 18.0)
    rainfall = st.sidebar.slider("Daily Rainfall (mm)", 0.0, 50.0, 5.0)

    # Research Notice regarding Spatial Isolation
    st.warning("""
    **Diagnostic Mode Active:** The spatial input is randomized to isolate the influence 
    of the tabular pathway. If the prediction changes based on sliders while the image 
    remains random, it confirms a mathematical dependency on weather priors.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Spatial Input (Isolator)")
        # Generating randomized pixels to prove imagery is being ignored
        dummy_image = np.random.rand(32, 32, 3)
        st.image(dummy_image, caption="Randomized RGB-NIR composite (Bias Test)", use_container_width=True)

    with col2:
        st.subheader("Model Inference")
        
        # Prepare tensors with strict index alignment:
        # [ndvi, evi, savi, temp_max, temp_min, rainfall]
        spatial_tensor = torch.randn(1, 4, 32, 32)
        tabular_tensor = torch.tensor([[ndvi, evi, savi, temp_max, temp_min, rainfall]], dtype=torch.float32)

        with torch.no_grad():
            outputs = model(spatial_tensor, tabular_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1).numpy()[0]
            prediction = np.argmax(probabilities)

        classes = ["Healthy", "Stressed", "Diseased"]
        
        st.metric(label="System Prediction", value=classes[prediction])
        
        # Professional Probability Distribution Plot
        fig, ax = plt.subplots()
        ax.bar(classes, probabilities, color=['#2ecc71', '#f1c40f', '#e74c3c'])
        ax.set_ylabel('Confidence Level')
        ax.set_title('Class Probability Distribution')
        st.pyplot(fig)

    st.divider()
    st.subheader("Technical Conclusion")
    st.write("""
    The observed behavior surfaces **Spectral Bias**. When the model produces a 'Healthy' 
    prediction based on favorable temperature and rainfall sliders—despite the 
    underlying image being pure noise—it proves that the Late-Fusion architecture 
    has prioritized low-dimensional tabular features over complex spatial ground truth.
    """)

if __name__ == "__main__":
    main()
