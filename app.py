import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Page config
st.set_page_config(
    page_title="CGPA to Package Predictor",
    page_icon="📈",
    layout="centered"
)

# Load trained model
model = joblib.load("resgressiom_model_joblib")

# Title & description
st.title("📈 CGPA to Package Predictor")
st.markdown(
    "💡 *Predict your expected salary package (LPA) based on CGPA*"
)

st.divider()

# -------- INPUT SECTION --------
st.subheader("🎓 Enter Your CGPA")

col1, col2 = st.columns(2)

with col1:
    cgpa = st.slider(
        "CGPA Slider",
        min_value=0.0,
        max_value=10.0,
        value=7.0,
        step=0.01
    )

with col2:
    cgpa = st.number_input(
        "CGPA Input",
        min_value=0.0,
        max_value=10.0,
        value=cgpa,
        step=0.01
    )

# -------- PREDICTION --------
if st.button("🔮 Predict Package", use_container_width=True):
    cgpa_array = np.array([[cgpa]])
    prediction = model.predict(cgpa_array)
    predicted_value = float(prediction[0])

    st.divider()

    # Result section
    st.subheader("📊 Prediction Result")

    st.metric(
        label="💰 Expected Package (LPA)",
        value=f"{predicted_value:.2f}"
    )

    # Progress bar (relative visualization)
    progress = min(predicted_value / 20, 1.0)  # assuming max ~20 LPA
    st.progress(progress)

    # -------- VISUALIZATION --------
    st.subheader("📈 CGPA vs Package Trend")

    cgpa_range = np.linspace(0, 10, 50).reshape(-1, 1)
    package_preds = model.predict(cgpa_range)

    chart_df = pd.DataFrame({
    "CGPA": cgpa_range.flatten(),
    "Predicted Package (LPA)": package_preds.flatten()
})


    st.line_chart(chart_df, x="CGPA", y="Predicted Package (LPA)")

    st.success("✅ Prediction completed successfully!")

# -------- FOOTER --------
st.divider()
st.caption("🚀 Built with Streamlit | ML Regression Model")

