import streamlit as st

st.set_page_config(page_title="Iris Classifier App", layout="centered")

st.title("🌸 Welcome to the Iris Classification App")

st.markdown("""
Use the sidebar to navigate between pages:

- **Classifier**: Input flower measurements to get species prediction.
- **About**: Learn more about the dataset and model.
""")