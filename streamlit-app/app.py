import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Prediksi Nilai Ujian Siswa", layout="wide")

# Load 
@st.cache_resource
def load_model():
    return joblib.load("model/model_rf.pkl")

@st.cache_resource
def load_pca():
    return joblib.load("model/pca.pkl")

@st.cache_resource
def load_scaler():
    return joblib.load("model/scaler.pkl")

model = load_model()
pca = load_pca()
scaler = load_scaler()

# Title
st.markdown("<h1 style='color:#2c3e50;'>Prediksi Nilai Ujian Siswa</h1>", unsafe_allow_html=True)
st.info("Pilih metode input: **1 siswa (manual)** atau **upload data banyak siswa (.csv/.xlsx)**.")

# Input mode selection
mode = st.radio("Pilih Metode Input:", ["Manual (1 siswa)", "Upload File (Banyak siswa)"])

# Manual input
if mode == "Manual (1 siswa)":
    with st.form("prediction_form"):
        st.subheader("Formulir Data Siswa")

        col1, col2, col3 = st.columns(3)
        with col1:
            Parental_Involvement = st.slider("Parental Involvement (1-3)", 1, 3, 2)
            Access_to_Resources = st.slider("Access to Resources (1-3)", 1, 3, 2)
            Previous_Scores = st.number_input("Previous Scores", 0, 100, 70)
            Tutoring_Sessions = st.slider("Tutoring Sessions per Week", 0, 5, 1)

        with col2:
            Parental_Education_Level = st.selectbox("Parental Education Level", ["High School", "College", "Postgraduate"])
            Extracurricular = st.radio("Ikut Ekstrakurikuler?", ["Yes", "No"])
            Internet = st.radio("Akses Internet di Rumah?", ["Yes", "No"])

        with col3:
            School_Type = st.radio("Tipe Sekolah", ["Private", "Public"])
            Peer_Influence = st.selectbox("Pengaruh Teman", ["Negative", "Neutral", "Positive"])
            Learning_Disabilities = st.radio("Hambatan Belajar?", ["Yes", "No"])
            Gender = st.radio("Jenis Kelamin", ["Male", "Female"])

        Attendance = st.slider("Kehadiran (%)", 0, 100, 50)
        Hours_Studied = st.slider("Jam Belajar / Minggu", 0, 50, 15)

        submitted = st.form_submit_button("Prediksi Sekarang")

        if submitted:
            df_attendance = pd.DataFrame([[Attendance, Hours_Studied]], columns=["Attendance", "Hours_Studied"])
            dimension = pca.transform(df_attendance)[0][0]
            scaled_df = pd.DataFrame([[Previous_Scores, dimension]], columns=["Previous_Scores", "dimension"])
            scaled = scaler.transform(scaled_df)[0]

            data = {
                'Parental_Involvement': Parental_Involvement,
                'Access_to_Resources': Access_to_Resources,
                'Previous_Scores': scaled[0],
                'Tutoring_Sessions': Tutoring_Sessions,
                'Parental_Education_Level': {'High School': 1, 'College': 2, 'Postgraduate': 3}[Parental_Education_Level],
                'Extracurricular_Activities_No': Extracurricular == 'No',
                'Extracurricular_Activities_Yes': Extracurricular == 'Yes',
                'Internet_Access_No': Internet == 'No',
                'Internet_Access_Yes': Internet == 'Yes',
                'School_Type_Private': School_Type == 'Private',
                'School_Type_Public': School_Type == 'Public',
                'Peer_Influence_Negative': Peer_Influence == 'Negative',
                'Peer_Influence_Neutral': Peer_Influence == 'Neutral',
                'Peer_Influence_Positive': Peer_Influence == 'Positive',
                'Learning_Disabilities_No': Learning_Disabilities == 'No',
                'Learning_Disabilities_Yes': Learning_Disabilities == 'Yes',
                'Gender_Female': Gender == 'Female',
                'Gender_Male': Gender == 'Male',
                'dimension': scaled[1]
            }

            df_input = pd.DataFrame([data])
            pred = model.predict(df_input)[0]

            st.success("Prediksi berhasil!")
            st.metric(label="Nilai Ujian yang Diprediksi", value=f"{pred:.2f}", delta="Prediksi AI")

# Upload file
else:
    st.subheader("Upload File Data Siswa")
    uploaded_file = st.file_uploader("Unggah file .csv atau .xlsx", type=["csv", "xlsx"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)

            expected = [
                'Parental_Involvement', 'Access_to_Resources', 'Previous_Scores',
                'Tutoring_Sessions', 'Parental_Education_Level',
                'Extracurricular_Activities_No', 'Extracurricular_Activities_Yes',
                'Internet_Access_No', 'Internet_Access_Yes',
                'School_Type_Private', 'School_Type_Public',
                'Peer_Influence_Negative', 'Peer_Influence_Neutral', 'Peer_Influence_Positive',
                'Learning_Disabilities_No', 'Learning_Disabilities_Yes',
                'Gender_Female', 'Gender_Male',
                'Attendance', 'Hours_Studied'
            ]

            missing = [col for col in expected if col not in df.columns]
            if missing:
                st.error(f"Kolom hilang dalam file: {missing}")
            else:
                df["Parental_Education_Level"] = df["Parental_Education_Level"].map({
                    "High School": 1,
                    "College": 2,
                    "Postgraduate": 3
                })

                dimension = pca.transform(df[["Attendance", "Hours_Studied"]])[:, 0]
                scaled_df = pd.DataFrame({
                    "Previous_Scores": df["Previous_Scores"],
                    "dimension": dimension
                })
                scaled = scaler.transform(scaled_df)

                df["Previous_Scores"] = scaled[:, 0]
                df["dimension"] = scaled[:, 1]

                df_model = df.drop(columns=["Attendance", "Hours_Studied"], errors='ignore')
                fit_columns = model.feature_names_in_
                df_model = df_model[fit_columns]

                df["Prediksi_Nilai_Ujian"] = model.predict(df_model)

                st.success("Prediksi berhasil untuk semua siswa!")
                with st.expander("Lihat Hasil Prediksi"):
                    st.dataframe(df)

                st.download_button(
                    label="Download Hasil Prediksi (CSV)",
                    data=df.to_csv(index=False).encode(),
                    file_name="hasil_prediksi.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses file:\n\n{e}")

# style
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .block-container {
            padding: 2rem 2rem;
        }
        .stRadio > div {
            flex-direction: row;
        }
        .stForm button {
            background-color: #4CAF50;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)