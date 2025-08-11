import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Prediksi Nilai Ujian Siswa / Student Performance Prediction", layout="wide")

# Text
TEXT = {
    "Indonesia": {
        "title": "Prediksi Nilai Ujian Siswa",
        "info": "Pilih metode input: **1 siswa (manual)** atau **upload data banyak siswa (.csv/.xlsx)**.",
        "lang_select": "Pilih Bahasa:",
        "input_method": "Pilih Metode Input:",
        "manual": "Manual (1 siswa)",
        "upload": "Upload File (Banyak siswa)",
        "form_title": "Formulir Data Siswa",
        "sliders": {
            "parental_involvement": "Keterlibatan orang tua (1-3)",
            "access_to_resources": "Akses ke sumber daya pendidikan (1-3)",
            "previous_scores": "Previous Scores",
            "tutoring_sessions": "Sesi Bimbingan / Minggu",
            "parent_edu": "Tingkat Pendidikan Orang Tua",
            "extracurricular": "Ikut Ekstrakurikuler?",
            "internet": "Akses Internet di Rumah?",
            "school_type": "Tipe Sekolah",
            "peer_influence": "Pengaruh Teman",
            "learning_disabilities": "Hambatan Belajar?",
            "gender": "Jenis Kelamin",
            "attendance": "Kehadiran (%)",
            "hours_studied": "Jam Belajar / Minggu",
        },
        "options": {
            "edu": ["SMA", "Kuliah", "Pascasarjana"],
            "yes_no": ["Ya", "Tidak"],
            "school": ["Swasta", "Negeri"],
            "peer": ["Negatif", "Netral", "Positif"],
            "gender": ["Laki-laki", "Perempuan"],
        },
        "predict_btn": "Prediksi Sekarang",
        "success": "Prediksi berhasil!",
        "predicted_score": "Nilai Ujian yang Diprediksi",
        "upload_title": "Upload File Data Siswa",
        "upload_info": "Unggah file .csv atau .xlsx",
        "missing_cols": "Kolom hilang dalam file:",
        "multi_success": "Prediksi berhasil untuk semua siswa!",
        "view_results": "Lihat Hasil Prediksi",
        "download_btn": "Download Hasil Prediksi (CSV)",
        "error_file": "Terjadi kesalahan saat memproses file:",
    },
    "English": {
        "title": "Student Performance Prediction",
        "info": "Choose input method: **1 student (manual)** or **upload multiple students (.csv/.xlsx)**.",
        "lang_select": "Select Language:",
        "input_method": "Select Input Method:",
        "manual": "Manual (1 student)",
        "upload": "Upload File (Multiple students)",
        "form_title": "Student Data Form",
        "sliders": {
            "parental_involvement": "Parental Involvement (1-3)",
            "access_to_resources": "access to educational resources (1-3)",
            "previous_scores": "Previous Scores",
            "tutoring_sessions": "Tutoring Sessions per Week",
            "parent_edu": "Parental Education Level",
            "extracurricular": "Join Extracurricular Activities?",
            "internet": "Internet Access at Home?",
            "school_type": "School Type",
            "peer_influence": "Peer Influence",
            "learning_disabilities": "Learning Disabilities?",
            "gender": "Gender",
            "attendance": "Attendance (%)",
            "hours_studied": "Hours Studied per Week",
        },
        "options": {
            "edu": ["High School", "College", "Postgraduate"],
            "yes_no": ["Yes", "No"],
            "school": ["Private", "Public"],
            "peer": ["Negative", "Neutral", "Positive"],
            "gender": ["Male", "Female"],
        },
        "predict_btn": "Predict Now",
        "success": "Prediction successful!",
        "predicted_score": "Predicted Exam Score",
        "upload_title": "Upload Student Data File",
        "upload_info": "Upload .csv or .xlsx file",
        "missing_cols": "Missing columns in file:",
        "multi_success": "Predictions successful for all students!",
        "view_results": "View Prediction Results",
        "download_btn": "Download Prediction Results (CSV)",
        "error_file": "Error processing file:",
    }
}

# Language
lang = st.radio(TEXT["Indonesia"]["lang_select"] + " / " + TEXT["English"]["lang_select"], ["Indonesia", "English"])

# Title and Infomation
st.markdown(f"<h1 style='color:#2c3e50;'>{TEXT[lang]['title']}</h1>", unsafe_allow_html=True)
st.info(TEXT[lang]['info'])

# input method selection
mode = st.radio(TEXT[lang]['input_method'], [TEXT[lang]['manual'], TEXT[lang]['upload']])

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

# Manual Input
if mode == TEXT[lang]['manual']:
    with st.form("prediction_form"):
        st.subheader(TEXT[lang]['form_title'])
        col1, col2, col3 = st.columns(3)
        with col1:
            Parental_Involvement = st.slider(TEXT[lang]['sliders']['parental_involvement'], 1, 3, 2)
            Access_to_Resources = st.slider(TEXT[lang]['sliders']['access_to_resources'], 1, 3, 2)
            Previous_Scores = st.number_input(TEXT[lang]['sliders']['previous_scores'], 0, 100, 70)
            Tutoring_Sessions = st.slider(TEXT[lang]['sliders']['tutoring_sessions'], 0, 5, 1)

        with col2:
            Parental_Education_Level = st.selectbox(TEXT[lang]['sliders']['parent_edu'], TEXT[lang]['options']['edu'])
            Extracurricular = st.radio(TEXT[lang]['sliders']['extracurricular'], TEXT[lang]['options']['yes_no'])
            Internet = st.radio(TEXT[lang]['sliders']['internet'], TEXT[lang]['options']['yes_no'])

        with col3:
            School_Type = st.radio(TEXT[lang]['sliders']['school_type'], TEXT[lang]['options']['school'])
            Peer_Influence = st.selectbox(TEXT[lang]['sliders']['peer_influence'], TEXT[lang]['options']['peer'])
            Learning_Disabilities = st.radio(TEXT[lang]['sliders']['learning_disabilities'], TEXT[lang]['options']['yes_no'])
            Gender = st.radio(TEXT[lang]['sliders']['gender'], TEXT[lang]['options']['gender'])

        Attendance = st.slider(TEXT[lang]['sliders']['attendance'], 0, 100, 50)
        Hours_Studied = st.slider(TEXT[lang]['sliders']['hours_studied'], 0, 50, 15)

        submitted = st.form_submit_button(TEXT[lang]['predict_btn'])

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
                'Parental_Education_Level': {e: i+1 for i, e in enumerate(TEXT[lang]['options']['edu'])}[Parental_Education_Level],
                'Extracurricular_Activities_No': Extracurricular == TEXT[lang]['options']['yes_no'][1],
                'Extracurricular_Activities_Yes': Extracurricular == TEXT[lang]['options']['yes_no'][0],
                'Internet_Access_No': Internet == TEXT[lang]['options']['yes_no'][1],
                'Internet_Access_Yes': Internet == TEXT[lang]['options']['yes_no'][0],
                'School_Type_Private': School_Type == TEXT[lang]['options']['school'][0],
                'School_Type_Public': School_Type == TEXT[lang]['options']['school'][1],
                'Peer_Influence_Negative': Peer_Influence == TEXT[lang]['options']['peer'][0],
                'Peer_Influence_Neutral': Peer_Influence == TEXT[lang]['options']['peer'][1],
                'Peer_Influence_Positive': Peer_Influence == TEXT[lang]['options']['peer'][2],
                'Learning_Disabilities_No': Learning_Disabilities == TEXT[lang]['options']['yes_no'][1],
                'Learning_Disabilities_Yes': Learning_Disabilities == TEXT[lang]['options']['yes_no'][0],
                'Gender_Female': Gender == TEXT[lang]['options']['gender'][1],
                'Gender_Male': Gender == TEXT[lang]['options']['gender'][0],
                'dimension': scaled[1]
            }

            df_input = pd.DataFrame([data])
            pred = model.predict(df_input)[0]

            st.success(TEXT[lang]['success'])
            st.metric(label=TEXT[lang]['predicted_score'], value=f"{pred:.2f}")

# Upload File
else:
    st.subheader(TEXT[lang]['upload_title'])
    uploaded_file = st.file_uploader(TEXT[lang]['upload_info'], type=["csv", "xlsx"])

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
                st.error(f"{TEXT[lang]['missing_cols']} {missing}")
            else:
                df["Parental_Education_Level"] = df["Parental_Education_Level"].map({
                    "High School": 1, "College": 2, "Postgraduate": 3,
                    "SMA": 1, "Kuliah": 2, "Pascasarjana": 3
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

                df["Predicted_Exam_Score"] = model.predict(df_model)

                st.success(TEXT[lang]['multi_success'])
                with st.expander(TEXT[lang]['view_results']):
                    st.dataframe(df)

                st.download_button(
                    label=TEXT[lang]['download_btn'],
                    data=df.to_csv(index=False).encode(),
                    file_name="hasil_prediksi.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"{TEXT[lang]['error_file']}\n\n{e}")