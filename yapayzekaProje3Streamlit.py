import numpy as np
import pandas as pd
import streamlit as st

# Sayfa Ayarları
st.set_page_config(
    page_title="fetal health",
    page_icon="https://miro.medium.com/v2/resize:fit:2400/1*rGi8_JUoGX0L3W6nivmIAg@2x.png",
    menu_items={
        'Get Help': 'mailto:ata.ozarslan@istdsa.com',
        "About": "For More Information\n" + "https://github.com/ataozarslan/DSNov22"
    }
)

# Başlık Ekleme
st.title("Fetal Health Project")

# Markdown Oluşturma
st.markdown("Cardiotocography (CTG) is used during pregnancy to monitor fetal heart rate and uterine contractions. It is monitor fetal well-being and allows early detection of fetal distress.")

st.markdown("CTG interpretation helps in determining if the pregnancy is high or low risk. An abnormal CTG may indicate the need for further investigations and potential intervention.")

st.markdown("In this project, I will create a model to classify the outcome of Cardiotocogram test to ensure the well being of the fetus.")

# Resim Ekleme
st.image("/Users/niluferceylan/Desktop/yapay zeka proje3/Unknown.png")


# Header Ekleme
st.header("Data Dictionary")



st.markdown("- **baseline_value**: FHR baseline (beats per minute)")
st.markdown("- **accelerations**: Number of accelerations per second")
st.markdown("- **fetal_movement**: Number of fetal movements per second")
st.markdown("- **uterine_contractions**: Number of uterine contractions per second")
st.markdown("- **light_decelerations**: Number of light decelerations per second")
st.markdown("- **severe_decelerations**: Number of severe decelerations per second")
st.markdown("- **prolongued_decelerations**: Number of prolonged decelerations per second")
st.markdown("- **abnormal_short_term_variability**: Percentage of time with abnormal short term variability")
st.markdown("- **mean_value_of_short_term_variability**: Mean value of short term variability")
st.markdown("- **percentage_of_time_with_abnormal_long_term_variability**: Percentage of time with abnormal long term variability ")
st.markdown("- **mean_value_of_long_term_variability**:  Mean value of long term variability")
st.markdown("- **histogram_width**: Width of FHR histogram")
st.markdown("- **histogram_min**: Minimum (low frequency) of FHR histogram")
st.markdown("- **histogram_max**: Maximum (high frequency) of FHR histogram")
st.markdown("- **histogram_number_of_peaks**: Number of histogram peaks")
st.markdown("- **histogram_number_of_zeroes**: Number of histogram zeros")
st.markdown("- **histogram_mode**: Histogram mode")
st.markdown("- **histogram_mean**: Histogram mean")
st.markdown("- **histogram_median**: Histogram median")
st.markdown("- **histogram_variance**: Histogram variance")
st.markdown("- **histogram_tendency**: Histogram tendency")
st.markdown("- **Target**")
st.markdown("- **fetal_health**: Tagged as 1 (Normal), 2 (Suspect) and 3 (Pathological)")

# Pandasla veri setini okuyalım
df = pd.read_pickle("/Users/niluferceylan/Desktop/yapay zeka proje3/train_df.pkl")

# Küçük bir düzenleme :)


# Tablo Ekleme
st.dataframe(df.sample(5))

#---------------------------------------------------------------------------------------------------------------------

# Sidebarda Markdown Oluşturma
st.sidebar.markdown("**Choose** the features below to see the result!")

# Sidebarda Kullanıcıdan Girdileri Alma
name = st.sidebar.text_input("Name", help="Please capitalize the first letter of your name!")
surname = st.sidebar.text_input("Surname", help="Please capitalize the first letter of your surname!")
prolongued_decelerations = st.sidebar.number_input("prolongued_decelerations")
abnormal_short_term_variability = st.sidebar.number_input("abnormal_short_term_variability")
percentage_of_time_with_abnormal_long_term_variability = st.sidebar.number_input("percentage_of_time_with_abnormal_long_term_variability")
histogram_variance = st.sidebar.number_input("histogram_variance")

#---------------------------------------------------------------------------------------------------------------------

# Pickle kütüphanesi kullanarak eğitilen modelin tekrardan kullanılması
from joblib import load

logreg_model = load('logreg_model.pkl')

input_df = pd.DataFrame({
    'prolongued_decelerations': [prolongued_decelerations],
    'abnormal_short_term_variability': [abnormal_short_term_variability],
    'percentage_of_time_with_abnormal_long_term_variability': [percentage_of_time_with_abnormal_long_term_variability],
    'histogram_variance': [histogram_variance]
})

# Verilerimizi ölçeklendirmeyi unutmuyoruz!
std_scale = load('scaler.pkl')
scaled_input_df = std_scale.transform(input_df)


pred = logreg_model.predict(scaled_input_df)
pred_probability = np.round(logreg_model.predict_proba(scaled_input_df), 2)

#---------------------------------------------------------------------------------------------------------------------

st.header("Results")

# Sonuç Ekranı
if st.sidebar.button("Submit"):

    # Info mesajı oluşturma
    st.info("You can find the result below.")

    # Sorgulama zamanına ilişkin bilgileri elde etme
    from datetime import date, datetime

    today = date.today()
    time = datetime.now().strftime("%H:%M:%S")

    # Sonuçları Görüntülemek için DataFrame
    results_df = pd.DataFrame({
    'Name': [name],
    'Surname': [surname],
    'Date': [today],
    'Time': [time],
    'prolongued_decelerations': [prolongued_decelerations],
    'abnormal_short_term_variability': [abnormal_short_term_variability],
    'percentage_of_time_with_abnormal_long_term_variability': [percentage_of_time_with_abnormal_long_term_variability],
    'histogram_variance': [histogram_variance],
    'Prediction': [pred],
    'Nomal Probability': [pred_probability[:,0]],
    'Suspect Probability': [pred_probability[:,1]],
    'Pathological Probability': [pred_probability[:,2]],
    
    })

    results_df["Prediction"] = results_df["Prediction"].apply(lambda x: str(x).replace("1","Normal"))
    results_df["Prediction"] = results_df["Prediction"].apply(lambda x: str(x).replace("2","Suspect"))
    results_df["Prediction"] = results_df["Prediction"].apply(lambda x: str(x).replace("3","Pathological"))

    st.table(results_df)

    if pred == 1:
        st.image("/Users/niluferceylan/Desktop/yapay zeka proje3/pngtree-normal-icon-png-image_6630696.jpg")
    elif pred ==2:
        st.image("/Users/niluferceylan/Desktop/yapay zeka proje3/360_F_79389077_B0Krv6zXPtMsqrcmKqacbvNYJpnJfsIh.jpg")
    else:
         st.image("/Users/niluferceylan/Desktop/yapay zeka proje3/images.png")
       
else:
    st.markdown("Please click the *Submit Button*!")