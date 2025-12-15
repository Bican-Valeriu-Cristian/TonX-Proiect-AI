import streamlit as st
import pandas as pd
import time
import numpy as np
from preprocessing import simple_clean

# --- CONFIGURARE PAGINĂ ---
st.set_page_config(
    page_title="TonX Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FUNCTIE MOCK PENTRU PREDICTIE ---
def get_prediction(text):
    """
    Simulare model AI.
    """
    clean_text = simple_clean(text)
    if "super" in clean_text or "bun" in clean_text:
        return "Pozitiv", 0.92, "green"
    elif "rau" in clean_text or "groaznic" in clean_text:
        return "Negativ", 0.85, "red"
    else:
        return "Neutru", 0.65, "gray"

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1055/1055644.png", width=50)
    st.title("TonX")
    st.markdown("---")
    
    # Am schimbat optiunea a doua in "Performanță Model"
    menu = st.radio("Meniu", ["Analiză Mesaj", "Performanță Model", "Despre Proiect"])
    
    st.markdown("---")
    st.info("Status Sistem: **Online** 🟢")
    st.caption("TonX Team")

# --- PAGINA: ANALIZA MESAJ ---
if menu == "Analiză Mesaj":
    st.title("Analiză Tonalitate & Categorie")
    st.markdown("Introduceți textul mesajului mai jos pentru a detecta sentimentul predominant.")

    col1, col2 = st.columns([2, 1])

    with col1:
        user_input = st.text_area("Mesaj de analizat", height=150, placeholder="Ex: Serviciile au fost excelente...")
        analyze_btn = st.button("🔍 Analizează Mesajul", type="primary")

    if analyze_btn and user_input:
        with st.spinner('Procesare text cu modelul TonX AI...'):
            time.sleep(0.8) 
            label, score, color = get_prediction(user_input)
        
        st.divider()
        st.subheader("Rezultate Analiză")
        
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(label="Sentiment Detectat", value=label)
        with m2:
            st.metric(label="Încredere Model", value=f"{score*100:.1f}%")
        with m3:
            st.metric(label="Categorie", value="Feedback Client")
            
        st.caption(f"Scor de probabilitate pentru clasa: {label}")
        st.progress(score, text=f"{score*100:.0f}%")

# --- PAGINA NOUA: PERFORMANTA MODEL (Diagrama) ---
elif menu == "Performanță Model":
    st.title("Metrice de Performanță")
    st.markdown("Evaluarea curentă a modelului pe setul de validare.")

    # 1. Metrice Globale (Top Level)
    # Acestea sunt date "dummy" care arata bine. Le vei inlocui cu date reale din antrenare mai tarziu.
    col_acc, col_f1, col_loss = st.columns(3)
    col_acc.metric("Acuratețe Globală", "87.5%", "+2.1%")
    col_f1.metric("F1-Score", "0.84", "+0.05")
    col_loss.metric("Training Loss", "0.12", "-0.03")

    st.divider()

    # 2. Diagrama Performanta pe Clase (Bar Chart)
    st.subheader("Acuratețe per Clasă (Sentiment)")
    
    # Cream un DataFrame mock pentru grafic
    # Aici simulam cat de bine recunoaste modelul fiecare emotie
    chart_data = pd.DataFrame({
        "Sentiment": ["Negativ", "Neutru", "Pozitiv"],
        "Acuratețe (%)": [82, 65, 94],  # Valori procentuale
        "Nr. Exemple": [1200, 850, 1300]
    })

    # Afisam graficul folosind chart-ul nativ Streamlit (foarte simplu si curat)
    st.bar_chart(
        chart_data, 
        x="Sentiment", 
        y="Acuratețe (%)", 
        color="Sentiment", # Coloreaza diferit fiecare bara
        horizontal=False
    )
    
    # Explicatie Business
    st.caption("""
    **Interpretare Grafic:**
    - **Pozitiv:** Modelul performează excelent (94%), având multe exemple clare.
    - **Negativ:** Performanță bună (82%), dar uneori confundă sarcasmul.
    - **Neutru:** Cea mai dificilă clasă (65%), necesită mai multe date de antrenare.
    """)

# --- PAGINA: DESPRE ---
elif menu == "Despre Proiect":
    st.header("Despre TonX")
    st.markdown("""TonX – Proiect AI pentru analiza tonului și clasificarea mesajelor 
Scopul proiectului TonX este de a antrena și integra un model de inteligență artificială care poate primi mesaje (emailuri sau texte) și poate determina tonul acestora (pozitiv, negativ sau neutru), precum și categoria din care fac parte (suport, ofertă, cerere, reclamație etc.).  
 
Obiective principale:\n
 -Detectarea tonului mesajului: pozitiv, negativ sau neutru. \n
 -Clasificarea mesajului în categorii predefinite (suport, ofertă, cerere, reclamație, follow-up). \n
 -Evidențierea frazelor cheie care influențează clasificarea (explicabilitate minimă). \n
 -Generarea unui răspuns standardizat în format JSON pentru integrare ușoară. \n
Pentru acesta utilizam dataset-uri publice (Sentiment140, twitter_training.csv, jason23322/high-accuracy-email-classifier) si modelul DistilBERT.
""")