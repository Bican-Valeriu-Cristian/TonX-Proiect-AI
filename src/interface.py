import streamlit as st
import pandas as pd
import time
import numpy as np
import os
from src.metrics_logger import MetricsLogger

# Importăm predictorul real
from predict import TonXPredictor

# --- CONFIGURARE PAGINĂ ---
st.set_page_config(
    page_title="TonX Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FUNCȚII CACHED (Pentru performanță) ---
@st.cache_resource
def load_predictor(task_name):
    """
    Încarcă modelul în cache pentru a nu-l reîncărca la fiecare interacțiune.
    """
    return TonXPredictor(task_name)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1055/1055644.png", width=50)
    st.title("TonX AI")
    st.markdown("---")
    
    menu = st.radio("Meniu", ["Analiză Mesaj", "Performanță Model", "Despre Proiect"])
    
    st.markdown("---")
    
    # Selector model activ doar pentru tab-ul de analiză
    selected_task = "sentiment" # default
    if menu == "Analiză Mesaj":
        st.subheader("⚙️ Configurare Model")
        selected_task = st.selectbox(
            "Alege Modelul Activ:", 
            ["sentiment", "category"],
            format_func=lambda x: x.capitalize()
        )
        st.info(f"Model selectat: **{selected_task.capitalize()}**")

    st.markdown("---")
    st.caption("TonX Team v1.0")

# --- PAGINA: ANALIZA MESAJ ---
if menu == "Analiză Mesaj":
    st.title(f"Analiză: {selected_task.capitalize()}")
    st.markdown("Introduceți textul mesajului mai jos pentru a fi procesat de modelul AI.")

    # Încărcăm modelul selectat
    predictor = load_predictor(selected_task)

    # Verificăm dacă modelul este gata
    if not predictor.ready:
        st.error(f"⚠️ Modelul '{selected_task}' nu a fost găsit sau nu este antrenat.")
        st.warning(f"Te rugăm să rulezi `python train.py --task {selected_task}` mai întâi.")
    else:
        col1, col2 = st.columns([2, 1])

        with col1:
            user_input = st.text_area("Mesaj de analizat", height=150, placeholder="Scrie aici mesajul...")
            analyze_btn = st.button("🔍 Analizează Mesajul", type="primary")

        if analyze_btn and user_input:
            with st.spinner('Procesare text cu DistilBERT...'):
                # Apelăm predicția reală
                start_time = time.time()
                label, score, class_idx = predictor.predict(user_input)
                duration = time.time() - start_time
            
            st.divider()
            st.subheader("Rezultate Analiză")
            
            # Determinăm culoarea în funcție de scor sau label (doar estetic)
            color_delta = "off"
            if score > 0.8: color_delta = "normal"
            if score < 0.5: color_delta = "inverse"

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric(label="Clasă Predicție", value=label)
            with m2:
                st.metric(label="Încredere (Confidence)", value=f"{score*100:.2f}%", delta=color_delta)
            with m3:
                st.metric(label="Timp Procesare", value=f"{duration:.4f} sec")
                
            st.caption(f"Index intern clasă: {class_idx}")
            
            # Progress bar colorat (custom hack sau standard)
            st.progress(score, text="Nivel de certitudine al modelului")
            
            # Afișăm JSON raw pentru debug/integrare API
            with st.expander("Vezi răspuns JSON (API format)"):
                st.json({
                    "text": user_input,
                    "prediction": label,
                    "confidence": score,
                    "model": selected_task,
                    "timestamp": time.time()
                })

# --- PAGINA: PERFORMANTA MODEL ---
elif menu == "Performanță Model":
    st.title("Metrice de Performanță")
    st.markdown("Evaluarea completă a modelului pe seturile de validare și test.")

    logger = MetricsLogger()
    
    # Selector pentru task în pagina de performanță
    task_choice = st.selectbox("Selectează Task-ul de vizualizat", ["sentiment", "category"])
    
    runs = logger.list_runs(task_choice)
    
    if not runs:
        st.warning("Nu există rulări salvate pentru acest task.")
    else:
        run_id = st.selectbox("Alege versiunea (run)", ["latest"] + runs)
        metrics = logger.load_metrics(task_choice, None if run_id == "latest" else run_id)

        if metrics:
            # Extragem datele
            final_metrics = metrics.get('final_metrics', {})
            test_results = metrics.get('test_results', None)
            train_history = metrics.get('train_history', {})
            val_history = metrics.get('val_history', {})
            class_metrics = metrics.get('class_metrics', {})
            config = metrics.get('config', {})
            
            # 1. KPI-uri principale
            st.subheader("📈 Metrici Globale")
            c1, c2, c3, c4 = st.columns(4)
            
            acc = 0
            if test_results:
                acc = test_results.get('accuracy', 0)
                f1 = test_results.get('f1_score_macro', 0)
            else:
                acc = final_metrics.get('val_accuracy', 0)
                f1 = 0

            c1.metric("Acuratețe", f"{acc*100:.2f}%")
            c2.metric("Loss (Validare)", f"{final_metrics.get('val_loss', 0):.4f}")
            c3.metric("F1 Score", f"{f1:.4f}")
            c4.metric("Epoci", config.get('epochs', '-'))

            st.divider()

            # 2. Grafice
            if train_history:
                st.subheader("📊 Evoluție Antrenare")
                tab1, tab2 = st.tabs(["Acuratețe", "Loss"])
                
                epochs_range = range(1, len(train_history['accuracy']) + 1)
                df_acc = pd.DataFrame({
                    "Epoch": list(epochs_range),
                    "Train": train_history['accuracy'],
                    "Validation": val_history['accuracy']
                }).set_index("Epoch")
                
                df_loss = pd.DataFrame({
                    "Epoch": list(epochs_range),
                    "Train": train_history['loss'],
                    "Validation": val_history['loss']
                }).set_index("Epoch")

                with tab1:
                    st.line_chart(df_acc, color=["#36a2eb", "#ff6384"])
                with tab2:
                    st.line_chart(df_loss, color=["#36a2eb", "#ff6384"])

            # 3. Matrice de confuzie și Clase
            if test_results and 'confusion_matrix' in test_results:
                st.divider()
                col_conf, col_class = st.columns([1, 1])
                
                with col_conf:
                    st.subheader("Matrice de Confuzie")
                    cm = np.array(test_results['confusion_matrix'])
                    class_names = config.get('class_names', [str(i) for i in range(len(cm))])
                    
                    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
                    st.dataframe(df_cm.style.background_gradient(cmap='Blues'))

                with col_class:
                    st.subheader("Top Performanță per Clasă")
                    if class_metrics:
                        # Pregătim datele pentru grafic
                        data = []
                        for k, v in class_metrics.items():
                            data.append({"Clasa": k, "F1": v['f1-score']})
                        df_cls = pd.DataFrame(data).set_index("Clasa")
                        st.bar_chart(df_cls)

# --- PAGINA: DESPRE ---
elif menu == "Despre Proiect":
    st.header("Despre TonX")
    st.info("""
    Această aplicație utilizează un model **DistilBERT** fine-tuned pentru a clasifica textul.
    
    Arhitectura:
    - **Backend:** PyTorch + Transformers
    - **Frontend:** Streamlit
    - **Model:** DistilBERT Base Uncased
    """)
    st.markdown("### Cum funcționează?")
    st.markdown("1. Textul este curățat (eliminare linkuri, caractere speciale).")
    st.markdown("2. Este tokenizat folosind vocabularul DistilBERT.")
    st.markdown("3. Modelul prezice probabilitățile pentru fiecare clasă.")
