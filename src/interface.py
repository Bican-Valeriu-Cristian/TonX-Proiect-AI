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
def load_predictor(task_name, is_raw=False):
    """
    Încarcă modelul în cache pentru a nu-l reîncărca la fiecare interacțiune.
    """
    return TonXPredictor(task_name, is_raw=is_raw)

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
            ["sentiment", "category","Sentiment & Category"],
            format_func=lambda x: x.capitalize()
        )
        compare_raw = st.checkbox("Folosește modelul RAW (neantrenat)")
        if compare_raw:
            st.info("Se va afișa o comparație **Antrenat vs RAW** pentru fiecare task.")
        else:
            st.info(f"Model selectat: **{selected_task.capitalize()}**")

    st.markdown("---")
    st.caption("TonX Team v1.0")

# --- PAGINA: ANALIZA MESAJ ---
if menu == "Analiză Mesaj":
    st.title(f"Analiză: {selected_task.capitalize()}")
    st.markdown("Introduceți textul mesajului mai jos pentru a fi procesat de modelul AI.")
    tasks_to_run = ["sentiment", "category"] if selected_task == "Sentiment & Category" else [selected_task]
    # Încărcăm modelul selectat
    predictors = {t: load_predictor(t) for t in tasks_to_run}
    raw_predictors = {}
    if compare_raw:
        raw_predictors = {t: load_predictor(t, is_raw=True) for t in tasks_to_run}
    # Verificăm dacă modelul este gata
    all_ready = all(p.ready for p in predictors.values())

    if not all_ready:
        for t, p in predictors.items():
            if not p.ready:
                st.error(f"⚠️ Modelul '{t}' nu a fost găsit sau nu este antrenat.")
        st.warning("Te rugăm să rulezi antrenarea pentru modelele lipsă.")
    else:
        col1, col2 = st.columns([2, 1])

        with col1:
            user_input = st.text_area("Mesaj de analizat", height=150, placeholder="Scrie aici mesajul...")
            analyze_btn = st.button("🔍 Analizează Mesajul", type="primary")
        results = {}
        raw_results = {}
        if analyze_btn and user_input:
            with st.spinner('Procesare text cu DistilBERT...'):
                for t_name, p_obj in predictors.items():
                    start_time = time.time()
                    label, score, class_idx = p_obj.predict(user_input)
                    results[t_name] = {
                        "label": label, 
                        "score": score, 
                        "duration": time.time() - start_time, 
                        "idx": class_idx
                    }
                if compare_raw:
                    for t_name, p_obj in raw_predictors.items():
                        start = time.time()
                        label, score, class_idx = p_obj.predict(user_input)
                        raw_results[t_name] = {
                            "label": label,
                            "score": score,
                            "duration": time.time() - start,
                            "idx": class_idx
                        }

            st.divider()
            st.subheader("Rezultate Analiză")
            
            for t_name in tasks_to_run:
                if compare_raw:
                    st.markdown(f"### 📊 Task: **{t_name.capitalize()}**")
                    col_trained, col_raw = st.columns(2)

                    with col_trained:
                        res = results[t_name]
                        st.markdown("#### ✅ Model Antrenat")
                        color_delta = "off"
                        if res["score"] > 0.8: color_delta = "normal"
                        if res["score"] < 0.5: color_delta = "inverse"
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Clasă", res["label"])
                        m2.metric("Încredere", f"{res['score']*100:.2f}%", delta=color_delta)
                        m3.metric("Timp", f"{res['duration']:.4f}s")
                        st.progress(res["score"], text="Certitudine (antrenat)")

                    with col_raw:
                        res_raw = raw_results.get(t_name, {})
                        st.markdown("#### 🔬 Model RAW (neantrenat)")
                        color_delta = "off"
                        if res_raw.get("score", 0) > 0.8: color_delta = "normal"
                        if res_raw.get("score", 0) < 0.5: color_delta = "inverse"
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Clasă", res_raw.get("label", "-"))
                        m2.metric("Încredere", f"{res_raw.get('score', 0)*100:.2f}%", delta=color_delta)
                        m3.metric("Timp", f"{res_raw.get('duration', 0):.4f}s")
                        st.progress(res_raw.get("score", 0), text="Certitudine (RAW)")
                    
                    st.divider()
                else:
                    res = results[t_name]
                    st.markdown(f"### Model: **{t_name.capitalize()}**")
                    color_delta = "off"
                    if res["score"] > 0.8: color_delta = "normal"
                    if res["score"] < 0.5: color_delta = "inverse"
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Clasă", res["label"])
                    m2.metric("Încredere", f"{res['score']*100:.2f}%", delta=color_delta)
                    m3.metric("Timp", f"{res['duration']:.4f}s")
                    st.progress(res["score"], text=f"Certitudine {t_name}")
                    st.caption(f"Index intern: {res['idx']}")

            with st.expander("Vezi răspuns JSON (API format)"):
                st.json({
                    "text": user_input,
                    "results": results,
                    "raw_results": raw_results if compare_raw else "N/A",
                    "model": selected_task,
                    "timestamp": time.time()
                })

# --- PAGINA: PERFORMANTA MODEL ---
elif menu == "Performanță Model":
    st.title("Metrice de Performanță")
    st.markdown("Evaluarea completă a modelului pe seturile de validare și test.")

    logger = MetricsLogger()
    task_choice = st.selectbox("Selectează Task-ul de vizualizat", ["sentiment", "category"])
    runs = logger.list_runs(task_choice)
    
    if not runs:
        st.warning("Nu există rulări salvate pentru acest task.")
    else:
        run_id = st.selectbox("Alege versiunea (run)", ["latest"] + runs)
        metrics = logger.load_metrics(task_choice, None if run_id == "latest" else run_id)

        if metrics:
            final_metrics = metrics.get('final_metrics', {})
            test_results = metrics.get('test_results', {})
            config = metrics.get('config', {})
            
            # --- 1. KPI-uri principale (Layout pe două rânduri pentru claritate) ---
            st.subheader("📈 Metrici de Test (General)")
            
            # Rândul 1: Cele mai importante metrici
            c1, c2, c3, c4 = st.columns(4)
            
            # Extragem valorile cu fallback la final_metrics dacă test_results lipsește
            acc = test_results.get('accuracy', final_metrics.get('val_accuracy', 0))
            f1_macro = test_results.get('f1_score_macro', 0)
            prec_macro = test_results.get('precision_macro', 0)
            rec_macro = test_results.get('recall_macro', 0)

            c1.metric("Acuratețe", f"{acc*100:.2f}%")
            c2.metric("F1 Score (Macro)", f"{f1_macro:.4f}")
            c3.metric("Precision (Macro)", f"{prec_macro:.4f}")
            c4.metric("Recall (Macro)", f"{rec_macro:.4f}")

            # Rândul 2: Detalii antrenare
            st.markdown("#### Detalii Antrenare")
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Loss (Validation)", f"{final_metrics.get('val_loss', 0):.4f}")
            d2.metric("Epoci Totale", config.get('epochs', '-'))
            d3.metric("Batch Size", config.get('batch_size', '-'))
            d4.metric("Learning Rate", config.get('learning_rate', '-'))

            st.divider()

            # --- 2. Tabel Detaliat per Clasă ---
            st.subheader("📋 Detalii per Clasă")
            class_metrics = metrics.get('class_metrics', {})
            if class_metrics:
                # Transformăm dicționarul într-un DataFrame pentru afișare
                df_metrics = pd.DataFrame(class_metrics).transpose()
                # Excludem rândurile de suport/total dacă nu sunt necesare
                st.table(df_metrics.style.format("{:.4f}").background_gradient(cmap='Greens', subset=['f1-score', 'precision', 'recall']))
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
