import streamlit as st
import pandas as pd
import time
import numpy as np
import os
from src.metrics_logger import MetricsLogger

# Importăm predictorul real
from predict import TonXPredictor
from preprocessing import simple_clean
from src.metrics_logger import MetricsLogger

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
# --- FUNCȚIE MOCK PENTRU PREDICȚIE ---
def get_prediction(text):
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
# --- PAGINA NOUA: PERFORMANTA MODEL (Cu date reale din JSON) ---
elif menu == "Performanță Model":
    st.title("Metrice de Performanță")
    st.markdown("Evaluarea completă a modelului pe seturile de validare și test.")

    # Încărcăm metricile
    logger = MetricsLogger()
    
    # Selector pentru task
    task_choice = st.selectbox("Selectează Task-ul", ["sentiment", "category"])
    
    runs = logger.list_runs(task_choice)
    run_id = st.selectbox("Alege versiunea (run)", ["latest"] + runs)

    metrics = logger.load_metrics(task_choice, None if run_id == "latest" else run_id)

    
    if metrics is None:
        st.error(f"⚠️ Nu există metrici salvate pentru task-ul '{task_choice}'.")
        st.info("💡 **Pași necesari:**")
        st.code(f"python train.py --task {task_choice}", language="bash")
        st.code(f"python test.py --task {task_choice}", language="bash")
    else:
        # Extragem datele din JSON
        final_metrics = metrics.get('final_metrics', {})
        test_results = metrics.get('test_results', None)
        train_history = metrics.get('train_history', {})
        val_history = metrics.get('val_history', {})
        class_metrics = metrics.get('class_metrics', {})
        config = metrics.get('config', {})
        
        # ==== SECȚIUNEA 1: METRICI GLOBALE ====
        st.subheader("📈 Metrici Globale")
        
        # Verificăm dacă avem metrici de test
        if test_results:
            col1, col2, col3, col4 = st.columns(4)
            
            test_acc = test_results.get('accuracy', 0) * 100
            test_f1 = test_results.get('f1_score_macro', 0)
            test_precision = test_results.get('precision_macro', 0)
            test_recall = test_results.get('recall_macro', 0)
            
            col1.metric("Acuratețe Test", f"{test_acc:.2f}%")
            col2.metric("F1-Score (Macro)", f"{test_f1:.4f}")
            col3.metric("Precision (Macro)", f"{test_precision:.4f}")
            col4.metric("Recall (Macro)", f"{test_recall:.4f}")
            
            st.info(f"📊 Testat pe {test_results.get('num_test_samples', 'N/A')} mostre")
        else:
            # Afișăm doar metricile de validare
            col1, col2, col3 = st.columns(3)
            
            val_acc = final_metrics.get('val_accuracy', 0) * 100
            train_loss = final_metrics.get('train_loss', 0)
            
            col1.metric("Acuratețe Validare", f"{val_acc:.1f}%")
            col2.metric("Training Loss", f"{train_loss:.4f}")
            col3.metric("Epoci", config.get('epochs', 'N/A'))
            
            st.warning("⚠️ Metricile de test (F1, Precision, Recall) nu sunt disponibile. Rulează `test.py` pentru a le genera.")

        st.divider()

        # ==== SECȚIUNEA 2: EVOLUȚIA ANTRENĂRII ====
        if train_history and val_history:
            st.subheader("📊 Evoluția Antrenării")
            
            tab1, tab2 = st.tabs(["Acuratețe", "Loss"])
            
            with tab1:
                epochs = list(range(1, len(train_history['accuracy']) + 1))
                chart_data_acc = pd.DataFrame({
                    'Epoch': epochs + epochs,
                    'Acuratețe': train_history['accuracy'] + val_history['accuracy'],
                    'Set': ['Train'] * len(epochs) + ['Validation'] * len(epochs)
                })
                
                st.line_chart(chart_data_acc, x='Epoch', y='Acuratețe', color='Set')
                st.caption("Evoluția acurateței pe setul de antrenare vs validare")
            
            with tab2:
                chart_data_loss = pd.DataFrame({
                    'Epoch': epochs + epochs,
                    'Loss': train_history['loss'] + val_history['loss'],
                    'Set': ['Train'] * len(epochs) + ['Validation'] * len(epochs)
                })
                
                st.line_chart(chart_data_loss, x='Epoch', y='Loss', color='Set')
                st.caption("Evoluția loss-ului pe setul de antrenare vs validare")

        st.divider()

        # ==== SECȚIUNEA 3: METRICI PER CLASĂ ====
        st.subheader("🎯 Performanță per Clasă")
        
        if class_metrics:
            # Creăm un DataFrame pentru afișare
            class_data = []
            for class_name, metrics_dict in class_metrics.items():
                class_data.append({
                    'Clasă': class_name,
                    'Precision': metrics_dict['precision'],
                    'Recall': metrics_dict['recall'],
                    'F1-Score': metrics_dict['f1-score'],
                    'Support': metrics_dict['support']
                })
            
            df_classes = pd.DataFrame(class_data)
            
            # Grafic cu F1-Score per clasă
            st.bar_chart(df_classes, x='Clasă', y='F1-Score', color='Clasă')
            
            # Tabel detaliat
            st.dataframe(
                df_classes.style.format({
                    'Precision': '{:.4f}',
                    'Recall': '{:.4f}',
                    'F1-Score': '{:.4f}',
                    'Support': '{:.0f}'
                }).background_gradient(subset=['F1-Score'], cmap='RdYlGn'),
                use_container_width=True
            )
            
            st.caption("""
            **Interpretare Metrici:**
            - **Precision**: Din toate predicțiile pentru această clasă, câte sunt corecte?
            - **Recall**: Din toate exemplele reale ale clasei, câte au fost detectate?
            - **F1-Score**: Media armonică între Precision și Recall (echilibru)
            - **Support**: Numărul de exemple reale din setul de test
            """)
        else:
            st.info("Metricile detaliate per clasă vor fi disponibile după rularea `test.py`")

        st.divider()

        # ==== SECȚIUNEA 4: MATRICE DE CONFUZIE ====
        if test_results and 'confusion_matrix' in test_results:
            st.subheader("🔢 Matrice de Confuzie")
            
            conf_matrix = np.array(test_results['confusion_matrix'])
            class_names = config.get('class_names', [f'Clasa_{i}' for i in range(len(conf_matrix))])
            
            # Creăm un heatmap folosind Streamlit
            df_conf = pd.DataFrame(
                conf_matrix,
                index=class_names,
                columns=class_names
            )
            
            st.dataframe(
                df_conf.style.background_gradient(cmap='Blues'),
                use_container_width=True
            )
            
            st.caption("""
            **Cum se citește matricea:**
            - Rândurile = Clase Reale
            - Coloanele = Clase Prezise
            - Diagonala = Predicții Corecte
            - Valorile în afara diagonalei = Confuzii între clase
            """)

        st.divider()

        # ==== SECȚIUNEA 5: CONFIGURARE ====
        with st.expander("⚙️ Detalii Configurare Antrenare"):
            col_cfg1, col_cfg2 = st.columns(2)
            
            with col_cfg1:
                st.metric("Learning Rate", config.get('learning_rate', 'N/A'))
                st.metric("Batch Size", config.get('batch_size', 'N/A'))
            
            with col_cfg2:
                st.metric("Număr Clase", config.get('num_classes', 'N/A'))
                st.metric("Epoci Totale", config.get('epochs', 'N/A'))
            
            st.json(config)
        
        # Timestamp
        last_updated = metrics.get('last_updated', 'N/A')
        st.caption(f"Ultima actualizare: {last_updated}")

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