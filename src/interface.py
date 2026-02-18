import streamlit as st
import pandas as pd
import time
import numpy as np
from preprocessing import simple_clean
from src.metrics_logger import MetricsLogger

# --- CONFIGURARE PAGINĂ ---
st.set_page_config(
    page_title="TonX Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FUNCȚIE MOCK PENTRU PREDICȚIE ---
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
    st.markdown("""TonX – Proiect AI pentru analiza tonului și clasificarea mesajelor 
Scopul proiectului TonX este de a antrena și integra un model de inteligență artificială care poate primi mesaje (emailuri sau texte) și poate determina tonul acestora (pozitiv, negativ sau neutru), precum și categoria din care fac parte (suport, ofertă, cerere, reclamație etc.).  
 
Obiective principale:\n
 -Detectarea tonului mesajului: pozitiv, negativ sau neutru. \n
 -Clasificarea mesajului în categorii predefinite (suport, ofertă, cerere, reclamație, follow-up). \n
 -Evidențierea frazelor cheie care influențează clasificarea (explicabilitate minimă). \n
 -Generarea unui răspuns standardizat în format JSON pentru integrare ușoară. \n
Pentru acesta utilizam dataset-uri publice (Sentiment140, twitter_training.csv, jason23322/high-accuracy-email-classifier) si modelul DistilBERT.
""")