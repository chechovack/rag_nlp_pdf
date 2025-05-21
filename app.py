from sentence_transformers import SentenceTransformer
from transformers import pipeline
import torch
import pandas as pd
import streamlit as st
from deep_translator import GoogleTranslator
import faiss
import torch
from sentence_transformers import SentenceTransformer

import json

df2 = pd.read_csv("D:/CHECHO/Codigo_Prueba_SegurosBolivar/bolivar_parte2_final-env/df2_final_con_embeddings.csv")

# Convertir la columna 'embedding' desde string JSON a lista de floats
df2['embedding'] = df2['embedding'].apply(json.loads)



# Parsear embeddings a tensores
def parse_embedding(embedding_str):
    if isinstance(embedding_str, str):
        embedding_list = embedding_str.strip('[]').split(', ')
        embedding_floats = [float(x) for x in embedding_list]
        return torch.tensor(embedding_floats, dtype=torch.float32)
    elif isinstance(embedding_str, torch.Tensor):
        return embedding_str
    else:
        return embedding_str

df2['embedding'] = df2['embedding'].apply(lambda x: parse_embedding(x) if isinstance(x, str) or isinstance(x, torch.Tensor) else x)

# Modelo de embeddings y QA


# Definir el dispositivo (GPU si está disponible, de lo contrario, CPU)




from sentence_transformers import SentenceTransformer
import torch

# Detecta si hay GPU disponible
device = "cpu"

# Carga el modelo (NO usar .to() ni .to_empty())
embedder = SentenceTransformer('all-MiniLM-L6-v2')





qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Función QA combinada
def ask_agent(query, k=10, section_filter=None):
    if section_filter:
        filtered_df = df2[df2['Sección'].str.contains(section_filter, case=False, na=False)]
    else:
        filtered_df = df2

    if filtered_df.empty:
        return {
            "Respuesta": "No se encontraron secciones relacionadas con ese filtro.",
            "Confianza": 0.0,
            "Sección": "N/A",
            "Categoría": "N/A",
            "Contexto usado": "N/A"
        }

    documents = filtered_df['Resumen'].tolist()
    sections = filtered_df['Sección'].tolist()
    categories = filtered_df['Categoría'].tolist()
    doc_embeddings = embedder.encode(documents, convert_to_numpy=True)

    index = faiss.IndexFlatL2(doc_embeddings.shape[1])
    index.add(doc_embeddings)

    query_vector = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(query_vector, k)

    best_result = {'score': 0, 'answer': '', 'section': '', 'category': '', 'context': ''}
    for i in I[0]:
        result = qa_pipeline(question=query, context=documents[i])
        if result['score'] > best_result['score']:
            best_result = {
                'score': result['score'],
                'answer': result['answer'],
                'section': sections[i],
                'category': categories[i],
                'context': documents[i]
            }

    if best_result['score'] < 0.15:
        return {
            "Respuesta": "No se encontró suficiente contexto relevante para dar una respuesta confiable.",
            "Confianza": best_result['score'],
            "Sección": best_result['section'],
            "Categoría": best_result['category'],
            "Contexto usado": best_result['context'][:1000] + '...'
        }

    return {
        "Respuesta": best_result['answer'],
        "Confianza": best_result['score'],
        "Sección": best_result['section'],
        "Categoría": best_result['category'],
        "Contexto usado": best_result['context'][:1000] + '...'
    }

# Traductor
def traducir(texto, origen='en', destino='es'):
    try:
        return GoogleTranslator(source=origen, target=destino).translate(texto)
    except Exception as e:
        return f"[Error al traducir]: {str(e)}"

# Interfaz Streamlit
st.set_page_config(page_title="Agente InsurTech", layout="centered")

st.title("🤖 Agente de Innovación InsurTech")
st.subheader("Consulta el reporte técnico (Gallagher Re 2024-Q4) en español")

pregunta = st.text_input("Haz tu pregunta sobre el reporte (en español):")
seccion = st.text_input("Filtra por sección (opcional):")

if st.button("Consultar"):
    if pregunta.strip() == "":
        st.warning("Por favor, escribe una pregunta.")
    else:
        st.info("Procesando tu pregunta...")

        resultado = ask_agent(pregunta, section_filter=seccion)

        st.markdown("### 📌 Respuesta:")
        st.success(traducir(resultado['Respuesta']))

        st.markdown("### 📊 Confianza:")
        st.write(f"{resultado['Confianza']:.2f}")

        st.markdown("### 📂 Sección:")
        st.write(traducir(resultado['Sección']))

        st.markdown("### 🧭 Categoría:")
        st.write(traducir(resultado['Categoría']))

        st.markdown("### 📚 Contexto usado:")
        st.info(traducir(resultado['Contexto usado']))



