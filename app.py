import os
import streamlit as st
import spacy
import PyPDF2
import pandas as pd
import time
from datetime import datetime
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from collections import defaultdict

class ResumeProcessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def extract_text_from_pdf(self, file):
        reader = PyPDF2.PdfReader(file)
        return ' '.join([page.extract_text() for page in reader.pages])

    def preprocess_text(self, text):
        doc = self.nlp(text)
        tokens = [token.lemma_.lower() for token in doc 
                 if not token.is_stop and not token.is_punct]
        return ' '.join(tokens)

    def extract_entities(self, text):
        doc = self.nlp(text)
        entities = defaultdict(set)
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PERSON', 'GPE', 'EDU', 'SKILL']:
                entities[ent.label_].add(ent.text.lower())
        return entities

    def calculate_similarity(self, jd_text, resumes):
        processed_jd = self.preprocess_text(jd_text)
        processed_resumes = [self.preprocess_text(resume) for resume in resumes]
        
        tfidf_matrix = self.vectorizer.fit_transform([processed_jd] + processed_resumes)
        tfidf_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
        
        jd_embedding = self.sentence_model.encode([processed_jd])
        resume_embeddings = self.sentence_model.encode(processed_resumes)
        semantic_scores = cosine_similarity(jd_embedding, resume_embeddings)[0]
        
        jd_entities = self.extract_entities(jd_text)
        entity_scores = []
        for resume in resumes:
            resume_entities = self.extract_entities(resume)
            score = sum(len(jd_entities[key] & resume_entities[key]) 
                      for key in jd_entities) / max(len(jd_entities), 1)
            entity_scores.append(score)
        
        combined_scores = (tfidf_scores + semantic_scores + entity_scores) / 3
        return combined_scores, tfidf_matrix, jd_entities

def get_top_terms(vector, feature_names, top_n=10):
    if vector.nnz == 0:
        return []
    indices = vector.indices
    data = vector.data
    sorted_terms = sorted(zip(indices, data), key=lambda x: -x[1])
    return [feature_names[idx] for idx, _ in sorted_terms[:top_n]]

def generate_llm_feedback(jd, resume):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user",
                "content": f"Job Description:\n{jd}\n\nResume:\n{resume}\n\nProvide brief feedback on resume suitability."
            }]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating feedback: {str(e)}"

def main():
    st.set_page_config(page_title="Resume Ranker Pro", layout="wide")
    st.title("🚀 AI-Powered Resume Ranking System 2.0")
    
    if 'metrics' not in st.session_state:
        st.session_state.metrics = {
            'total_processed': 0,
            'avg_time': 0,
            'last_processed': None,
            'errors': []
        }
    
    processor = ResumeProcessor()
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        jd_file = st.file_uploader("Job Description (TXT)", type="txt")
        resume_files = st.file_uploader("Resumes (PDF/TXT)", 
                                      type=["pdf", "txt"],
                                      accept_multiple_files=True)
        
        st.divider()
        st.header("📊 AIOPS Monitoring")
        st.metric("Total Processed", st.session_state.metrics['total_processed'])
        st.metric("Avg Processing Time", f"{st.session_state.metrics['avg_time']:.2f}s")
        st.metric("Last Processed", st.session_state.metrics['last_processed'] or "Never")
        
        st.divider()
        st.header("🔧 MLOps Settings")
        st.write("Model Version: 1.1.0")
        if st.button("Retrain Model (Mock)"):
            with st.spinner("Simulating retraining..."):
                time.sleep(2)
                st.success("Model updated to v1.1.1")
        
        st.divider()
        llm_enabled = st.checkbox("Enable LLM Feedback")
        
        # Get OpenAI key from environment variable
        openai_key = os.environ.get("OPENAI_API_KEY")
        
        # Only show API key input if not running in production environment
        if not openai_key and llm_enabled:
            openai_key = st.text_input("OpenAI API Key", type="password")
        
        if llm_enabled:
            openai.api_key = openai_key

    if jd_file and resume_files:
        start_time = time.time()
        try:
            jd_text = jd_file.read().decode()
            resume_texts = []
            for file in resume_files:
                if file.type == "application/pdf":
                    text = processor.extract_text_from_pdf(file)
                else:
                    text = file.read().decode()
                resume_texts.append(text)
            
            scores, tfidf_matrix, jd_entities = processor.calculate_similarity(jd_text, resume_texts)
            feature_names = processor.vectorizer.get_feature_names_out()
            jd_top_terms = get_top_terms(tfidf_matrix[0], feature_names)
            
            results = []
            for i, (score, text) in enumerate(zip(scores, resume_texts)):
                resume_vector = tfidf_matrix[i+1]
                resume_terms = get_top_terms(resume_vector, feature_names)
                common_terms = set(jd_top_terms) & set(resume_terms)
                resume_entities = processor.extract_entities(text)
                matched_entities = []
                for key in jd_entities:
                    matched_entities.extend(jd_entities[key] & resume_entities.get(key, set()))
                
                results.append({
                    "Filename": resume_files[i].name,
                    "Score": score,
                    "Top Terms": ", ".join(common_terms),
                    "Matched Entities": ", ".join(matched_entities),
                    "Resume Text": text
                })
            
            df = pd.DataFrame(results).sort_values("Score", ascending=False)
            
            st.subheader("📊 Ranking Results")
            st.dataframe(
                df[["Filename", "Score", "Top Terms", "Matched Entities"]],
                column_config={
                    "Score": st.column_config.ProgressColumn(
                        format="%.4f",
                        min_value=0,
                        max_value=1.0
                    )
                },
                use_container_width=True,
                hide_index=True
            )
            
            if llm_enabled and openai_key:
                st.subheader("🧠 LLM Feedback")
                for idx, row in df.iterrows():
                    with st.expander(f"Feedback for {row['Filename']}"):
                        feedback = generate_llm_feedback(jd_text, row['Resume Text'])
                        st.write(feedback)
            
            processing_time = time.time() - start_time
            st.session_state.metrics['total_processed'] += len(resume_files)
            st.session_state.metrics['avg_time'] = (
                st.session_state.metrics['avg_time'] * (st.session_state.metrics['total_processed'] - len(resume_files)) +
                processing_time
            ) / st.session_state.metrics['total_processed']
            st.session_state.metrics['last_processed'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
        except Exception as e:
            st.error(f"Error processing files: {str(e)}")
            st.session_state.metrics['errors'].append(str(e))

if __name__ == "__main__":
    main()
