import streamlit as st
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import torch
from huggingface_hub import hf_hub_download

embedding_path = "abokbot/wikipedia-embedding"

st.header("Wikipedia Search Engine app")

st_model_load = st.text('Loading embeddings, encoders and dataset (takes about 5min)')

@st.cache_resource
def load_embedding():
    print("Loading embedding...")
    path = hf_hub_download(repo_id="abokbot/wikipedia-embedding", filename="wikipedia_en_embedding.pt")
    wikipedia_embedding = torch.load(path, map_location=torch.device('cpu')) 
    print("Embedding loaded!")
    return wikipedia_embedding

wikipedia_embedding = load_embedding()

@st.cache_resource
def load_encoders():
    print("Loading encoders...")
    bi_encoder = SentenceTransformer('msmarco-MiniLM-L-6-v3')
    bi_encoder.max_seq_length = 256     #Truncate long passages to 256 tokens
    top_k = 32  
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2-v2')
    print("Encoders loaded!")
    return bi_encoder, cross_encoder

bi_encoder, cross_encoder = load_encoders()

@st.cache_resource
def load_wikipedia_dataset():
    print("Loading wikipedia dataset...")
    dataset = load_dataset("abokbot/wikipedia-first-paragraph")["train"]
    print("Dataset loaded!")
    return dataset
    
dataset = load_wikipedia_dataset()
st.success('Search engine ready')
st_model_load.text("")
    
if 'text' not in st.session_state:
    st.session_state.text = ""
st.markdown("Enter query")
st_text_area = st.text_area(
    'E.g. What is the hashing trick? or Largest city in Morocco', 
    value=st.session_state.text, 
    height=25
)


def search():
    st.session_state.text = st_text_area
    query = st_text_area
    print("Input question:", query)
    
    ##### Sematic Search #####
    print("Semantic Search")
    # Encode the query using the bi-encoder and find potentially relevant passages
    top_k = 32
    question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(question_embedding, wikipedia_embedding, top_k=top_k)
    hits = hits[0]  # Get the hits for the first query

    ##### Re-Ranking #####
    # Now, score all retrieved passages with the cross_encoder
    print("Re-Ranking")
    cross_inp = [[query, dataset[hit['corpus_id']]["text"]] for hit in hits]
    cross_scores = cross_encoder.predict(cross_inp)

    # Sort results by the cross-encoder scores
    for idx in range(len(cross_scores)):
        hits[idx]['cross-score'] = cross_scores[idx]
    
    hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
    # Output of top-3 hits from re-ranker
    print("\n-------------------------\n")
    print("Top-3 Cross-Encoder Re-ranker hits")
    results = []
    for hit in hits[:3]:
        results.append(
            {
                "score": round(hit['cross-score'], 3),
                "title": dataset[hit['corpus_id']]["title"],
                "abstract": dataset[hit['corpus_id']]["text"].replace("\n", " "),
                "link": dataset[hit['corpus_id']]["url"]
            }
        )
    return results


# search button
st_search_button = st.button('Search')
if st_search_button:
    results = search()
    st.subheader("Top-3 Search results")
    for i, result in enumerate(results):
        st.markdown(f"#### Result {i+1}")
        st.markdown("**Wikipedia article:** " + result["title"])
        st.markdown("**Link:** " + result["link"])
        st.markdown("**First paragraph:** " + result["abstract"])
        st.text("")
