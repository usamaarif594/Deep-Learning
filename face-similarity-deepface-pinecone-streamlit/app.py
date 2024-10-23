import streamlit as st
import matplotlib.pyplot as plt
from deepface import DeepFace
from pinecone import Pinecone, ServerlessSpec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import os
import glob
import contextlib

# Import Pinecone API key and Index Name from api.py
from api import PINECONE_API_KEY, INDEX_NAME

# Initialize Pinecone with API key
pinecone = Pinecone(api_key=PINECONE_API_KEY)
MODEL = "Facenet"  # Define the model for facial recognition

# Function to display an image
def show_img(f):
    img = plt.imread(f)
    plt.figure(figsize=(2, 2))
    plt.imshow(img)
    st.pyplot(plt)

# Function to generate embeddings for a single image
def generate_embedding(file_path):
    try:
        embedding = DeepFace.represent(img_path=file_path, model_name=MODEL, enforce_detection=False)[0]['embedding']
        return embedding
    except Exception as e:
        print(e)
        return None

# Function to generate embeddings for all images and store in a file
def generate_vectors():
    VECTOR_FILE = "./vectors.vec"
    with contextlib.suppress(FileNotFoundError):
        os.remove(VECTOR_FILE)
    files = [f for person in ["mom", "dad", "child"] for f in glob.glob(f'family/{person}/*')]
    total_files = len(files)
    
    with open(VECTOR_FILE, "w") as f:
        progress_bar = st.progress(0)
        for i, file in enumerate(files):
            try:
                embedding = DeepFace.represent(img_path=file, model_name=MODEL, enforce_detection=False)[0]['embedding']
                person = os.path.basename(os.path.dirname(file))
                f.write(f'{person}:{os.path.basename(file)}:{embedding}\n')
            except (ValueError, UnboundLocalError, AttributeError) as e:
                print(e)
            progress_bar.progress((i + 1) / total_files)

# Function to generate t-SNE DataFrame for a specific person
def gen_tsne_df(person, perplexity):
    vectors = []
    with open('./vectors.vec', 'r') as f:
        for line in f:
            p, orig_img, v = line.split(':')
            if person == p:
                vectors.append(eval(v))
    
    if not vectors:
        raise ValueError(f"No vectors found for the specified person: {person}")
    
    vectors = np.array(vectors)
    
    if vectors.shape[0] > 1:
        pca = PCA(n_components=min(vectors.shape[0], vectors.shape[1]))
        pca_transform = pca.fit_transform(vectors)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=0, n_iter=1000, verbose=0, metric='euclidean', learning_rate=75)
        embeddings2d = tsne.fit_transform(pca_transform)
        return pd.DataFrame({'x': embeddings2d[:, 0], 'y': embeddings2d[:, 1]})
    else:
        raise ValueError("Insufficient data points for PCA.")

# Function to plot t-SNE visualization
def plot_tsne(perplexity, model):
    fig, ax = plt.subplots(figsize=(5, 4))
    plt.grid(color='#EAEAEB', linewidth=0.5)
    ax.spines['top'].set_color(None)
    ax.spines['right'].set_color(None)
    ax.spines['left'].set_color('#2B2F30')
    ax.spines['bottom'].set_color('#2B2F30')
    colormap = {'dad': '#ee8933', 'child': '#4fad5b', 'mom': '#4c93db'}

    for person in colormap:
        embeddingsdf = gen_tsne_df(person, perplexity)
        ax.scatter(embeddingsdf.x, embeddingsdf.y, alpha=.5, label=person, color=colormap[person])
    
    ax.set_title(f'Scatter plot of faces using {model}', fontsize=16, fontweight='bold', pad=15)
    fig.suptitle(f't-SNE [perplexity={perplexity}]', y=0.92, fontsize=13)
    ax.legend(loc='best', frameon=True)
    st.pyplot(fig)

# Function to display matching images based on uploaded child image
def display_matching_images(uploaded_child_image, comparison_parent):
    if uploaded_child_image is not None:
        st.image(uploaded_child_image, caption="Uploaded image for child", width=300)
        with open(f'uploaded_child_image.jpg', 'wb') as f:
            f.write(uploaded_child_image.getbuffer())
        
        embedding =generate_embedding(f'uploaded_child_image.jpg')
        if embedding is not None:
            query_response = pinecone.Index(INDEX_NAME).query(
                top_k=3,
                vector=embedding,
                filter={"person": {"$eq": comparison_parent}},
                include_metadata=True
            )
            
            if 'matches' in query_response and query_response['matches']:
                photo = query_response['matches'][0]['metadata']['file']
                similarity_score = query_response['matches'][0]['score']
                st.image(f'family/{comparison_parent}/{photo}', caption=f'Matching image for {comparison_parent} (Similarity: {similarity_score:.2f})', use_column_width=False, width=300)
            else:
                st.write(f"No matching images found for {comparison_parent}.")
        else:
            st.write("Failed to generate embeddings for the uploaded child image.")
    else:
        st.write("Upload an image for the child to find matching images.")

# Function to test similarity scores between parent and child
def test(vec_groups, parent, child):
    index = pinecone.Index(INDEX_NAME)
    parent_vecs = vec_groups[parent]
    K = 10  # Number of top matches to retrieve
    SAMPLE_SIZE = 10  # Number of samples to test
    sum_scores = 0
    progress_bar = st.progress(0)
    for i in range(SAMPLE_SIZE):
        query_response = index.query(
            top_k=K,
            vector=parent_vecs[i],
            filter={"person": {"$eq": child}}
        )
        for row in query_response["matches"]:
            sum_scores += row["score"]
        progress_bar.progress((i + 1) / SAMPLE_SIZE)
    st.write(f'\n\n{parent} AVG: {sum_scores / (SAMPLE_SIZE * K)}')

# Function to store embeddings in Pinecone
def store_vectors():
    if INDEX_NAME in [index.name for index in pinecone.list_indexes()]:
        pinecone.delete_index(INDEX_NAME)
    pinecone.create_index(
        name=INDEX_NAME,
        dimension=128,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
    index = pinecone.Index(INDEX_NAME)
    with open("vectors.vec", "r") as f:
        total_lines = sum(1 for line in f)
        f.seek(0)
        progress_bar = st.progress(0)
        for i, line in enumerate(f):
            person, file, vec = line.split(':')
            index.upsert([(f'{person}-{file}', eval(vec), {"person": person, "file": file})])
            progress_bar.progress((i + 1) / total_lines)

# Function to compute similarity scores between parents and child
# Function to compute similarity scores between parents and child
def compute_scores():
    vec_groups = {"dad": [], "mom": [], "child": []}
    with open("vectors.vec", "r") as f:
        for line in f:
            person, file, vec = line.split(':')
            vec_groups[person].append(eval(vec))
    for parent in ["dad", "mom"]:
        st.write(f"{parent.upper()} {'-' * 20}")
        test(vec_groups, parent, "child")


# streamlit App
def main():
    st.title("Facial Similarity Search")
    st.subheader('Exmaple Images')

    col1, col2, col3 = st.columns(3)

    # Initialize session states
    if 'embeddings_generated' not in st.session_state:
        st.session_state.embeddings_generated = False
    if 'vectors_generated' not in st.session_state:
        st.session_state.vectors_generated = False
    if 'vectors_stored' not in st.session_state:
        st.session_state.vectors_stored = False
    if 'similarity_computed' not in st.session_state:
        st.session_state.similarity_computed = False

    # Upload Images for Mom
    with col1:
        uploaded_image_mom = st.sidebar.file_uploader("Upload mom's image", type=['jpg', 'png'], key="mom")
        if uploaded_image_mom is not None:
            st.image(uploaded_image_mom, caption='Mom', width=150)
            with open(f'uploaded_mom_image.jpg', 'wb') as f:
                f.write(uploaded_image_mom.getbuffer())

    # Upload Images for Dad
    with col2:
        uploaded_image_dad = st.sidebar.file_uploader("Upload dad's image", type=['jpg', 'png'], key="dad")
        if uploaded_image_dad is not None:
            st.image(uploaded_image_dad, caption='Dad', width=150)
            with open(f'uploaded_dad_image.jpg', 'wb') as f:
                f.write(uploaded_image_dad.getbuffer())


    # Upload Images for Child
    with col3:
        uploaded_image_child = st.sidebar.file_uploader("Upload child's image", type=['jpg', 'png'], key="child")
        if uploaded_image_child is not None:
            st.image(uploaded_image_child, caption='Child', width=150)
            with open(f'uploaded_child_image.jpg', 'wb') as f:
                f.write(uploaded_image_child.getbuffer())
  
    # Button to generate and save embeddings
    if st.button('Generate and Save Embeddings'):
        st.session_state.vectors_generated = True
        st.session_state.vectors_stored = True

    if st.session_state.vectors_generated:
        generate_vectors()
        store_vectors()
        st.text("Embeddings generated, saved to vectors.vec, and stored in Pinecone.")
        st.session_state.vectors_generated = False

    # Button to calculate similarity scores
    if st.button('Calculate Similarity Scores'):
        st.session_state.similarity_computed = True

    if st.session_state.similarity_computed:
        compute_scores()
        
        st.session_state.similarity_computed = True

    # Dropdown to select parent and button to show matching images for child
    comparison_parent = st.selectbox("Select parent to compare with", options=["dad", "mom"])
    col4, col5 = st.columns(2)
    with col4:
        if st.button('Show Matching Images'):
            display_matching_images(uploaded_image_child, comparison_parent)

    # Sidebar options for t-SNE plot, generate/save embeddings, calculate scores
    if st.sidebar.checkbox('Show t-SNE plot'):
        st.header("t-SNE Plotting")
        perplexity = st.slider("Select perplexity for t-SNE", min_value=5, max_value=50, value=20, step=1)
        plot_tsne(perplexity, 'Facenet')


if __name__ == '__main__':
    main()

