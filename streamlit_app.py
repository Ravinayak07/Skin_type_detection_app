import streamlit as st
from streamlit_option_menu import option_menu
from numpy.core.fromnumeric import prod
import tensorflow as tf
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# Import the Dataset
skincare = pd.read_csv("export_skincare.csv", encoding="utf-8", index_col=None)

# Header
st.set_page_config(
    page_title="Skin Type Detection and product recommendation system",
    page_icon="./skin.png",
    layout="wide",
)
# # Set page config
# st.set_page_config(page_title="Skin Care App", page_icon="💧", layout="wide")

# Inject CSS to style st.radio like a navbar
st.markdown(
    """
    <style>
    div[data-baseweb="radio"] {
        display: flex;
        justify-content: center;
        gap: 1rem;
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    div[data-baseweb="radio"] > div {
        background-color: #ffffff;
        padding: 0.5rem 1.2rem;
        border-radius: 8px;
        transition: all 0.2s ease;
        cursor: pointer;
        border: 1px solid transparent;
    }
    div[data-baseweb="radio"] > div:hover {
        background-color: #e2e6ea;
        border-color: #007bff;
    }
    div[data-baseweb="radio"] > div[data-selected="true"] {
        background-color: #007bff;
        color: white;
        font-weight: bold;
        border-color: #0056b3;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# Now based on selection, display different sections
selected = st.radio(
    "Navigation",
    ["Home", "Get Recommendation", "Analysis", "About"],
    horizontal=True,
)

st.write("---")


def home():

    # Create two columns
    col1, col2 = st.columns([1, 1.5])  # Adjust the ratio as needed

    # Left Column: Image
    with col1:
        image = Image.open("./skin.png")  # Replace with your image path
        st.image(image, caption="Skin Care", use_container_width=True)

    with col2:
        st.markdown(
            """
        ### 🌸 **Skin Care**

        Welcome to the **AI-Powered Skin Care Advisor**! 💆‍♀️✨  
        Your smart companion for healthy, glowing skin. Here's what you can do with this app:

        ---

        #### 🌿 What You Can Do:
        - 🧴 **Get personalized skincare product recommendations**  
          Based on your **skin type**, **concerns**, and desired **benefits**
          
        - 📷 **Upload your skin images for AI analysis**  
          Detect skin types and common skin conditions automatically

        - 🔍 **Understand your skin concerns better**  
          Dive into the **science behind skincare** and get tips that work for you

        - 📊 **Compare products and learn from data**  
          View **notable effects**, pricing, and other important details to make the right choice

        - 🧠 **Powered by Machine Learning**  
          Our content-based recommendation system uses NLP and vector similarity to find products tailored just for *you*

        """
        )


# TF-IDF Vectorization and Cosine Similarity Setup

tf = TfidfVectorizer()
tfidf_matrix = tf.fit_transform(skincare["notable_effects"])

cosine_sim = cosine_similarity(tfidf_matrix)

cosine_sim_df = pd.DataFrame(
    cosine_sim, index=skincare["product_name"], columns=skincare["product_name"]
)


def skincare_recommendations(
    nama_produk,
    similarity_data=cosine_sim_df,
    items=skincare[["product_name", "price", "description"]],
    k=5,
):

    # Mengambil data dengan menggunakan argpartition untuk melakukan partisi secara tidak langsung sepanjang sumbu yang diberikan
    # Dataframe diubah menjadi numpy
    # Range(start, stop, step)
    index = (
        similarity_data.loc[:, nama_produk].to_numpy().argpartition(range(-1, -k, -1))
    )

    # Mengambil data dengan similarity terbesar dari index yang ada
    closest = similarity_data.columns[index[-1 : -(k + 2) : -1]]

    # Drop nama_produk agar nama produk yang dicari tidak muncul dalam daftar rekomendasi
    closest = closest.drop(nama_produk, errors="ignore")
    df = pd.DataFrame(closest).merge(items).head(k)
    return df


def getrecommendation():
    st.title(f"Your Skincare Application")

    st.write(
        """
        ##### **Enter your skin type, concerns, and desired benefits to receive the best skincare product recommendations tailored to your needs.**
        """
    )

    st.write("---")

    # Create two columns
    col1, col2 = st.columns([1, 2])  # Adjust the ratio as needed

    with col1:
        st.subheader("📸 Upload Your Skin Image")
        uploaded_file = st.file_uploader(
            "Upload an image (jpg/png)", type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            uploaded_image = Image.open(uploaded_file)
            st.image(
                uploaded_image,
                caption="Your Uploaded Skin Image",
                use_container_width=True,
            )

    # Right Column: Form
    with col2:
        first, last = st.columns(2)

        # Choose a product type category
        category = first.selectbox(
            label="Product Category : ", options=skincare["product_type"].unique()
        )
        category_pt = skincare[skincare["product_type"] == category]

        # Choose a skin type
        skin_type = last.selectbox(
            label="Your Skin Type : ",
            options=["Normal", "Dry", "Oily"],
        )
        category_st_pt = category_pt[category_pt[skin_type] == 1]

        # Skin problems
        prob = st.multiselect(
            label="Skin Problems : ",
            options=[
                "Dull Skin",
                "Acne",
                "Acne Scars",
                "Large Pores",
                "Black Spots",
                "Fine Lines and Wrinkles",
                "Comedo",
                "Uneven Skin Tone",
                "Redness",
                "Sagging Skin",
            ],
        )

        # Notable effects
        opsi_ne = category_st_pt["notable_effects"].unique().tolist()
        selected_options = st.multiselect("Notable Effects : ", opsi_ne)
        category_ne_st_pt = category_st_pt[
            category_st_pt["notable_effects"].isin(selected_options)
        ]

        # Choose product
        opsi_pn = category_ne_st_pt["product_name"].unique().tolist()
        product = st.selectbox("Recommended Products For You", options=sorted(opsi_pn))

        # Button to find recommendations
        model_run = st.button("Find Other Product Recommendations!")

        if model_run:
            st.write(
                "Here are recommendations for other similar products according to what you want"
            )
            st.write(skincare_recommendations(product))


# Example usage:
if selected == "Home":
    st.title("🏠 Welcome to the Skin Care Recommender App")
    home()
elif selected == "Get Recommendation":
    st.title("🧴 Product Recommendation Section")
    getrecommendation()
elif selected == "Analysis":
    st.title("📊 Analyze Your Skin")
elif selected == "About":
    st.title("❓ About This App")
