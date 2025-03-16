from numpy.core.fromnumeric import prod
import tensorflow as tf
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
skincare = pd.read_csv("export_skincare.csv", encoding='utf-8', index_col=None)

# Header
st.set_page_config(page_title="Skin Care Product Recommendation App", page_icon=":blossom:", layout="wide",)

# Displaying the main page
st.title("Skin Care Product Recommendation App :sparkles:")

st.write('---') 

# Displaying a local video file
video_file = open("skincare.mp4", "rb").read()
st.video(video_file, start_time=1)  # Displaying the video

st.write('---') 

st.write(
    """
    ##### **The Skin Care Product Recommendation App is an implementation of a machine learning project that provides skin care product recommendations based on your skin type and issues. You can enter your skin type, concerns, and desired benefits to get the right skin care product recommendations.**
    """)  
st.write('---') 

first, last = st.columns(2)
category = first.selectbox(label='Product Category: ', options=skincare['tipe_produk'].unique())
category_pt = skincare[skincare['tipe_produk'] == category]

# Choose a skin type
# st = skin type
skin_type = last.selectbox(label='Your Skin Type: ', options=['Normal', 'Dry', 'Oily', 'Combination', 'Sensitive'])
category_st_pt = category_pt[category_pt[skin_type] == 1]

# Choose concerns
concerns = st.multiselect(label='Your Skin Issues: ', options=['Dull Skin', 'Acne', 'Acne Scars', 'Large Pores', 'Dark Spots', 'Fine Lines and Wrinkles', 'Blackheads', 'Uneven Skin Tone', 'Redness', 'Sagging Skin'])

# Choose notable effects
# From the products that are already filtered by product type and skin type (category_st_pt), we will get the unique values in the "notable_effects" column
options_ne = category_st_pt['notable_effects'].unique().tolist()
# The unique notable effects are placed into the variable options_ne and used for the value in the multiselect below, which is wrapped in the selected_options variable
selected_options = st.multiselect('Desired Effects: ', options_ne)
# The result from selected_options is stored in the variable category_ne_st_pt
category_ne_st_pt = category_st_pt[category_st_pt["notable_effects"].isin(selected_options)]

# Choose product
# The products that are already filtered and exist in the filtered_df are then filtered again and the unique ones based on product_name are stored in the variable options_pn
options_pn = category_ne_st_pt['product_name'].unique().tolist()
# Create a selectbox that contains the filtered product options above
product = st.selectbox(label='Recommended Product For You', options=sorted(options_pn))
# The selected product will hold a product name that will display other recommended products

## MODELLING with Content-Based Filtering
# Initialize TfidfVectorizer
tf = TfidfVectorizer()

# Calculate idf for the 'notable_effects' column
tf.fit(skincare['notable_effects']) 

# Map the integer index array to feature names
tf.get_feature_names()

# Fit and transform the data into a matrix
tfidf_matrix = tf.fit_transform(skincare['notable_effects']) 

# View the size of the tfidf matrix
shape = tfidf_matrix.shape

# Convert the tf-idf vector into a dense matrix
tfidf_matrix.todense()

# Create a dataframe to view the tf-idf matrix
# The columns are filled with the desired effects
# The rows are filled with product names
pd.DataFrame(
    tfidf_matrix.todense(), 
    columns=tf.get_feature_names(),
    index=skincare.product_name
).sample(shape[1], axis=1).sample(10, axis=0)

# Calculate cosine similarity on the tf-idf matrix
cosine_sim = cosine_similarity(tfidf_matrix) 

# Create a dataframe from the cosine_sim variable with rows and columns as product names
cosine_sim_df = pd.DataFrame(cosine_sim, index=skincare['product_name'], columns=skincare['product_name'])

# View the similarity matrix for each product name
cosine_sim_df.sample(7, axis=1).sample(10, axis=0)

# Function to get product recommendations
def skincare_recommendations(product_name, similarity_data=cosine_sim_df, items=skincare[['product_name', 'produk-href', 'price', 'description']], k=5):
    
    # Use argpartition to perform partitioning along the given axis
    # Convert the dataframe to numpy
    # Range(start, stop, step)
    index = similarity_data.loc[:, product_name].to_numpy().argpartition(range(-1, -k, -1))
    
    # Get the data with the largest similarity from the indices
    closest = similarity_data.columns[index[-1:-(k+2):-1]]
    
    # Drop the original product name so it doesn't appear in the recommendation list
    closest = closest.drop(product_name, errors='ignore')
    df = pd.DataFrame(closest).merge(items).head(k)
    return df

# Create a button to display recommendations
model_run = st.button('Find Other Similar Products!')

if model_run:
    st.write('Here are other similar product recommendations based on your selection:')
    st.write(skincare_recommendations(product))
    st.snow()
