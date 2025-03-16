import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import os

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time

# Import the Dataset 
skincare = pd.read_csv("export_skincare.csv", encoding='utf-8', index_col=None)

# Set page configuration
st.set_page_config(
    page_title="Skin Care Recommender System", 
    page_icon=":heart:", 
    layout="wide"
)

# Function to create the sidebar menu
def create_sidebar():
    with st.sidebar:
        st.markdown(":heart: Skin Care Recommendation")
        selected = option_menu(
            menu_title="Main Menu",
            options=["Skin Care", "Get Recommendation", "Skinfinity Tips"],
            icons=["house", "stars", "book"],
            menu_icon="cast",
            default_index=0,
        )
    return selected

  # Use the sidebar menu
def skincare_recommendations(product_name, similarity_data, items, k=5):
    index = similarity_data.loc[:, product_name].to_numpy().argpartition(range(-1, -k, -1))
    closest = similarity_data.columns[index[-1:-(k+2):-1]]
    closest = closest.drop(product_name, errors='ignore')
    df = pd.DataFrame(closest).merge(items).head(k)
    return df

# Sidebar menu selection
selected = create_sidebar()

if selected == "Skin Care":
    st.title("Interactive Skin Care Product Finder :sparkles:")
    st.write('---') 

    st.write("""
        ##### Welcome to the Interactive Skin Care Product Finder! :blush:
        This machine learning-based tool is designed to suggest skincare products tailored specifically to your skin type and concerns.
    """)
    image_file = "December_Blog_Banners_Benefits_of_Organic_Skin_Care_Products.webp"
    image = Image.open(image_file)
    st.image(image, caption="Skincare Products", use_container_width=True)
    
    st.write("""
        Discover a world of over 1,200 skincare products, tailored to your unique skin type and concerns!
        ✨ Get Personalized Recommendations: Head over to the Get Recommendation page and let our smart system match you with products that perfectly suit your skin’s needs.
        💡 Expert Tips & Tricks: Unlock pro-level skincare advice in our Skinfinity Tips section—your go-to guide for radiant, flawless skin. 
        Your journey to beautiful skin starts here. Dive in and discover your perfect skincare routine! 💖
        """)
    
    st.write("Happy exploring your skin care journey! :smiley:")


elif selected == "Get Recommendation":
    st.title(f"Let's {selected}")
    st.write(
        """
        ##### *To get recommendations, please input your skin type, issues, and desired benefits for the most suitable skincare products*
        """) 
    
    st.write('---') 

    first,last = st.columns(2)

    # Choose a product type category
    category = first.selectbox(label='Product Category : ', options= skincare['product_type'].unique() )
    category_pt = skincare[skincare['product_type'] == category]

    # Choose a skin type
    skin_type = last.selectbox(label='Your Skin Type : ', options= ['Normal', 'Dry', 'Oily', 'Combination', 'Sensitive'] )
    category_st_pt = category_pt[category_pt[skin_type] == 1]

    # Choose problems
    prob = st.multiselect(label='Skin Problems : ', options= ['Dull Skin', 'Acne', 'Acne Scars', 'Large Pores', 'Dark Spots', 'Fine Lines and Wrinkles', 'Blackheads', 'Uneven Skin Tone', 'Redness', 'Sagging Skin'])

    # Choose notable effects
    opsi_ne = category_st_pt['notable_effects'].unique().tolist()
    selected_options = st.multiselect('Notable Effects : ',opsi_ne)
    category_ne_st_pt = category_st_pt[category_st_pt["notable_effects"].isin(selected_options)]

    # Choose product
    opsi_pn = category_ne_st_pt['product_name'].unique().tolist()
    product = st.selectbox(label='Recommended Product For You', options = sorted(opsi_pn))

    ## MODELLING with Content-Based Filtering
    # Initialize TfidfVectorizer
    tf = TfidfVectorizer()

    # Fit the 'notable_effects' data
    tf.fit(skincare['notable_effects']) 

    # Mapping array of feature indices to feature names
    tf.get_feature_names_out()

    # Fit and transform into a matrix
    tfidf_matrix = tf.fit_transform(skincare['notable_effects']) 

    # View the size of the matrix
    shape = tfidf_matrix.shape

    # Convert the tf-idf vector to a matrix using todense()
    tfidf_matrix.todense()

    # Create a dataframe to view the tf-idf matrix
    pd.DataFrame(
        tfidf_matrix.todense(), 
        columns=tf.get_feature_names_out(),
        index=skincare.product_name
    ).sample(shape[1], axis=1).sample(10, axis=0)

    # Calculate cosine similarity on the tf-idf matrix
    cosine_sim = cosine_similarity(tfidf_matrix) 

    # Create a dataframe from the cosine similarity
    cosine_sim_df = pd.DataFrame(cosine_sim, index=skincare['product_name'], columns=skincare['product_name'])

    # View the similarity matrix
    cosine_sim_df.sample(7, axis=1).sample(10, axis=0)

    # Function to get recommendations
    def skincare_recommendations(nama_produk, similarity_data=cosine_sim_df, items=skincare[['product_name', 'brand', 'description']], k=5):
        index = similarity_data.loc[:,nama_produk].to_numpy().argpartition(range(-1, -k, -1))
        closest = similarity_data.columns[index[-1:-(k+2):-1]]
        closest = closest.drop(nama_produk, errors='ignore')
        df = pd.DataFrame(closest).merge(items).head(k)
        return df

    # Button to display recommendations
    model_run = st.button('Find More Similar Product Recommendations!')
    
    if model_run:
        st.write('Here are the Similar Products Based on Your Input:')
        st.write(skincare_recommendations(product))
    
    # Add your recommendation logic here...

elif selected == "Skinfinity Tips":
    st.title(f"Take a Look at {selected}")
    
    st.write("---") 
    st.write("Here are some tips and tricks for skincare!")
    st.write(
        """
        ##### *Here are some tips and tricks to maximize the use of your skincare products*
        """) 
    
    image = Image.open('imagepic.jpg')
    st.image(image, caption='Skin Care 101')
    selected = st.selectbox("Select a Skincare Topic", ["Facial Wash", "Toner", "Serum", "Moisturizer", "Sunscreen", "General Tips"])
if selected == "Facial Wash":
    st.title(f"Take a Look at {selected}")
    st.write("---")
    st.write("Here are some tips and tricks for Facial Wash!")
    st.write("### *Facial Wash Routine for Healthy Skin 🌸*")
    st.write("1. **Choose the Right Facial Wash 🧴**")
    st.write("Tip: Select a facial wash that suits your skin type (e.g., gel-based for oily skin, cream-based for dry skin, etc.).")
    st.write("2. **Morning & Night Routine 🌞🌙**")
    st.write("Wash twice a day: Cleanse in the morning to remove impurities and at night to wash away makeup, dirt, and oil.")
    st.write("3. **Gentle Cleansing 👐**")
    st.write("Tip: Always wash your face gently using your fingertips—no harsh scrubbing!")
    st.write("4. **Water Temperature 💧**")
    st.write("Tip: Use lukewarm water to rinse your face. Water that's too hot or too cold can irritate the skin.")
    st.write("5. **Pat Dry, Don’t Rub ✨**")
    st.write("Tip: After washing, gently pat your face dry with a soft towel. Rubbing your face can cause irritation and damage the skin.")
    st.write("6. **Hydrate After Washing 💦**")
    st.write("Tip: Follow up with a moisturizer to lock in hydration after cleansing.")
    st.write("7. **Cleansing at Night 🌙**")
    st.write("Tip: If you wear makeup, consider a two-step cleanse: first, use an oil-based cleanser to remove makeup, followed by your regular face wash to deep clean your skin.")

elif selected == "Toner":
    st.title(f"Take a Look at {selected}")
    st.write("---")
    st.write("Here are some tips for using toner!")
    st.write("### *Toner: Refresh & Balance Your Skin 💧*")
    st.write("1. **Why Use a Toner? 🌿**")
    st.write("Purpose: Toners help to balance your skin’s pH, remove any remaining impurities after cleansing, and hydrate the skin, preparing it for the next steps in your routine (like serums and moisturizers).")
    st.write("2. **Choose the Right Toner for Your Skin Type 💖**")
    st.write("For Oily Skin: Choose a toner with ingredients like witch hazel or salicylic acid that can help control excess oil and prevent breakouts.")
    st.write("For Dry Skin: Opt for hydrating toners with ingredients like glycerin, hyaluronic acid, or rose water to boost moisture.")
    st.write("For Sensitive Skin: Look for soothing toners with chamomile, aloe vera, or cucumber to calm the skin without irritation.")
    st.write("3. **How to Apply Toner ✨**")
    st.write("Step 1: After cleansing your face, pour a small amount of toner onto a cotton pad (or use your hands for a more eco-friendly approach).")
    st.write("Step 2: Gently sweep the cotton pad over your face, starting from the center and working outwards. Avoid the eye area.")
    st.write("Tip: You can also use a toner as a refreshing mist throughout the day to keep your skin hydrated and revitalized.")
    st.write("4. **Don't Overdo It! 🚫**")
    st.write("Tip: While toner can be beneficial, overusing it can lead to dryness or irritation, especially if it's an exfoliating toner. Stick to once or twice a day for best results.")
    st.write("5. **What’s Next After Toner? 💆‍♀️**")
    st.write("Follow-up: After applying toner, continue with the rest of your skincare routine (serums, moisturizers, and sunscreen).")
    st.write("6. **Special Toner Uses ✨**")
    st.write("For Acne-Prone Skin: Look for toners that contain salicylic acid or tea tree oil to help prevent breakouts.")
    st.write("For Dull Skin: Exfoliating toners with alpha hydroxy acids (AHAs) can help brighten the complexion.")

elif selected == "Serum":
    st.title(f"Take a Look at {selected}")
    st.write("---")
    st.write("Here are some tips for using serum!")
    st.write("### Serum: Targeted Treatment for Your Skin ✨")
    st.write("1. **What is a Serum? 💧**")
    st.write("Purpose: Serums are concentrated skincare products designed to target specific skin concerns like wrinkles, dark spots, acne, or dehydration. They typically contain active ingredients like vitamins, antioxidants, and acids.")
    st.write("2. **Choose the Right Serum for Your Skin Concern 🌟**")
    st.write("For Aging/Anti-Aging: Look for serums with retinol, vitamin C, or peptides to help reduce fine lines and promote collagen production.")
    st.write("For Hyperpigmentation: Use serums with ingredients like vitamin C, niacinamide, or alpha arbutin to brighten dark spots and even skin tone.")
    st.write("For Acne-Prone Skin: Choose serums with salicylic acid, tea tree oil, or niacinamide to fight breakouts and reduce inflammation.")
    st.write("For Hydration: Opt for hyaluronic acid-based serums that provide deep hydration and plump up your skin.")
    st.write("3. **How to Apply Serum Correctly 👐**")
    st.write("Step 1: After toning your skin, apply 2-3 drops of serum onto your fingertips.")
    st.write("Step 2: Gently press the serum into your skin, focusing on areas of concern (e.g., fine lines, dark spots). Avoid rubbing too harshly.")
    st.write("Step 3: Allow the serum to fully absorb before moving on to your moisturizer.")
    st.write("Tip: Serums are lightweight and quickly absorbed, so use them before heavier products like moisturizers.")
    st.write("4. **How Often Should You Use a Serum? ⏰**")
    st.write("Frequency: Most serums can be used once or twice a day—morning and night—depending on your skin's needs and the product's strength.")
    st.write("Tip: Some serums, like those with retinol or strong acids, may be better for nighttime use, as they can increase sun sensitivity.")
    st.write("5. **Layering Serums 🌿**")
    st.write("Tip: If you use more than one serum, apply them from thinnest to thickest consistency. For example, start with a vitamin C serum in the morning and a hydrating serum at night.")
    st.write("6. **Follow Up with Moisturizer 🌙**")
    st.write("Tip: After applying your serum, always follow up with a moisturizer to lock in hydration and maximize the effects of the serum.")
    st.write("7. **Special Tips for Serum Use 💡**")
    st.write("Sensitive Skin: If you're new to serums, introduce them slowly into your routine (e.g., every other day) to avoid irritation.")
    st.write("Boost Effectiveness: Use serums consistently for at least 4–6 weeks to see visible results. Track your skin’s progress using our Serum Progress Log.")

elif selected == "Moisturizer":
    st.title(f"Take a Look at {selected}")
    st.write("---")
    st.write("Here are some tips for using moisturizer!")
    st.write("### Moisturizer: Lock in Hydration & Nourish Your Skin 🌸")
    st.write("1. **What is a Moisturizer? 💧**")
    st.write("Purpose: Moisturizers help to hydrate the skin, lock in moisture, and maintain the skin's natural barrier. They prevent dryness, promote smoothness, and can even reduce the appearance of fine lines.")
    st.write("2. **Choose the Right Moisturizer for Your Skin Type 💖**")
    st.write("For Oily Skin: Opt for lightweight, oil-free, or gel-based moisturizers that hydrate without clogging pores.")
    st.write("For Dry Skin: Choose rich, creamy moisturizers with ingredients like shea butter, ceramides, or glycerin to deeply hydrate and restore moisture.")
    st.write("For Sensitive Skin: Go for gentle, fragrance-free moisturizers that soothe and protect the skin without irritation.")
    st.write("For Combination Skin: Look for an oil-free moisturizer with added hydrating ingredients to balance areas that may be dry.")
    st.write("3. **How to Apply Moisturizer ✨**")
    st.write("Step 1: After applying your serum, take a small amount of moisturizer on your fingertips.")
    st.write("Step 2: Gently massage the moisturizer into your skin, starting from the center of your face and working outward.")
    st.write("Step 3: Don’t forget your neck and décolletage! They need hydration too.")
    st.write("Tip: For extra hydration, you can use a thicker moisturizer at night.")
    st.write("4. **How Often Should You Moisturize? ⏰**")
    st.write("Frequency: Moisturize twice a day—once in the morning and once at night—to keep your skin nourished and protected.")
    st.write("Tip: If you have dry skin, you may want to apply more moisturizer during the day, especially in colder months.")
    st.write("5. **Special Moisturizer Tips ✨**")
    st.write("For Dry Winter Skin: Consider using a heavier moisturizer during winter months when the air is dry and your skin needs extra hydration.")
    st.write("For Oily Skin: Use an oil-free moisturizer to avoid excess shine while still keeping your skin hydrated.")
    
elif selected == "Sunscreen":
    st.title(f"Take a Look at {selected}")
    st.write("---")
    st.write("Here are some tips for using sunscreen!")
    st.write("### Sunscreen: Protect Your Skin from Harmful UV Rays 🌞")
    st.write("1. **Why is Sunscreen Important? 🌟**")
    st.write("Protects your skin from harmful UVA/UVB rays, preventing premature aging, sunburns, and skin cancer.")
    st.write("Tip: Always apply sunscreen daily, even on cloudy days, as UV rays can penetrate through clouds.")
    st.write("2. **Choose the Right Sunscreen for Your Skin Type 💖**")
    st.write("For Oily Skin: Use lightweight, oil-free sunscreens to avoid clogged pores.")
    st.write("For Dry Skin: Opt for hydrating sunscreens with ingredients like aloe vera or hyaluronic acid.")
    st.write("For Sensitive Skin: Use physical sunscreens with zinc oxide or titanium dioxide, as they are less likely to cause irritation.")
    st.write("3. **How to Apply Sunscreen 🌞**")
    st.write("Step 1: Apply sunscreen as the last step of your skincare routine, right before makeup.")
    st.write("Step 2: Apply generously, covering all exposed areas, including your neck, ears, and hands.")
    st.write("Tip: Use at least a nickel-sized amount for your face.")
    st.write("4. **How Often Should You Reapply? ⏰**")
    st.write("Reapply every 2 hours, and more frequently if swimming, sweating, or towel-drying.")
    
elif selected == "General Tips":
    st.title(f"Take a Look at {selected}")
    st.write("---")
    st.write("Here are some general skincare tips!")
    st.write("General Skincare Tips for Healthy Skin 🌿")
    st.write("1. **Stay Hydrated 💧**")
    st.write("Drink plenty of water throughout the day to keep your skin hydrated from the inside out.")
    st.write("2. **Get Enough Sleep 💤**")
    st.write("Sleep helps to regenerate your skin cells, reduce puffiness, and improve your skin's appearance.")
    st.write("3. **Follow a Consistent Routine 🧴**")
    st.write("Stick to a daily skincare routine that includes cleansing, toning, moisturizing, and sunscreen.")
    st.write("4. **Exfoliate Regularly ✨**")
    st.write("Exfoliate once or twice a week to remove dead skin cells and reveal a brighter complexion.")
    st.write("5. **Use Sunscreen Daily 🌞**")
    st.write("Never skip sunscreen—it helps protect your skin from harmful UV rays and prevents premature aging.")
    st.write("6. **Don’t Skip Moisturizing 🌙**")
    st.write("Keep your skin nourished by moisturizing every day, even if you have oily skin.")
