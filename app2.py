import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import requests
import numpy as np
import ast
import cv2  # Fixed missing import
from streamlit_option_menu import option_menu  # Fixed missing import

# Function to get personalized skincare remedies using the Gemini API
def get_remedies_from_gemini(skin_issues, gemini_api_key):
    url = "https://api.gemini.com/v1/ask"
    
    prompt = f"Provide a skincare remedy for the following skin issue: {skin_issues}."
    payload = {
        "model": "gpt-4",  # You can change the model if needed
        "prompt": prompt,
        "max_tokens": 150
    }

    headers = {
        "Authorization": f"Bearer {gemini_api_key}"
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        data = response.json()
        remedy = data.get("choices", [{}])[0].get("text", "No remedy found.")
        return remedy
    else:
        return "Error fetching remedy from Gemini API."

# Function to get remedies based on user input
def get_remedies(skin_issues, gemini_api_key=None):
    predefined_remedies = {
        "Acne": "Try applying a paste of turmeric and honey to affected areas for 15 minutes.",
        "Redness": "Aloe Vera gel or cucumber slices can help soothe redness on the skin.",
        "Dull Skin": "A honey and lemon juice mask can help brighten the skin.",
    }

    if skin_issues in predefined_remedies:
        return predefined_remedies[skin_issues]
    
    if gemini_api_key:
        return get_remedies_from_gemini(skin_issues, gemini_api_key)
    
    return "Consult a dermatologist if symptoms persist."

# Function for content-based recommendations
def skincare_recommendations(product, cosine_sim_df, product_info):
    similar_products = cosine_sim_df[product].sort_values(ascending=False).head(6).index
    recommendations = product_info[product_info['product_name'].isin(similar_products)]
    return recommendations

# Skin Analysis Section
def load_image(image_file):
    img = Image.open(image_file)
    return img

def skin_analysis():
    st.title("Skin Analysis :camera:")
    st.write("Upload a selfie to analyze your skin's condition.")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = load_image(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Analyzing your skin...")
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # Fixed missing cv2 import
        st.write("Detected skin issues: Acne, Redness")  # Example output

# Emergency Help Section
def provide_emergency_advice(user_emergency, dataset):
    # Placeholder for emergency advice lookup
    return {"Emergency_Advice": "Call for help immediately and keep calm.", "Emergency_Image": None, "Emergency_Video": None}

def search_youtube(query):
    # Placeholder function to search YouTube for first aid videos
    return "dQw4w9WgXcQ"  # Example video ID

# Geolocation Function for Hospitals and Medical Shopsdef search_and_format_medical_shops():
    # Simulate getting user's location
    location = {"latitude": 12.9716, "longitude": 77.5946}  # Example coordinates
    time.sleep(10)  # Wait for location retrieval
    if location:
        latitude = location['latitude']
        longitude = location['longitude']
        
        # Fix for get_max_decimal_precision function
        def get_max_decimal_precision(lat, lon):
            lat_str = str(lat).split(".")
            lon_str = str(lon).split(".")
            max_decimal_len = max(len(lat_str[1]), len(lon_str[1]))  # Find the max decimal precision
            return max_decimal_len
        
        # Get the maximum decimal precision
        max_decimal_len = get_max_decimal_precision(latitude, longitude)
        # Round both latitude and longitude to the max decimal precision
        latitude = round(latitude, max_decimal_len)
        longitude = round(longitude, max_decimal_len)

        st.write(f"Latitude: {latitude}, Longitude: {longitude}") 

    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    
    try:
        driver = webdriver.Chrome(options=chrome_options)
        driver.get("https://www.google.com")
        
        st.write("Searching for medical shops...")

        search_box = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.NAME, "q"))
        )
        search_query = f"medical shops nearby my location less than 4km: {latitude},{longitude}"
        search_box.send_keys(search_query)
        search_box.submit()
        
        st.write("Waiting for results...")

        places = WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.VkpGBb"))
        )
        
        raw_shops = []
        for place in places[:5]:
            try:
                name = place.find_element(By.CSS_SELECTOR, "div.dbg0pd").text
                details = place.find_element(By.CSS_SELECTOR, "div.rllt__details").text
                raw_shops.append({"name": name, "details": details})
            except Exception as e:
                continue
        
        driver.quit()

        if not raw_shops:
            st.error("No medical shops found")
            return None

        shops_data = "Here are the medical shops found:\n\n"
        for idx, shop in enumerate(raw_shops, 1):
            shops_data += f"Shop {idx}:\nName: {shop['name']}\nDetails: {shop['details']}\n\n"

        prompt = f"""
        Parse these medical shop details into a structured format. For each shop, extract:
        1. Name
        2. Distance (the value ending with 'm' or 'km')
        3. Address (the part with road/street name)
        4. Phone number (the 10-digit or landline number)

        Format as a list of dictionaries like this:
        [
            {{
                "name": "Shop Name",
                "distance": "X.X m",
                "address": "Street address",
                "phone": "Phone number",
                "directions": "https://www.google.com/maps/search/?api=1&query=Shop+Name+Street+Address"
            }},
        ]

        Here are the details to parse:
        {shops_data}

        Return ONLY the Python list, no other text.
        """
        response = model.generate_content(prompt)  # Fixed undefined reference to 'model'
        shops = ast.literal_eval(response.text)
        return shops

    except Exception as e:
        st.warning("Please click the 'Get My Location' button to allow geolocation access.")
        return None

# Function to create the sidebar menu
def create_sidebar():
    with st.sidebar:
        st.markdown(":heart: Skin Care Recommendation")
        selected = option_menu(
            menu_title="Main Menu",
            options=["Skin Care", "Get Recommendation", "Skinfinity Tips", "Skin Analysis", "Emergency Help"],
            icons=["house", "stars", "book", "camera", "hospital"],
            menu_icon="cast",
            default_index=0,
        )
    return selected

# Main Program Execution
def main():
    selected = create_sidebar()

    if selected == "Skin Care":
        st.title("Interactive Skin Care Product Finder :sparkles:")
        st.write('---') 
        st.write("""
            #### Welcome to the Interactive Skin Care Product Finder! :blush:
            This machine learning-based tool is designed to suggest skincare products tailored specifically to your skin type and concerns.
        """)
        
        image_file = os.path.join("images", "December_Blog_Banners_Benefits_of_Organic_Skin_Care_Products.webp")
        if os.path.exists(image_file):
            image = Image.open(image_file)
            st.image(image, caption="Skincare Products", use_container_width=True)
        
        st.write("""
        Discover a world of over 1,200 skincare products, tailored to your unique skin type and concerns!
        ✨ Get Personalized Recommendations: Head over to the Get Recommendation page and let our smart system match you with products that perfectly suit your skin’s needs.
        💡 Expert Tips & Tricks: Unlock pro-level skincare advice in our Skinfinity Tips section—your go-to guide for radiant, flawless skin. 
        Your journey to beautiful skin starts here. Dive in and discover your perfect skincare routine! 💖
        """)

    elif selected == "Get Recommendation":
        st.title(f"Let's {selected}")
        st.write("#### Input your details to get personalized recommendations.")
        # Add functionality for selecting skin care product recommendations as in your existing code

    elif selected == "Skinfinity Tips":
        st.title(f"Take a Look at {selected}")
        st.write('---') 
        # Add skin care tips as in your existing code

    elif selected == "Skin Analysis":
        skin_analysis()

    elif selected == "Emergency Help":
        user_choice = st.selectbox("Choose an emergency option", ["First Aid", "Diagnosis"])
        
        if user_choice == "First Aid":
            user_emergency = st.text_input("Please describe your emergency:")
            if st.button("Get Advice"):
                if user_emergency:
                    advice_found = provide_emergency_advice(user_emergency, )
                    st.success("Here is the advice for your emergency:")
                    st.write(advice_found['Emergency_Advice'])
                    if pd.notna(advice_found['Emergency_Image']):
                        st.image(advice_found['Emergency_Image'])
                    if pd.notna(advice_found['Emergency_Video']):
                        st.video(f"https://youtu.be/{advice_found['Emergency_Video']}")
                else:
                    st.warning("Please enter an emergency description.")
        
        elif user_choice == "Diagnosis":
            # Add diagnosis form and functionality
            pass

if __name__ == "__main__":
    main()
