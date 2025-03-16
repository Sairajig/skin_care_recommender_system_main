from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

def get_product_availability_with_selenium(product_url):
    options = Options()
    options.add_argument('--headless')  # Run in headless mode for faster execution
    service = Service("path_to_chromedriver")  # Replace with the path to your ChromeDriver

    driver = webdriver.Chrome(service=service, options=options)
    driver.get(product_url)

    try:
        product_name = driver.find_element(By.CSS_SELECTOR, "h1[class*='product-title']").text
        price = driver.find_element(By.CSS_SELECTOR, "span[class*='product-price']").text
        add_to_basket_button = driver.find_elements(By.CSS_SELECTOR, "button[data-at='add_to_basket']")
        availability = "In Stock" if add_to_basket_button else "Out of Stock"

        return {
            "product_name": product_name,
            "price": price,
            "availability": availability
        }
    finally:
        driver.quit()

# Example usage:
product_data = get_product_availability_with_selenium('https://www.sephora.com/shop/skincare')
print(product_data)
