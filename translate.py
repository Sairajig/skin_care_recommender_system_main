

import pandas as pd
from deep_translator import GoogleTranslator

# Load the CSV or Excel file
file_path = 'export_skincare.csv'  # Update with the file's location
data = pd.read_csv(file_path)

# Translate descriptions
data['description'] = data['description'].apply(
    lambda x: GoogleTranslator(source='auto', target='en').translate(x)
)

# Save the updated file
updated_file_path = 'translated_skincare.xlsx'
data.to_excel(updated_file_path, index=False)
print(f"Translated file saved as {updated_file_path}")