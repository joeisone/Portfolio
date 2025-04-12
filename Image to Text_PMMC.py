!sudo apt install tesseract-ocr
!sudo apt install libtesseract-dev
!pip install pytesseract

from PIL import Image
import pytesseract
from IPython import get_ipython
from IPython.display import display
import json
import re


result=pytesseract.image_to_string(Image.open('/content/INVOICE Assay-PMMCMIIFGRL0902024G_GRLMIIF062_SCD025-MC011_page-0001.jpg'))
print(result)
print(type(result))

def extract_data_PMMC(text):
    """
    Extracts NET WEIGHT, LME PRICE, TOTAL VALUE, Reference, and Date from a given text.

    Args:
        text (str): The input text containing the data.

    Returns:
        dict: A dictionary containing the extracted data, or None if a field is not found.
    """
    # expand patterns with gross weight and net weight with
    patterns = {
        "NET WEIGHT (oz)": r"NET WEIGHT \(oz\) :- ([\d,.]+)",
        "LME PRICE/USS": r"LME PRICE/USS :- ([\d,.]+)",
        "TOTAL VALUE IN USS": r"TOTAL VALUE IN USS :- ([\d,.]+)",
        "Reference": r"Reference : ([\w/]+)",
        "Date": r"Date: ([\d]{2} [A-Za-z]{3} [\d]{4})"
    }

    extracted_data = {}

    for field, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            value = match.group(1)
            if field in ["NET WEIGHT (oz)", "LME PRICE/USS", "TOTAL VALUE IN USS"]:
                value = float(value.replace(",", ""))
            extracted_data[field] = value
        else:
            extracted_data[field] = None # or you can use a default value, or leave it out.

    return extracted_data

def extract_data_INVOICE(text):









# Example usage (if statement based on ML model output to call
# specific image to text function




text = result
extracted_values = extract_data_PMMC(text)

# Print the extracted data
for key, value in extracted_values.items():
    print(f"{key}: {value}")




