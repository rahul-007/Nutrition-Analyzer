import streamlit as st
from langchain_groq import ChatGroq
from langchain.messages import HumanMessage, SystemMessage
import base64
import os

# ------------------------------
# STREAMLIT UI
# ------------------------------
st.set_page_config(page_title="Nutrition Analyzer", layout="centered")

st.title("🥗 AI Nutrition Analyzer")
st.write("Upload a food image to estimate its nutritional content and get healthy insights.")

uploaded_file = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])

# ------------------------------
# LLM CONFIGURATION
# ------------------------------

groq_api_key = st.secrets["GROQ_API_KEY"]
llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct",groq_api_key=groq_api_key)

nutrition_prompt = """
You are a certified nutritionist and food recognition expert.
You will be provided with an image of food.

Your goal:
Analyze the image and provide a detailed nutritional summary for a health-conscious individual.

Follow these steps:
1. Identify all food items visible in the image.
2. Estimate approximate portion sizes in grams, cups, or slices.
3. For each food item, estimate the nutritional content:
   - Protein (g)
   - Carbohydrates (g)
   - Fats (g)
   - Fiber (g)
   - Total Calories (kcal)
4. Provide a total nutritional estimate for the whole meal.
5. Share insights for a health-conscious user:
   - Macronutrient balance (high/low in protein, carbs, or fat)
   - Whether it’s suitable for weight loss, muscle gain, or maintenance
   - Simple healthy swaps to improve the meal quality.

Output format (in plain text, not JSON):
---
### 🥗 Items Detected:
• ...
### ⚖️ Portion Estimates:
• ...
### 🍎 Nutrition Estimates: create a table and mention below information in rows:
• Protein: ... g
• Carbs: ... g
• Fats: ... g
• Fiber: ... g
• Total Calories: ... kcal
### 💡 Insights:
• ...
### 🩺 Health Tips:
• ...
---
Keep the tone informative and supportive, suitable for a health-conscious audience.
"""

# ------------------------------
# PROCESS IMAGE
# ------------------------------
if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Meal", use_column_width=True)
    st.write("Analyzing image... please wait ⏳")

    # Convert image to base64 for Gemini model input
    image_bytes = uploaded_file.getvalue()
    image_b64 = base64.b64encode(image_bytes).decode()

    # Build LLM message
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": nutrition_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}",
                    },
                },
            ],
        }
    ]

    # Run the model
    response = llm.invoke(messages)
    result = response.content

    # Display nicely
    st.markdown("---")
    st.markdown(result)








