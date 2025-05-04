import streamlit as st
import pandas as pd
import openai
import os
from openai import OpenAI
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col, months_between, current_date, lower
from pyspark.ml import PipelineModel
from pyspark.ml.regression import LinearRegressionModel
from pyspark.sql import Row
from google import genai

# ---------- Setup ----------
# Spark config
os.environ["PYSPARK_PYTHON"] = "/Users/jsujanchowdary/Downloads/langchain-ask-csv-main/myenv/bin/python"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/Users/jsujanchowdary/Downloads/langchain-ask-csv-main/myenv/bin/python"
spark = SparkSession.builder.appName("RealEstateApp").getOrCreate()

# Load models
buy_pipeline = PipelineModel.load("models/preprocessing_pipeline/content/preprocessing_pipeline")
buy_model = LinearRegressionModel.load("models/house_price_model/content/house_price_model")
rent_pipeline = PipelineModel.load("models/rental_pipeline/content/rental_pipeline")
rent_model = LinearRegressionModel.load("models/rental_model/content/rental_model")

# OpenAI Key
openai.api_key = "key"
# ---------- Streamlit UI ----------
st.set_page_config(page_title="PropIntel", page_icon="🏠", layout="wide")
st.title("🏠 PropIntel: Real Estate Prediction")
option = st.radio("What would you like to predict?", ["🏠 Buy House Price", "🏘️ Rent Monthly Cost"])

# ---------- OpenAI Insight ----------
client = OpenAI(api_key=openai.api_key)

def get_openai_insight(user_input_dict, prediction_value, mode="buy"):
    input_summary = "\n".join([f"{k}: {v}" for k, v in user_input_dict.items()])
    prediction_line = (
        f"Predicted Buy Price per Sqft: ₹{prediction_value:,}" if mode == "buy"
        else f"Estimated Monthly Rent: ₹{prediction_value:,}"
    )
    prompt = f"""
    A owner provided the following real estate property details:
    {input_summary}

    {prediction_line}
    you need to say "Great!!" and you need to appriciate 
    Based on this information, provide a helpful real estate suggestion for increase in price or not if possible provide the numbers, 
    you need to make the 
    owner think its the best and updated on going sale market trend. Also give the owner profit of this choice..
    """

    chat = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.7
    )
    return chat.choices[0].message.content.strip()
# ---------- Form ----------
def get_common_inputs(is_rent=False):
    city = st.selectbox("City", ["chennai"])
    localities = ["Adyar", "Anna Nagar", "T. Nagar", "Velachery", "OMR", "Tambaram", "Porur", "Guindy",
                  "Mylapore", "Kilpauk", "Perambur", "Vadapalani", "Kodambakkam", "Royapettah", "Egmore",
                  "Pallavaram", "Avadi", "Chromepet", "Sholinganallur", "Ambattur"]
    locality = st.selectbox("Locality", localities)
    col1, col2 = st.columns(2)
    with col1:
        house_type = st.selectbox("House Type", ["apartment", "individual house", "villa", "gated community"])
        bhk = st.number_input("BHK", 1, 10, 2)
        house_floor = st.number_input("House Floor", 0, 50, 1)
        building_floor = st.number_input("Building Floor", 1, 50, 5)
        property_age = st.number_input("Property Age", 0, 100, 5)
        facing = st.selectbox("Facing", ["north", "south", "east", "west", "north-east", "north-west", "south-east", "south-west"])
        sqft = st.number_input("Square Feet", 100, 10000, 1000)
        bathrooms = st.slider("Bathrooms", 1, 5, 2)
        balcony = st.slider("Balcony", 0, 5, 1)
        lot = st.selectbox("Lot Shape", ["rectangular", "irregular", "square"])
        sewer = st.selectbox("Sewer System", ["public", "septic"])
    with col2:
        security = st.selectbox("Security", ["none", "cctv", "guards", "gated"])
        gym = st.selectbox("Gym", ["yes", "no"])
        pool = st.selectbox("Swimming Pool", ["yes", "no"])
        convention = st.selectbox("Convention Hall", ["yes", "no"])
        parking = st.selectbox("Parking", ["car", "bike", "both", "no parking"])
        water = st.selectbox("Water Supply", ["corporation", "borewell", "both"])
        flooring = st.selectbox("Flooring Type", ["marble", "wood", "tiles", "vinyl", "laminate"])
        roof = st.selectbox("Roof Type", ["concrete", "tile", "metal", "slate"])
        style = st.selectbox("Property Style", ["modern", "traditional", "victorian", "contemporary"])
        view = st.selectbox("View", ["sea view", "city view", "garden view", "lake view", "none"])
    availability = st.date_input("Availability Date")
    furnishing = st.selectbox("Furnishing", ["full", "semi", "unfurnished"])
    hoa = st.number_input("HOA Fees", 0.0, 100000.0, 1500.0)
    schools = st.slider("Nearby Schools", 0, 5, 2)
    hospitals = st.slider("Nearby Hospitals", 0, 5, 1)
    supermarkets = st.slider("Nearby Supermarkets", 0, 5, 2)
    it_hub = st.selectbox("IT Hub Present", ["none", "tcs", "infosys", "cts", "zoho", "accenture", "wipro"])
    photos = st.selectbox("Photos Available?", ["yes", "no"])

    inputs = {
        "City": city,
        "Locality": locality,
        "House Type": house_type,
        "BHK": bhk,
        "House Floor": house_floor,
        "Building Floor": building_floor,
        "Property Age": property_age,
        "Facing": facing.lower(),
        "Square Feet": sqft,
        "Security": security.lower(),
        "Gym": gym.lower(),
        "Convention Hall": convention.lower(),
        "Parking": parking.lower(),
        "Water Supply": water.lower(),
        "Bathrooms": bathrooms,
        "Balcony": balcony,
        "Availability Date": str(availability),
        "Furnishing": furnishing.lower(),
        "HOA Fees": hoa,
        "Flooring Type": flooring.lower(),
        "Roof Type": roof.lower(),
        "Property Style": style.lower(),
        "View": view.lower(),
        "Nearby Schools": schools,
        "Nearby Hospitals": hospitals,
        "Nearby Supermarkets": supermarkets,
        "IT Hub Present": it_hub.lower(),
        "Swimming Pool": pool.lower(),
        "Lot Shape": lot.lower(),
        "Sewer System": sewer.lower(),
        "Photos": photos.lower()
    }

    if is_rent:
        deposit = st.number_input("Deposit Cost", 0, 1000000, 50000)
        hoa_r = st.number_input("HOA Fees ", 0.0, 100000.0, 1500.0)
        inputs["Deposit Cost"] = deposit
        inputs["HOA Fees "] = hoa_r

    return inputs

# ---------- Buy Prediction ----------
if option == "🏠 Buy House Price":
    st.header("Buy House Price Prediction")
    user_input = get_common_inputs()
    if st.button("🔮 Predict Buy Price"):
        user_row = Row(**{**user_input, "Property Purpose": "buy"})
        df = spark.createDataFrame([user_row])
        df = df.withColumn("BHK_Sqft", col("BHK") * col("Square Feet"))
        df = df.withColumn("Months_To_Availability", months_between(col("Availability Date"), current_date()))
        df = df.withColumn("Has_Photos", when(col("Photos").isNotNull(), 1).otherwise(0))
        df = df.withColumn("Amenity_Score", (
            when(lower(col("Gym")) == "yes", 1).otherwise(0) +
            when(lower(col("Swimming Pool")) == "yes", 1).otherwise(0) +
            when(lower(col("Security")) != "none", 1).otherwise(0) +
            when(lower(col("Convention Hall")) == "yes", 1).otherwise(0)
        ))
        df = df.withColumn("Is_Near_IT_Hub", when(lower(col("IT Hub Present")) != "none", 1).otherwise(0))

        transformed = buy_pipeline.transform(df)
        prediction = buy_model.transform(transformed)
        prediction = prediction.withColumn("Total_Predicted_Price", col("prediction") * col("Square Feet"))
        row = prediction.select("prediction", "Total_Predicted_Price").first()
        price_per_sqft = round(row["prediction"])
        total_price = round(row["Total_Predicted_Price"])

        st.success(f"🏷 Predicted Price per Sqft: ₹{price_per_sqft:,}")
        st.success(f"🏠 Total Estimated Property Price: ₹{total_price:,}")
        insight = get_openai_insight(user_input, price_per_sqft, "buy")
        st.text_area("🤖 AI Insight", insight, height=300)

# ---------- Rent Prediction ----------
elif option == "🏘️ Rent Monthly Cost":
    st.header("Rent Monthly Cost Prediction")
    user_input = get_common_inputs(is_rent=True)
    if st.button("🔮 Predict Rent Price"):
        user_row = Row(**{**user_input, "Property Purpose": "rent"})
        df = spark.createDataFrame([user_row])
        df = df.withColumn("BHK_Sqft", col("BHK") * col("Square Feet"))
        df = df.withColumn("Months_To_Availability", months_between(col("Availability Date"), current_date()))
        df = df.withColumn("Has_Photos", when(col("Photos").isNotNull(), 1).otherwise(0))
        df = df.withColumn("Amenity_Score", (
            when(lower(col("Gym")) == "yes", 1).otherwise(0) +
            when(lower(col("Swimming Pool")) == "yes", 1).otherwise(0) +
            when(lower(col("Security")) != "none", 1).otherwise(0) +
            when(lower(col("Convention Hall")) == "yes", 1).otherwise(0)
        ))
        df = df.withColumn("Is_Near_IT_Hub", when(lower(col("IT Hub Present")) != "none", 1).otherwise(0))

        transformed = rent_pipeline.transform(df)
        prediction = rent_model.transform(transformed)
        rent = round(prediction.select("prediction").first()["prediction"])

        st.success(f"🏘️ Estimated Monthly Rent: ₹{rent:,}")
        insight = get_openai_insight(user_input, rent, "rent")
        st.text_area("🤖 AI Insight", insight, height=300)