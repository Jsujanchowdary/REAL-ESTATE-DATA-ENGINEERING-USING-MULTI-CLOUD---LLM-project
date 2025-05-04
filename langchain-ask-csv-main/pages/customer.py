import streamlit as st
import boto3
import pandas as pd
import os
import openai
from openai import OpenAI
from datetime import datetime
from google import genai




# Instantiate the client
# client = OpenAI(api_key="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")
# Only run this block for Gemini Developer API
client = genai.Client(api_key='key')

# AWS credentials (Replace with st.secrets in production)
AWS_ACCESS_KEY = "key"
AWS_SECRET_KEY = "key"
AWS_REGION = "us-east-1"

# Initialize S3 client

s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION,
)

# Reusable UI component
def get_common_inputs(is_rent=False):
    property_purpose = st.selectbox("Property Purpose", ["rent", "buy"])
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
        parking = st.selectbox("Parking", ["car", "bike", "both", "no parking"])
    with col2:
        facing = st.selectbox("Facing", ["north", "south", "east", "west", "north-east", "north-west", "south-east", "south-west"])
        sqft = st.number_input("Square Feet", 100, 10000, 1000)
        security = st.selectbox("Security", ["none", "cctv", "guards", "gated"])
        gym = st.selectbox("Gym", ["yes", "no"])
        convention = st.selectbox("Convention Hall", ["yes", "no"])
        water = st.selectbox("Water Supply", ["corporation", "borewell", "both"])
    bathrooms = st.slider("Bathrooms", 1, 5, 2)
    balcony = st.slider("Balcony", 0, 5, 1)
    schools = st.slider("Nearby Schools", 0, 5, 2)
    col3, col4 = st.columns(2)
    with col3:
        availability = st.date_input("Availability Date")
        furnishing = st.selectbox("Furnishing", ["full", "semi", "unfurnished"])
        flooring = st.selectbox("Flooring Type", ["marble", "wood", "tiles", "vinyl", "laminate"])
        roof = st.selectbox("Roof Type", ["concrete", "tile", "metal", "slate"])
        style = st.selectbox("Property Style", ["modern", "traditional", "victorian", "contemporary"])
    with col4:
        it_hub = st.selectbox("IT Hub Present", ["none", "tcs", "infosys", "cts", "zoho", "accenture", "wipro"])
        pool = st.selectbox("Swimming Pool", ["yes", "no"])
        hospitals = st.slider("Nearby Hospitals", 0, 5, 1)
        supermarkets = st.slider("Nearby Supermarkets", 0, 5, 2)

    inputs = {
        "property_purpose": property_purpose.lower(),
        "locality": locality.lower(),
        "house type": house_type.lower(),
        "bhk": bhk,
        "house floor": house_floor,
        "building floor": building_floor,
        "property age": property_age,
        "facing": facing.lower(),
        "square feet": sqft,
        "security": security.lower(),
        "gym": gym.lower(),
        "convention hall": convention.lower(),
        "parking": parking.lower(),
        "water supply": water.lower(),
        "bathrooms": bathrooms,
        "balcony": balcony,
        "availability date": str(availability),
        "furnishing": furnishing.lower(),
        "flooring type": flooring.lower(),
        "roof type": roof.lower(),
        "property style": style.lower(),
        "nearby schools": schools,
        "nearby hospitals": hospitals,
        "nearby supermarkets": supermarkets,
        "it hub present": it_hub.lower(),
        "swimming pool": pool.lower()
    }

    return inputs

def fetch_properties(bucket_name, file_key, filters):
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        df = pd.read_csv(response["Body"])
        df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

        # Simplified filter keys
        normalized_filters = {
            k.strip().lower().replace(" ", "_"): v for k, v in filters.items()
        }

        # Manually filter with debug logging
        debug_keys = ["property_purpose", "locality", "house_type", "bhk", "property_age"]

        for key in debug_keys:
            if key in df.columns and key in normalized_filters:
                value = normalized_filters[key]
                st.write(f"🔍 Filtering: {key} == {value}")
                if isinstance(value, str):
                    df = df[df[key].str.strip().str.lower() == value.strip().lower()]
                else:
                    df = df[df[key] == value]

        st.write("📄 Filtered Preview", df.head())
        return df

    except Exception as e:
        st.error(f"Error fetching file from S3: {e}")
        return pd.DataFrame()



def get_gemini_insight(df):
    if df.empty:
        return "No data to analyze."

    summary_data = df.head(10).to_string(index=False)

    prompt = f"""
    You are a real estate expert.

    Analyze the following property listings and recommend the best one, including clear reasoning and give them a bargain amount or the percentage.

    === Property Data ===
    {summary_data}
    =====================

    Please follow this format for the answer:

    Property Recommendation:
    [Your recommendation here]

    Reasoning:
    1. [Point 1]
    2. [Point 2]
    3. [Point 3]
    ...

    Concerns:
    1. [Concern 1]
    2. [Concern 2]
    ...

    Negotiation Advice:
    1. [Advice 1]
    2. [Advice 2]
    3. [Advice 3]
    ...

    Important Considerations Before Finalizing:
    1. [Consideration 1]
    2. [Consideration 2]
    3. [Consideration 3]

    Write the response in plain text.
    Avoid using stars (*), dashes (-), or any bullet symbols.
    Make the text look like a clean written report.
    """

    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash-001',
            contents=prompt
        )

        return response.text.strip()

    except Exception as e:
        return f"Gemini Error: {e}"



# 🚀 Main App
def main():
    st.title("🏡 Real Estate Property Insights for Customers")
    bucket_name = "magedataset"
    file_key = "real_state_data.csv"

    filters = get_common_inputs(is_rent=True)

    if st.button("Search Properties"):
        result_df = fetch_properties(bucket_name, file_key, filters)
        if not result_df.empty:
            st.success(f"✅ Found {len(result_df)} matching properties.")
            st.dataframe(result_df)
            insight = get_gemini_insight(result_df)
            st.text_area("🧠 OpenAI Recommendation", insight, height=200)
        else:
            st.warning("⚠️ No matching properties found.")

if __name__ == "__main__":
    main()
