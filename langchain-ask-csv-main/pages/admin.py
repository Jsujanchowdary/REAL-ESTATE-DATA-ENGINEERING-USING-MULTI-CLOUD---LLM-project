import streamlit as st
import requests
import streamlit_lottie
import pandas as pd
import boto3
import json
from io import StringIO
import uuid

st.set_page_config(page_title="PropIntel", page_icon="🏠", layout="wide")

def load_lottieur(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

l1 = "https://lottie.host/20ddb2ba-92af-4a75-8efd-8e04beeb5dd7/vZ9Xa4h9ug.json"

# 🔹 Hide Sidebar for Admin Dashboard
st.markdown("""
    <style>
    [data-testid="stSidebar"] {visibility: hidden;}
    .logout-container {
        position: absolute;
        top: 10px;
        right: 20px;
        z-index: 1000;
    }
    </style>
    """, unsafe_allow_html=True)

# 🔹 Ensure User is Logged In
if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
    st.warning("You must be logged in to access this page.")
    if st.button("Go to Login", use_container_width=True):
        st.switch_page("/Users/jsujanchowdary/Downloads/langchain-ask-csv-main/main.py")  # ✅ Redirect to Login Page
    st.stop()

# st.success(f"Welcome, {st.session_state['user_name']}! You are logged in as an Admin.")


# 🔹 Move Logout Button to the Top Right
logout_placeholder = st.empty()
with logout_placeholder.container():
    col1, col2 = st.columns([7, 1])  # Right-align using columns
    with col1:
        st.title(f"Hello, {st.session_state['user_name']}")

    with col2:
        if st.button("Logout", use_container_width=True):
            st.session_state.clear()  # ✅ Clears all session data
            st.switch_page("/Users/jsujanchowdary/Downloads/langchain-ask-csv-main/main.py")  # ✅ Redirect to login page

col3, col4 = st.columns(2)
with col4:
    st.lottie(l1)


# AWS S3 Configuration
S3_BUCKET = "magedataset"
S3_FILE_KEY = "real_state_data.csv"
AWS_ACCESS_KEY = "key"
AWS_SECRET_KEY = "key"
AWS_REGION = "us-east-1"


# Function to Load Data from S3
def load_data_from_s3():
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION,
    )
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_FILE_KEY)
        df = pd.read_csv(obj["Body"])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Function to Save Data to S3
def save_data_to_s3(df):
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION,
    )
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=S3_BUCKET, Key=S3_FILE_KEY, Body=csv_buffer.getvalue())
    st.success("Property data saved successfully!")

# Function to Generate Unique Property ID
def generate_unique_property_id(existing_df):
    while True:
        new_id = f"PROP-{uuid.uuid4().hex[:6].upper()}"  # Generates a random 6-character alphanumeric ID
        if new_id not in existing_df["Property_ID"].values:
            return new_id

# Streamlit UI
st.title("Owner Property Data Entry")

# Load existing data
df = load_data_from_s3()

with st.expander("Basic Property Details", expanded=True):
    city = st.selectbox("City", ["Chennai"], index=0)
    locality = st.selectbox("Locality", ["Adyar", "Anna Nagar", "T. Nagar", "Velachery", "OMR", "Tambaram", "Porur", "Guindy", "Mylapore", "Kilpauk", "Perambur", "Vadapalani", "Kodambakkam", "Royapettah", "Egmore", "Pallavaram", "Avadi", "Chromepet", "Sholinganallur", "Ambattur"], index=0)
    property_purpose = st.radio("Property Purpose", ["Rent", "Sale"], horizontal=True)
    house_type = st.selectbox("House Type", ["apartment", "individual house", "villa", "gated community"], index=0)
    bhk = st.selectbox("BHK", list(range(1, 7)), index=2)
    sqft = st.number_input("Square Feet", min_value=100, step=10, value=1000)
    sqft_cost = st.number_input("Cost per Square Feet", min_value=1000, step=100, value=5000)


with st.expander("Building Details"):
    house_floor = st.selectbox("House Floor", list(range(1, 16)), index=0)
    building_floor = st.selectbox("Building Floor", list(range(1, 16)), index=4)
    property_age = st.number_input("Property Age (years)", min_value=0, step=1, value=5)
    facing = st.selectbox("Facing Direction", ["North", "South", "East", "West", "North-East", "North-West", "South-East", "South-West"], index=0)
    photos = st.text_input("past the image links")
    availability_date = st.date_input("Availability Date")
    furnishing = st.selectbox("Furnishing Type", ["full", "semi", "unfurnished"])
    flooring_type = st.selectbox("Flooring Type", ["Marble", "Wood", "Tiles", "Vinyl", "Laminate"])
    roof_type = st.selectbox("Roof Type", ["Concrete", "Tile", "Metal", "Slate"])
    property_style = st.selectbox("Property Style", ["Modern", "Traditional", "Victorian", "Contemporary"])
    sewer_system = st.selectbox("Sewer System", ["Public", "Septic"])

with st.expander("Amenities"):
    security = st.multiselect("Security Features", ["Police Station Nearby", "CCTV", "Gated Community", "Security Guards"], default=["CCTV", "Gated Community"])
    gym = st.checkbox("Gym Available", value=True)
    swimming_pool = st.checkbox("Swimming Pool Available", value=False)
    convention_hall = st.checkbox("Convention Hall Available", value=False)
    parking = st.selectbox("Parking Availability", ["car", "bike", "both", "no parking"], index=2)
    property_conditions = st.selectbox("Current Property Condition", ["vacant", "tenant on notice", "new property"])
    bathrooms = st.number_input("Number of Bathrooms", min_value=1, step=1)
    balcony = st.number_input("Number of Balconies", min_value=0, step=1)
    water_supply = st.selectbox("Water Supply", ["corporation", "borewell", "both"])
    view = st.selectbox("View", ["Sea View", "City View", "Garden View", "Lake View", "None"])
    lot_shape = st.selectbox("Lot Shape", ["Rectangular", "Irregular", "Square"])

with st.expander("Nearby property"):
    nearby_schools = st.text_input("Nearby Schools")
    nearby_hospitals = st.text_input("Nearby Hospitals")
    nearby_supermarkets = st.text_input("Nearby Supermarkets")
    it_hub_present = st.checkbox("IT Hub Present Nearby")

with st.expander("Phots of Property"):
    photos = st.text_input("Past the Property image links")

with st.expander("Financial Details"):
    deposit_cost = st.number_input("Deposit Cost", min_value=1000, step=1000, value=50000)
    hoa_fees = st.number_input("HOA Fees ", min_value=0, step=100, value=500)
    maint_fee = st.number_input("Maintenance Fee per Sqft", min_value=1, step=1, value=2)

with st.expander("Owner Details"):
    owner_name = st.text_input("Owner Name")
    owner_email = st.text_input("Owner Email")
    address = st.text_area("Full Address")
    pincode = st.text_input("Pincode")
    state = st.text_input("State", value="Tamil Nadu")

if st.button("Submit Property Data"):
    new_entry = {
        "City": city,
        "Locality": locality,
        "Property Purpose": property_purpose,
        "House Type": house_type,
        "BHK": bhk,
        "House Floor": house_floor,
        "Building Floor": building_floor,
        "Property Age": property_age,
        "Facing": facing,
        "Square Feet": sqft,
        "Total sqft Cost": sqft * sqft_cost,
        "Security": ", ".join(security),
        "Maintenance Fee": maint_fee * sqft,
        "Gym": gym,
        "Convention Hall": convention_hall,
        "Property Condition":property_conditions,
        "Bathrooms": bathrooms,
        "Balcony": balcony,
        "Water Supply": water_supply,
        "Parking": parking,
        "Availability Date": availability_date,
        "Deposit Cost": deposit_cost,
        "Furnishing": furnishing,
        "HOA Fees ": hoa_fees,
        "Flooring Type": flooring_type,
        "Roof Type": roof_type,
        "Property Style": property_style,
        "View": view,
        "Nearby Schools": nearby_schools,
        "Nearby Hospitals": nearby_hospitals,
        "Nearby Supermarkets": nearby_supermarkets,
        "IT Hub Present": it_hub_present,
        "Address": address,
        "Owner_Name": owner_name,
        "Owner Email": owner_email,
        "Pincode": pincode,
        "State": state,
        "Swimming Pool": swimming_pool,
        "Photos": photos,
        "Lot Shape": lot_shape,
        "Sewer System": sewer_system
    }
    
    existing_records = df.drop(columns=["Property_ID"], errors="ignore")
    if any(existing_records.apply(lambda row: row.to_dict() == new_entry, axis=1)):
        st.error("Property already exists in the database!")
    else:
        property_id = generate_unique_property_id(df)
        new_entry["Property_ID"] = property_id
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        save_data_to_s3(df)
        st.success(f"Property added successfully! Generated Property ID: {property_id}")
