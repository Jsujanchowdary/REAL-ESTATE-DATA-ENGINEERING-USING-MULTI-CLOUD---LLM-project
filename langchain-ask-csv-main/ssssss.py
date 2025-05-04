pip install pyspark
pip install boto3
import os
import boto3
os.environ["AWS_ACCESS_KEY_ID"] = "*********************"
os.environ["AWS_SECRET_ACCESS_KEY"] = "************************"

s3 = boto3.client('s3')
bucket = "magedataset"
key = "real_state_data.csv"
local_file = "/tmp/real_state_data.csv"
# Download the file
s3.download_file(bucket, key, local_file)
print("Download complete.")

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("RealEstateData").getOrCreate()
df = spark.read.csv(local_file, header=True, inferSchema=True)
df.printSchema()
df.show(5)

 Property_ID: string (nullable = true)
 |-- City: string (nullable = true)
 |-- Locality: string (nullable = true)
 |-- Property Purpose: string (nullable = true)
 |-- House Type: string (nullable = true)
 |-- BHK: integer (nullable = true)
 |-- House Floor: integer (nullable = true)
 |-- Building Floor: integer (nullable = true)
 |-- Property Age: integer (nullable = true)
 |-- Facing: string (nullable = true)
 |-- Square Feet: integer (nullable = true)
 |-- rent_montly_cost: integer (nullable = true)
 |-- Owner Price per Sqft: integer (nullable = true)
 |-- GOV Price per Sqft: integer (nullable = true)
 |-- Security: string (nullable = true)
 |-- Gym: string (nullable = true)
 |-- Convention Hall: string (nullable = true)
 |-- Parking: string (nullable = true)
 |-- Water Supply: string (nullable = true)
 |-- Bathrooms: integer (nullable = true)
 |-- Balcony: integer (nullable = true)
 |-- Availability Date: date (nullable = true)
 |-- Deposit Cost: integer (nullable = true)
 |-- Furnishing: string (nullable = true)
 |-- HOA Fees : double (nullable = true)
 |-- Flooring Type: string (nullable = true)
 |-- Roof Type: string (nullable = true)
 |-- Property Style: string (nullable = true)
 |-- View: string (nullable = true)
 |-- Nearby Schools: integer (nullable = true)
 |-- Nearby Hospitals: integer (nullable = true)
 |-- Nearby Supermarkets: integer (nullable = true)
 |-- IT Hub Present: string (nullable = true)
 |-- Swimming Pool: string (nullable = true)
 |-- Lot Shape: string (nullable = true)
 |-- Sewer System: string (nullable = true)
 |-- Address: string (nullable = true)
 |-- owner_name: string (nullable = true)
 |-- owner_email: string (nullable = true)
 |-- Owner Phone: string (nullable = true)
 |-- Pincode: integer (nullable = true)
 |-- State: string (nullable = true)
 |-- Photos: string (nullable = true)



from pyspark.sql.functions import trim, lower, col
string_columns = [col_name for col_name, dtype in df.dtypes if dtype == 'string']
for col_name in string_columns:
    df = df.withColumn(col_name, trim(lower(col(col_name))))
# Replace empty strings with nulls
from pyspark.sql.functions import when

for c in df.columns:
    df = df.withColumn(c, when(col(c) == "", None).otherwise(col(c)))
# Fill missing numeric values with 0 or median (simplified here as 0)
num_cols = [col_name for col_name, dtype in df.dtypes if dtype in ['int', 'double']]
df = df.fillna(0, subset=num_cols)
# Fill string columns with "unknown"
df = df.fillna("unknown", subset=string_columns)

# Step 3: Partition
buyer_df = df.filter(col("Property Purpose") == "buy").drop("rent_montly_cost", "Deposit Cost")
rental_df = df.filter(col("Property Purpose") == "rent").drop("Owner Price per Sqft", "GOV Price per Sqft")

from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline

# Define columns
numeric_features = [
    "BHK", "House Floor", "Building Floor", "Property Age", "Square Feet",
    "Bathrooms", "Balcony", "HOA Fees ", "Nearby Schools", "Nearby Hospitals",
    "Nearby Supermarkets"
]

categorical_features = [
    "City", "Locality", "Property Purpose", "House Type", "Facing", "Security", "Gym",
    "Convention Hall", "Parking", "Water Supply", "Furnishing", "Flooring Type",
    "Roof Type", "Property Style", "View", "IT Hub Present", "Swimming Pool",
    "Lot Shape", "Sewer System"
]

# You may drop or convert Availability Date and Photos later if needed



from pyspark.sql.functions import col, months_between, current_date, when, lower

# 1. BHK × Sqft
ml_df = buyer_df.withColumn("BHK_Sqft", col("BHK") * col("Square Feet"))

# 2. Months to Availability
ml_df = ml_df.withColumn("Months_To_Availability",months_between(col("Availability Date"), current_date()))

# 3. Has Photos
ml_df = ml_df.withColumn("Has_Photos", when(col("Photos").isNotNull(), 1).otherwise(0))

# 4. Amenity Score (gym, pool, security, convention hall)
ml_df = ml_df.withColumn("Amenity_Score",
    (when(lower(col("Gym")) == "yes", 1).otherwise(0) +
     when(lower(col("Swimming Pool")) == "yes", 1).otherwise(0) +
     when(lower(col("Security")) != "none", 1).otherwise(0) +
     when(lower(col("Convention Hall")) == "yes", 1).otherwise(0))
)

# 5. Is Near IT Hub
ml_df = ml_df.withColumn("Is_Near_IT_Hub", when(lower(col("IT Hub Present")) != "none", 1).otherwise(0))

new_numeric_features = [
    "BHK_Sqft", "Months_To_Availability", "Amenity_Score", "Has_Photos", "Is_Near_IT_Hub"
]

numeric_features += new_numeric_features

# Index and OneHotEncode categorical features
indexers = [StringIndexer(inputCol=col, outputCol=col + "_Index", handleInvalid='keep') for col in categorical_features]
encoders = [OneHotEncoder(inputCol=col + "_Index", outputCol=col + "_OHE") for col in categorical_features]

# Final input features (numeric + encoded)
encoded_features = [col + "_OHE" for col in categorical_features]
assembler_inputs = numeric_features + encoded_features

assembler = VectorAssembler(
    inputCols=assembler_inputs,
    outputCol="features"
)

# Target column
target_column = "Owner Price per Sqft"

pipeline = Pipeline(stages=indexers + encoders + [assembler])
pipeline_model = pipeline.fit(ml_df)
processed_df = pipeline_model.transform(ml_df)

train_df, test_df = processed_df.randomSplit([0.8, 0.2], seed=42)
from pyspark.ml.regression import LinearRegression

lr = LinearRegression(
    featuresCol="features",
    labelCol=target_column,
    predictionCol="prediction"
)

lr_model = lr.fit(train_df)
results = lr_model.evaluate(test_df)

print("RMSE:", results.rootMeanSquaredError)
print("R2:", results.r2)

RMSE: 3188.8767498553666
R2: 0.8178149248860942

user_input = {
    "City": "chennai",
    "Locality": "adyar",
    "Property Purpose": "buy",
    "House Type": "villa",
    "BHK": 4,
    "House Floor": 0,
    "Building Floor": 2,
    "Property Age": 3,
    "Facing": "north-east",
    "Square Feet": 3200,
    "Security": "cctv",
    "Gym": "no",
    "Convention Hall": "no",
    "Parking": "car",
    "Water Supply": "corporation",
    "Bathrooms": 3,
    "Balcony": 3,
    "Availability Date": "2025-04-10",
    "Furnishing": "full",
    "HOA Fees ": 1500.0,
    "Flooring Type": "tiles",
    "Roof Type": "concrete",
    "Property Style": "modern",
    "View": "city view",
    "Nearby Schools": 4,
    "Nearby Hospitals": 4,
    "Nearby Supermarkets": 5,
    "IT Hub Present": "tcs",
    "Swimming Pool": "yes",
    "Lot Shape": "rectangular",
    "Sewer System": "public",
    "Photos": "yes"
}

from pyspark.sql import Row
from pyspark.sql.functions import col, when, months_between, current_date, lower

# 1. Create the user input DataFrame
user_row = Row(**user_input)
user_df = spark.createDataFrame([user_row])

# 2. Manually add engineered features
user_df = user_df.withColumn("BHK_Sqft", col("BHK") * col("Square Feet"))

user_df = user_df.withColumn("Months_To_Availability",
                             months_between(col("Availability Date"), current_date()))

user_df = user_df.withColumn("Has_Photos", when(col("Photos").isNotNull(), 1).otherwise(0))

user_df = user_df.withColumn("Amenity_Score",
    (when(lower(col("Gym")) == "yes", 1).otherwise(0) +
     when(lower(col("Swimming Pool")) == "yes", 1).otherwise(0) +
     when(lower(col("Security")) != "none", 1).otherwise(0) +
     when(lower(col("Convention Hall")) == "yes", 1).otherwise(0))
)

user_df = user_df.withColumn("Is_Near_IT_Hub",
                             when(lower(col("IT Hub Present")) != "none", 1).otherwise(0))

# Transform and predict
user_transformed = pipeline_model.transform(user_df)
prediction = lr_model.transform(user_transformed)
prediction.select("prediction").show()
---------------+
|     prediction|
+---------------+
|20715.184733821|
+---------------+


from pyspark.sql.functions import round as spark_round

# 2. Add Total_Predicted_Price column
prediction = prediction.withColumn(
    "Total_Predicted_Price", 
    spark_round(col("prediction") * col("Square Feet"))
)

# 3. Get values from prediction DataFrame
row = prediction.select("prediction", "Square Feet", "Total_Predicted_Price").first()

# 4. Extract values
predicted_price_per_sqft = round(row["prediction"])
total_price = row["Total_Predicted_Price"]

# 🎉 Display nicely
print(f"🏷 Predicted Price per Sqft: ₹{predicted_price_per_sqft:,}")
print(f"🏠 Total Estimated Property Price: ₹{total_price:,}")


 Predicted Price per Sqft: ₹20,715
🏠 Total Estimated Property Price: ₹66,288,591.0


from pyspark.sql.functions import round as spark_round

# 2. Add Total_Predicted_Price column
prediction = prediction.withColumn(
    "Total_Predicted_Price", 
    spark_round(col("prediction") * col("Square Feet"))
)

# 3. Get values from prediction DataFrame
row = prediction.select("prediction", "Square Feet", "Total_Predicted_Price").first()

# 4. Extract values
predicted_price_per_sqft = round(row["prediction"])
total_price = row["Total_Predicted_Price"]

# 🎉 Display nicely
print(f"🏷 Predicted Price per Sqft: ₹{predicted_price_per_sqft:,}")
print(f"🏠 Total Estimated Property Price: ₹{total_price:,}")

 Predicted Price per Sqft: ₹20,768
🏠 Total Estimated Property Price: ₹66,457,444.0

# Save regression model
lr_model.save("house_price_model")

# Save preprocessing pipeline
pipeline_model.save("preprocessing_pipeline")

from pyspark.sql.functions import col

rental_df = rental_df.filter(col("Property Purpose") == "rent")


numeric_features = [
    "BHK", "House Floor", "Building Floor", "Property Age", "Square Feet",
    "Bathrooms", "Balcony", "Deposit Cost", "HOA Fees ",
    "Nearby Schools", "Nearby Hospitals", "Nearby Supermarkets"
]

categorical_features = [
    "City", "Locality", "House Type", "Facing", "Security", "Gym", "Convention Hall",
    "Parking", "Water Supply", "Furnishing", "Flooring Type", "Roof Type",
    "Property Style", "View", "IT Hub Present", "Swimming Pool",
    "Lot Shape", "Sewer System"
]

from pyspark.sql.functions import months_between, current_date, when, lower

rental_df = rental_df.withColumn("BHK_Sqft", col("BHK") * col("Square Feet"))

rental_df = rental_df.withColumn("Months_To_Availability",
    months_between(col("Availability Date"), current_date()))

rental_df = rental_df.withColumn("Has_Photos", when(col("Photos").isNotNull(), 1).otherwise(0))

rental_df = rental_df.withColumn("Amenity_Score",
    (when(lower(col("Gym")) == "yes", 1).otherwise(0) +
     when(lower(col("Swimming Pool")) == "yes", 1).otherwise(0) +
     when(lower(col("Security")) != "none", 1).otherwise(0) +
     when(lower(col("Convention Hall")) == "yes", 1).otherwise(0))
)

rental_df = rental_df.withColumn("Is_Near_IT_Hub",
    when(lower(col("IT Hub Present")) != "none", 1).otherwise(0))

numeric_features += ["BHK_Sqft", "Months_To_Availability", "Has_Photos", "Amenity_Score", "Is_Near_IT_Hub"]
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline

# Step 4.1: Index and Encode Categorical
indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_Index", handleInvalid="keep") for c in categorical_features]
encoders = [OneHotEncoder(inputCol=f"{c}_Index", outputCol=f"{c}_OHE") for c in categorical_features]

# Step 4.2: Assemble Features
encoded_features = [f"{c}_OHE" for c in categorical_features]
assembler_inputs = numeric_features + encoded_features

assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

# Step 4.3: Pipeline
pipeline = Pipeline(stages=indexers + encoders + [assembler])
pipeline_model = pipeline.fit(rental_df)
processed_df = pipeline_model.transform(rental_df)


train_df, test_df = processed_df.randomSplit([0.8, 0.2], seed=42)

from pyspark.ml.regression import LinearRegression

lr = LinearRegression(
    featuresCol="features",
    labelCol="rent_montly_cost",
    predictionCol="prediction"
)

lr_model = lr.fit(train_df)
results = lr_model.evaluate(test_df)

print("✅ Rental Model Performance")
print("RMSE:", results.rootMeanSquaredError)
print("R²:", results.r2)

✅ Rental Model Performance
RMSE: 11004.180780888408
R²: 0.5816076317772093

user_input_rent = {
    "City": "chennai",
    "Locality": "velachery",
    "Property Purpose": "rent",
    "House Type": "apartment",
    "BHK": 2,
    "House Floor": 3,
    "Building Floor": 5,
    "Property Age": 8,
    "Facing": "north-east",
    "Square Feet": 1700,
    "Security": "cctv",
    "Gym": "yes",
    "Convention Hall": "no",
    "Parking": "both",
    "Water Supply": "corporation",
    "Bathrooms": 2,
    "Balcony": 1,
    "Availability Date": "2025-04-15",
    "Deposit Cost": 50000,
    "Furnishing": "semi",
    "HOA Fees ": 1800.0,
    "Flooring Type": "tiles",
    "Roof Type": "concrete",
    "Property Style": "modern",
    "View": "city view",
    "Nearby Schools": 3,
    "Nearby Hospitals": 2,
    "Nearby Supermarkets": 3,
    "IT Hub Present": "tcs",
    "Swimming Pool": "no",
    "Lot Shape": "rectangular",
    "Sewer System": "public",
    "Photos": "yes"
}

from pyspark.sql import Row
from pyspark.sql.functions import col, months_between, current_date, when, lower

# Convert to DataFrame
user_row = Row(**user_input_rent)
user_rent_df = spark.createDataFrame([user_row])

# Add engineered features
user_rent_df = user_rent_df.withColumn("BHK_Sqft", col("BHK") * col("Square Feet"))

user_rent_df = user_rent_df.withColumn("Months_To_Availability",
                                       months_between(col("Availability Date"), current_date()))

user_rent_df = user_rent_df.withColumn("Has_Photos", when(col("Photos").isNotNull(), 1).otherwise(0))

user_rent_df = user_rent_df.withColumn("Amenity_Score",
    (when(lower(col("Gym")) == "yes", 1).otherwise(0) +
     when(lower(col("Swimming Pool")) == "yes", 1).otherwise(0) +
     when(lower(col("Security")) != "none", 1).otherwise(0) +
     when(lower(col("Convention Hall")) == "yes", 1).otherwise(0))
)

user_rent_df = user_rent_df.withColumn("Is_Near_IT_Hub",
                                       when(lower(col("IT Hub Present")) != "none", 1).otherwise(0))

# Transform input
user_transformed = pipeline_model.transform(user_rent_df)

# Predict
prediction = lr_model.transform(user_transformed)

# Format Output
row = prediction.select("prediction").first()
predicted_rent = round(row["prediction"])

print(f"🏷 Predicted Monthly Rent: ₹{predicted_rent:,}")

 Predicted Monthly Rent: ₹33,203

!zip -r house_price_model.zip house_price_model
!zip -r preprocessing_pipeline.zip preprocessing_pipeline

!zip -r rental_model.zip rental_model
!zip -r rental_pipeline.zip rental_pipeline