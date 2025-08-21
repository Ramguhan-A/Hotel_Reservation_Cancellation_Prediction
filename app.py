import streamlit as st
from config.paths_config import *
import pandas as pd
from utils.common_function import load_data
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
from src.logger import get_logger

logger = get_logger(__name__)

# from src.data_preprocessing import 
# import sha
st.write("""
         # 🏨 Hotel Reservation Cancellation Prediction App
         
         This app predicts Whether the customer will **Cancel the Reservation or Not**
         """)
st.write("---")

# Loads the Data set

Hotel_reservation_df = load_data(RAW_FILE_PATH)
X = Hotel_reservation_df.drop(columns="booking_status")
y = Hotel_reservation_df["booking_status"]

# Data input for prediction (sidebar)

st.sidebar.header("Specify Input Parameters")

def user_input_features():
    Booking_ID = st.sidebar.text_input(label="Enter Customer ID ex: INN00002")
    no_of_adults = st.sidebar.slider("no_of_adults", X.no_of_adults.min(), X.no_of_adults.max(), X.no_of_adults.mean().astype(int))
    no_of_children = st.sidebar.slider("no_of_children", X.no_of_children.min(), X.no_of_children.max(), X.no_of_children.mean().astype(int))
    no_of_weekend_nights = st.sidebar.slider("no_of_weekend_nights", X.no_of_weekend_nights.min(), X.no_of_weekend_nights.max(), X.no_of_weekend_nights.mean().astype(int))
    no_of_week_nights = st.sidebar.slider("no_of_week_nights", X.no_of_week_nights.min(), X.no_of_week_nights.max(), X.no_of_week_nights.mean().astype(int))
    type_of_meal_plan = st.sidebar.selectbox("type_of_meal_plan", {"Meal Plan 1","Meal Plan 2", "Meal Plan 3" ,"None"})
    
    # Mapping again to values
    mapping_type_of_meal_plan = { "Meal Plan 1": "Meal Plan 1" ,"Meal Plan 2": "Meal Plan 2", "Meal Plan 3":"Meal Plan 3", "None": "Not Selected"}
    type_of_meal_plan = mapping_type_of_meal_plan[type_of_meal_plan]
    
    required_car_parking_space = st.sidebar.selectbox("required_car_parking_space", {"Yes", "No"})
    
    # Mapping again to values
    mapping_required_car_parking_space = {"Yes": 1, "No": 0}
    required_car_parking_space = mapping_required_car_parking_space[required_car_parking_space]
    
    room_type_reserved = st.sidebar.selectbox("Room type", ("Room_Type 1", "Room_Type 2", "Room_Type 3", "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7" ))
    lead_time = st.sidebar.number_input("Lead time", 0, 500, step=1)
    arrival_year = st.sidebar.number_input("Arrival year",2017,2030, step=1)
    arrival_month = st.sidebar.number_input("Arrival Month", 1, 12, step=1)
    arrival_date = st.sidebar.number_input("Arrival date",1, 31, step=1)
    market_segment_type = st.sidebar.selectbox("Market segment type", {"Aviation", "Complementary", "Corporate", "Offline", "Online"})
    repeated_guest = st.sidebar.selectbox("Repeated guest",["Yes", "No"])
    
    # Mapping again to values
    mapping_requested_guest = {"Yes": 1, "No": 0}
    repeated_guest = mapping_requested_guest[repeated_guest]
    
    no_of_previous_cancellations = st.sidebar.slider("No of previous cancellation", 0, 10)
    no_of_previous_bookings_not_canceled = st.sidebar.slider("No of previous booking not cancelled", 0, 10)
    avg_price_per_room = st.sidebar.number_input("Avg price per room", 0, 200)
    no_of_special_requests = st.sidebar.slider("No of special requests",0, 5)
    submit = st.sidebar.button(label = "Submit")
                          
    data = {"Booking_ID": Booking_ID,
            "no_of_adults": no_of_adults,
            "no_of_children": no_of_children,
            "no_of_weekend_nights": no_of_weekend_nights,
            "no_of_week_nights": no_of_week_nights,
            "type_of_meal_plan": type_of_meal_plan,
            "required_car_parking_space": required_car_parking_space,
            "room_type_reserved": room_type_reserved,
            "lead_time": lead_time,
            "arrival_year": arrival_year,
            "arrival_month": arrival_month,
            "arrival_date": arrival_date,
            "market_segment_type": market_segment_type,
            "repeated_guest": repeated_guest,
            "no_of_previous_cancellations": no_of_previous_cancellations,
            "no_of_previous_bookings_not_canceled": no_of_previous_bookings_not_canceled,
            "avg_price_per_room": avg_price_per_room,
            "no_of_special_requests": no_of_special_requests}
    features = pd.DataFrame(data, index=[0])
    return features, submit


df, submit = user_input_features()

# Print specific input parameters
st.header("Specified Input Parameter") 
st.write(df)
st.write("---")

# pkl paths
processing_path = os.path.join("artifacts","pipeline","preprocessing_pipeline.pkl")
model_path = os.path.join("artifacts","models","lgbm_model_2.pkl")

preprocessing_artifacts = joblib.load(processing_path)
loaded_model = joblib.load(model_path)

# Applying transformer

label_encoder = preprocessing_artifacts["label_encoders"]
selected_features  = preprocessing_artifacts["selected_features"]
skewed_columns  = preprocessing_artifacts["skewed_columns"]

if submit:
    
    for col, le in label_encoder.items():
        if col in df.columns:
            df[col] = le.transform(df[col])
            
    for col in skewed_columns:
        if col in df.columns:
            df[col] = np.log1p(df[col])
        
    df = df[selected_features]

    prediction = loaded_model.predict(df)

    st.header("Prediction of Cancellation")
    
    if prediction[0] == 1:
        st.success("🟢 Prediction: Customer has a high chance of keeping the reservation")
    else:
        st.error("🔴 Prediction: Customer has a low chance of keeping the reservation")

    st.write("---")

    # Feature importance graph using shap
    
    explainer = shap.TreeExplainer(loaded_model)
    shap_values = explainer.shap_values(df)

    if isinstance(shap_values, list):
        shap_values = shap_values[1] 

    st.header('Feature Importance (SHAP)')

    # SHAP plotting (Local SHAP - passing only the prediction row to see individual prediction made by a model.)
    #  Feature importance plot tells on what basis our model is predicting that particular class) 
    #  This helps us understand why the model made a particular decision for that unique case.
    fig = plt.figure()
    shap.summary_plot(shap_values, df, plot_type="bar", show=False)
    st.pyplot(fig, bbox_inches='tight')
    st.write('---')