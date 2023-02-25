import streamlit as st
import pandas as pd
import xgboost

# Title
st.header("Paris Housing Price - Machine Learning App")

squareMeters = st.number_input("Enter house size (in square meters):")
made = st.number_input("Enter year made:")
numPrevOwners = st.number_input("Enter number of previous owners:")
numberOfRooms = st.number_input("Enter number of rooms:")
floors = st.number_input("Enter number of floors:")

basement = st.number_input("Enter basement size (in square meters):")
attic = st.number_input("Enter attic size (in square meters):")
garage = st.number_input("Enter garage size (in square meters):")

cityCode = st.number_input("Enter building city code: ")
cityPartRange = st.number_input("Enter city part range: ")

isNewBuilt = st.selectbox("Is this house newly built?", ("Yes", "No"))

hasYard = st.selectbox("Does this house has a yard?", ("Yes", "No"))
hasPool = st.selectbox("Does this house has a pool?", ("Yes", "No"))
hasStormProtector = st.selectbox("Does this house has a storm protector?", ("Yes", "No"))
hasStorageRoom = st.selectbox("Does this house has a storage room?", ("Yes", "No"))
hasGuestRoom = st.number_input("Enter number of guest rooms this house has:")

preds = ['squareMeters', 'numberOfRooms', 'hasYard', 'hasPool',
 'floors', 'cityPartRange', 'numPrevOwners', 'made', 'isNewBuilt',
 'hasStormProtector', 'basement', 'attic', 'garage', 'hasStorageRoom',
 'hasGuestRoom']

# If button is pressed
if st.button("Submit"):

    # Unpickle classifier
    model = xgboost.XGBRegressor()
    model.load_model("model.bin")

    # Store inputs into dataframe
    X = pd.DataFrame([[squareMeters, numberOfRooms, hasYard, hasPool,
  floors, cityPartRange, numPrevOwners, made, isNewBuilt,
  hasStormProtector, basement, attic, garage, hasStorageRoom,
  hasGuestRoom]], columns=preds)
    X.replace(["Yes", "No"], [1, 0])

    # Get prediction
    prediction = model.predict(X)[0]

    # Output prediction
    st.text(f"The price of this house is {prediction}")
