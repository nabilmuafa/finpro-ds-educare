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

boolean_cols = ['hasYard', 'hasPool', 'isNewBuilt', 'hasStormProtector']

preds = ['squareMeters', 'numberOfRooms',
 'floors', 'cityPartRange', 'numPrevOwners', 'made',
 'basement', 'attic', 'garage', 'hasStorageRoom',
 'hasGuestRoom', 'countFac']

# If button is pressed
if st.button("Submit"):

    # Unpickle classifier
    model = xgboost.XGBRegressor()
    model.load_model("model.bin")

    # Store inputs into dataframe
    X = pd.DataFrame([[squareMeters, numberOfRooms,
  floors, cityPartRange, numPrevOwners, made,
  basement, attic, garage, hasStorageRoom,
  hasGuestRoom]], columns=preds)
    X = X.replace(["Yes", "No"], [1, 0])
    X['countFac']=X[boolean_cols].sum(axis=1)
    X.drop(boolean_cols, inplace=True)

    # Get prediction
    prediction = model.predict(X)[0]

    # Output prediction
    st.text(f"The price of this house is {prediction}")
