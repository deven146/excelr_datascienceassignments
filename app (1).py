import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Titanic Survival Prediction", layout="wide")
@st.cache_resource
def load_model():
    try:
        model = joblib.load('titanic_logistic_model.joblib')
        return model
    except FileNotFoundError:
        st.error("Model file ('titanic_logistic_model.joblib') not found. "
                 "Please ensure it's in the same folder as app.py and was saved correctly.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data 
def load_columns():
    try:
        model_columns = joblib.load('model_columns.joblib')
        return model_columns
    except FileNotFoundError:
        st.error("Model columns file ('model_columns.joblib') not found. "
                 "Please ensure it's in the same folder as app.py and was saved correctly.")
        return None
    except Exception as e:
        st.error(f"Error loading columns: {e}")
        return None

model = load_model()
model_columns = load_columns()



st.title("🚢 Titanic Survival Prediction")
st.markdown("Enter passenger details below to predict their survival probability.")

st.sidebar.header("Passenger Features")

col1, col2 = st.sidebar.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class (Pclass)", options=[1, 2, 3], index=2, help="1=1st, 2=2nd, 3=3rd")
    sex = st.radio("Sex", options=['male', 'female'], index=1) # Default to female
    age = st.slider("Age", min_value=0, max_value=85, value=30, step=1)

with col2:
    sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0, step=1)
    parch = st.number_input("Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0, step=1)
    fare_max = 65.65
    fare = st.number_input("Fare Paid (£)", min_value=0.0, max_value=fare_max, value=30.0, step=0.5, format="%.2f",
                           help=f"Max based on training data capping: approx £{fare_max:.2f}")
    embarked = st.selectbox("Port of Embarkation", options=['S', 'C', 'Q'], index=0,
                            help="S=Southampton, C=Cherbourg, Q=Queenstown")

def preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked, model_cols):
        input_data_raw = {
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Sex': [sex],        
        'Embarked': [embarked],
        'Pclass': [pclass]     
    }
    input_df = pd.DataFrame.from_dict(input_data_raw)

    input_df['Sex_male'] = (input_df['Sex'] == 'male').astype(int)
    input_df.drop('Sex', axis=1, inplace=True)

  
    input_df['Embarked_C'] = (input_df['Embarked'] == 'C').astype(int)
    input_df['Embarked_Q'] = (input_df['Embarked'] == 'Q').astype(int)
    input_df['Embarked_S'] = (input_df['Embarked'] == 'S').astype(int)
    input_df.drop('Embarked', axis=1, inplace=True)

   
    input_df['Pclass_1'] = (input_df['Pclass'] == 1).astype(int)
    input_df['Pclass_2'] = (input_df['Pclass'] == 2).astype(int)
    input_df['Pclass_3'] = (input_df['Pclass'] == 3).astype(int)
    input_df.drop('Pclass', axis=1, inplace=True)

    input_df_processed = input_df.reindex(columns=model_cols, fill_value=0)

    return input_df_processed

if model is not None and model_columns is not None:
    if st.sidebar.button("Predict Survival"):
        input_processed_df = preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked, model_columns)
        input_array = input_processed_df.values # Convert to NumPy array for prediction

        # 2. Make Prediction (directly on unscaled data)
        try:
            prediction = model.predict(input_array)
            probability = model.predict_proba(input_array)

            # 3. Display Result on the main page
            st.subheader("Prediction Result")
            if prediction[0] == 1:
                st.success("Prediction: **Likely Survived** 👍")
                # Display probability of survival (class 1)
                st.write(f"Probability of Survival: {probability[0][1]*100:.2f}%")
            else:
                st.error("Prediction: **Likely Did Not Survive** 👎")
                # Still display probability of survival (class 1)
                st.write(f"Probability of Survival: {probability[0][1]*100:.2f}%")

            st.subheader("Input Features Provided (After Processing):")

            st.dataframe(input_processed_df.style.format("{:.2f}", subset=['Age', 'Fare']))

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.error("Please ensure the model and columns files are correct and match the "
                     "preprocessing steps used during training.")

# Add a message if the model or columns couldn't be loaded
else:
    st.sidebar.error("Critical error: Model or column information could not be loaded. "
                 "Predictions cannot be made. Please check your .joblib files.")

st.sidebar.markdown("---")
st.sidebar.markdown("Created for Titanic dataset analysis.")