import streamlit as st
import pandas as pd
import pickle
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# Set up the app title and image
st.title('Traffic Volume Predictor')
st.write("This app predicts traffic volume based on weather and time-related features.")
st.image('traffic_image.gif', use_column_width=True)

# Load the model and dataset with caching

with open('traffic.pickle', 'rb') as model_pickle:
    reg_model = pickle.load(model_pickle)
    default_df = pd.read_csv('cleandata.csv')


# Confidence interval slider
alpha = st.slider("Select alpha for confidence level", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
st.write(f"Confidence Interval Alpha: {alpha}")



default_df.fillna({'holiday': 'None'}, inplace=True)

# Sidebar for feature inputs
st.sidebar.image('traffic_sidebar.jpg', use_column_width=True)
st.sidebar.header('Input Features')

# Option 1: Manual Data Entry Form
with st.sidebar.expander("Option 1: Fill out Form"):
    with st.form("user_inputs_form"):
        st.subheader("Enter the traffic details manually using the form below")
        holiday = st.selectbox("Choose holiday", [
            None, "Christmas Day", "Columbus Day", "Independence Day", "Labor Day",
            "Martin Luther King Jr Day", "Memorial Day", "New Years Day", "State Fair",
            "Thanksgiving Day", "Veterans Day", "Washingtons Birthday"
        ])
        temp = st.number_input('Average temperature in Kelvin', min_value=200.0, max_value=350.0, value=290.0)
        rain_1h = st.number_input('Amount of rain in mm', min_value=0.0, max_value=50.0, value=0.0)
        snow_1h = st.number_input('Amount of snow in mm', min_value=0.0, max_value=50.0, value=0.0)
        clouds_all = st.number_input('Cloud cover percentage', min_value=0, max_value=100, value=50)
        weather_main = st.selectbox('Current weather', ['Clear', 'Clouds', 'Rain', 'Snow', 'Mist','Drizzle','Haze','Thunderstorm','Fog','Smoke','Squall'])
        month = st.selectbox('Month', [
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ])
        day_of_week = st.selectbox('Day of the Week', ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])
        hour = st.selectbox('Hour of the Day', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])
        submit_button = st.form_submit_button("Submit Form Data")

# Process form submission and make predictions
if submit_button:
    # Create a DataFrame from user inputs
    default_df.loc[len(default_df)]= [holiday,temp,rain_1h,snow_1h,clouds_all,weather_main,month,day_of_week,hour] 


    default_df['hour']=default_df['hour'].astype('object')
    
    

    input_df=pd.get_dummies(default_df)
    userinput=input_df.tail(1)
    

    # Make prediction with confidence intervals

    prediction, intervals = reg_model.predict(userinput, alpha=alpha)
    pred_value = prediction[0]
    lower_limit = max(0, intervals[0][0])
    upper_limit = max(0, intervals[0][1])

    # Display results
    st.metric(label="Predicted Traffic Volume", value=f"{int(pred_value):,}")
    st.write(f"**Confidence Interval ({100 * (1 - alpha):.0f}%):** [{int(lower_limit):,}, {int(upper_limit):,}]")




# Option 2: File Upload Prediction
with st.sidebar.expander("Option 2: Upload a CSV file"):
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    st.write("Example of the file format to upload:")
    st.write(default_df.head())

# Process uploaded file data 
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Drop the 'weekday' column if it exists
    if 'weekday' in df.columns:
        df = df.drop('weekday', axis=1)

    # Ensure all categorical features (except 'hour') are treated as strings
    categorical_features = ['holiday', 'weather_main', 'month', 'day_of_week']
    for feature in categorical_features:
        if feature in df.columns:
            df[feature] = df[feature].astype(str)

    # Convert 'hour' to string for one-hot encoding
    if 'hour' in df.columns:
        df['hour'] = df['hour'].astype(object)

    # Concatenate the uploaded DataFrame with the sample DataFrame (default_df)
    combined_df = pd.concat([default_df, df], ignore_index=True)

    # One-hot encode the entire DataFrame, including 'hour'
    encoded_combined_df = pd.get_dummies(combined_df, columns=categorical_features + ['hour'])

    # Drop any duplicate columns
    encoded_combined_df = encoded_combined_df.loc[:, ~encoded_combined_df.columns.duplicated()]

    # Infer the feature names from the sample encoded DataFrame (default_df)
    sample_encoded = pd.get_dummies(default_df, columns=categorical_features + ['hour'])
    model_features = sample_encoded.columns.to_list()

    # Reindex the combined DataFrame to match the model's expected features
    encoded_combined_df = encoded_combined_df.reindex(columns=model_features, fill_value=0)

    # Extract only the user's data (the tail of the combined DataFrame)
    user_encoded = encoded_combined_df.tail(len(df))

    # Make predictions
    predictions, intervals = reg_model.predict(user_encoded, alpha=alpha)

    # Check length match

    df['Predicted Volume'] = predictions
    df['Lower_CI'] = intervals[:, 0]
    df['Upper_CI'] = intervals[:, 1]

    # Display the full DataFrame
    st.write("### Predicted Traffic Volume with Confidence Intervals")
    st.dataframe(df)


# Model Insights Section
st.subheader("Model Insights")
tab1, tab2, tab3, tab4 = st.tabs(["Feature Importance", "Histogram of Residuals", "Predicted Vs. Actual", "Coverage Plot"])

with tab1:
    st.image('feature_imp_traffic.svg')
    st.caption("Relative importance of features in prediction.")

with tab2:
    st.image('traffic_residuals.svg')
    st.caption("Distribution of residuals.")

with tab3:
    st.image('pred_vs_actual_traffic.svg')
    st.caption("Comparison of predicted and actual traffic volume values.")

with tab4:
    st.image('coverage_traffic.svg')
    st.caption("Prediction intervals for traffic volume predictions.")
