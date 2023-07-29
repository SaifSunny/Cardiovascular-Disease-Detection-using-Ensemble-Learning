import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

st.title('Cardiovascular Disease Prediction Application')
st.write('''
         Please fill in the attributes below, then hit the Predict button
         to get your results. 
         ''')

st.header('Input Attributes')
ag = st.slider('Your Age (Years)', min_value=0.0, max_value=120.0, value=60.0, step=0.2)
st.write(''' ''')
gen = st.radio("Your Gender", ('Male', 'Female'))
st.write(''' ''')
height = st.slider('Your Height (cm)', min_value=0.0, max_value=300.0, value=150.0, step=1.0)
st.write(''' ''')
weight = st.slider('Your Weight (Kg.)', min_value=0.0, max_value=300.0, value=150.0, step=1.0)
st.write(''' ''')
ap_hi = st.slider('Systolic Blood Pressure', min_value=0.0, max_value=300.0, value=150.0, step=1.0)
st.write(''' ''')
ap_lo = st.slider('Diastolic Blood Pressure', min_value=0.0, max_value=300.0, value=150.0, step=1.0)
st.write(''' ''')
chol = st.radio("Cholesterol Amount", ('Normal', 'Above Normal', 'Well Above Normal'))
st.write(''' ''')
gluc = st.radio("Glucose Amount", ('Normal', 'Above Normal', 'Well Above Normal'))
st.write(''' ''')
sm = st.radio("Do You Smoke?", ('Yes', 'No'))
st.write(''' ''')
alco = st.radio("Do You Drink?", ('Yes', 'No'))
st.write(''' ''')
act = st.radio("Do you Exersice?", ('Yes', 'No'))
st.write(''' ''')

#Age conversion
age=ag*365

# gender conversion
if gen == "Male":
    gender = 2
else:
    gender = 1

# cholesterol conversion
if chol == "Normal":
    cholesterol = 1
elif chol == "Above Normal":
    cholesterol = 2
else:
    cholesterol = 3

# cholesterol conversion
if gluc == "Normal":
    glucose = 1
elif gluc == "Above Normal":
    glucose = 2
else:
    glucose = 3

# Smoke conversion
if sm == "Yes":
    smoke = 1
else:
    smoke = 0

# Alcohol conversion
if alco == "Yes":
    alcohol = 1
else:
    alcohol = 0

# Active conversion
if act == "Yes":
    active = 1
else:
    active = 0

# Convert height to meters
height_m = height / 100

# Calculate BMI
bmi = weight / (height_m ** 2)

user_input = np.array([age, gender, height, weight, ap_hi,
                       ap_lo, cholesterol, glucose, smoke, alcohol,
                       active, bmi]).reshape(1, -1)


# Function to display instructions for users with high chance of cardiovascular disease
def display_high_chance_instructions():
    st.title("You Have a Very High Chance of Having a Cardiovascular disease.")
    st.write("We recommend you follow the following Instructions")
    st.write(
        "1. Consult a healthcare professional: It is important to consult with a healthcare professional for a thorough evaluation and diagnosis.")
    st.write(
        "2. Follow recommended lifestyle changes: Implement positive lifestyle changes such as adopting a heart-healthy diet, engaging in regular physical activity, quitting smoking, managing stress levels, and maintaining a healthy weight.")
    st.write(
        "3. Take prescribed medications: Follow your healthcare professional's instructions and take prescribed medications as directed.")
    st.write(
        "4. Monitor and track your health: Keep track of your blood pressure, heart rate, and any symptoms you experience. Regularly check in with your healthcare professional.")


# Function to display instructions for users without cardiovascular disease
def display_low_chance_instructions():
    st.title("You Have a Very Low Chance of Having a Cardiovascular disease.")
    st.write("We recommend you follow the following Instructions")

    st.write(
        "1. Maintain a healthy lifestyle: Focus on balanced nutrition, regular physical activity, stress management, and maintaining a healthy weight.")
    st.write(
        "2. Regular check-ups: Schedule regular check-ups with your healthcare professional to monitor your overall health.")
    st.write("3. Stay informed: Educate yourself about cardiovascular health and risk factors.")
    st.write(
        "4. Support others: Encourage your loved ones to prioritize their cardiovascular health and promote healthy habits.")


# import dataset
def get_dataset():
    data = pd.read_csv('Cardio_train.csv', delimiter=';')

    # Calculate the correlation matrix
    #corr_matrix = data.corr()

    # Create a heatmap of the correlation matrix
    #plt.figure(figsize=(10, 8))
    #sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    #plt.title('Correlation Matrix')
    #plt.xticks(rotation=45)
    #plt.yticks(rotation=0)
    #plt.tight_layout()

    # Display the heatmap in Streamlit
    #st.pyplot()

    return data

if st.button('Submit'):
    data = get_dataset()

    # fix column names
    data.columns = (["id", "age", "gender", "height", "weight", "ap_hi",
                     "ap_lo", "cholesterol", "gluc", "smoke",
                     "alco", "active", "cardio"])

    data.drop("id", axis=1, inplace=True)
    data.drop_duplicates(inplace=True)
    data["bmi"] = data["weight"] / (data["height"] / 100) ** 2
    out_filter = ((data["ap_hi"] > 250) | (data["ap_lo"] > 200))
    data = data[~out_filter]


    # Data Split
    y = data['cardio']
    X = data.drop(['cardio'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1337)


    # Preprocessing and Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    user_input_scaled = scaler.transform(user_input)

    # Logistic Regression
    logreg = LogisticRegression()
    logreg.fit(X_train_scaled, y_train)
    logreg_pred = logreg.predict(user_input_scaled)
    logreg_score = accuracy_score(y_test, logreg.predict(X_test_scaled))
    #st.write('The chances of having Cardio Vascular Disease (LogisticRegression): ', logreg_pred * 100)
    #st.write('Logistic Regression Accuracy:', logreg_score)

    # Gradient Boosting Classifier
    gbclf = GradientBoostingClassifier()
    gbclf.fit(X_train_scaled, y_train)
    gbclf_pred = gbclf.predict(user_input_scaled)
    gbclf_score = accuracy_score(y_test, gbclf.predict(X_test_scaled))
    #st.write('The chances of having Cardio Vascular Disease (GradientBoostingClassifier): ', gbclf_pred * 100)
    #st.write('Gradient Boosting Classifier Accuracy:', gbclf_score)

    # RandomForestClassifier
    rfclf = RandomForestClassifier(random_state=1337)
    rfclf.fit(X_train_scaled, y_train)
    rfclf_pred = rfclf.predict(user_input_scaled)
    rfclf_score = accuracy_score(y_test, rfclf.predict(X_test_scaled))
    #st.write('The chances of having Cardio Vascular Disease (RandomForestClassifier): ', rfclf_pred * 100)
    #st.write('RandomForestClassifier Accuracy:', rfclf_score)

    # Define the ensemble classifier
    ensemble = VotingClassifier(estimators=[('lr', logreg), ('gb', gbclf), ('rf', rfclf)], voting='hard')

    # Train the ensemble classifier
    ensemble.fit(X_train_scaled, y_train)

    # Predict on the user input
    ensemble_pred = ensemble.predict(user_input_scaled)
    prediction = ensemble_pred * 100

    if prediction == 100:
        display_high_chance_instructions()
    else:
        display_low_chance_instructions()

    # Evaluate the ensemble classifier
    ensemble_score = accuracy_score(y_test, ensemble.predict(X_test_scaled))
    st.write('Ensemble Model Accuracy:', ensemble_score)

    # Classification Matrix
    # ensemble_report = classification_report(y_test, ensemble.predict(X_test_scaled),output_dict=True)
    # st.write('Ensemble Model Classification Report:')
    # st.write(''' ''')
    # st.write(ensemble_report)