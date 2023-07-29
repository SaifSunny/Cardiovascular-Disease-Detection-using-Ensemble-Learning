# Cardiovascular Disease Detection using Ensemble Learning
This repository contains a machine learning software that predicts the likelihood of an individual having cardiovascular disease based on input attributes. The software uses an ensemble learning approach that combines multiple machine learning models to achieve better predictive performance. The models included in the ensemble are Logistic Regression, Gradient Boosting Classifier, and Random Forest Classifier.

**Live Demo:** Cardiovascular Disease Prediction Web Application

**GitHub Repository:** https://github.com/SaifSunny/Cardiovascular-Disease-Detection-using-Ensemble-Learning

# Input Attributes
The software takes the following input attributes to make predictions:
<ul>
<li>Age (Years): The age of the individual in years.</li>
<li>Gender: The gender of the individual (Male or Female).</li>
<li>Height (cm): The height of the individual in centimeters.</li>
<li>Weight (Kg.): The weight of the individual in kilograms.</li>
<li>Systolic Blood Pressure: The systolic blood pressure value of the individual.</li>
<li>Diastolic Blood Pressure: The diastolic blood pressure value of the individual.</li>
<li>Cholesterol Amount: The cholesterol level of the individual (Normal, Above Normal, or Well Above Normal).</li>
<li>Glucose Amount: The glucose level of the individual (Normal, Above Normal, or Well Above Normal).</li>
<li>Do You Smoke?: Whether the individual smokes or not (Yes or No).</li>
<li>Do You Drink?: Whether the individual drinks alcohol or not (Yes or No).</li>
<li>Do You Exercise?: Whether the individual exercises or not (Yes or No).</li>
</ul>
# Getting Started
To use the Cardiovascular Disease Prediction software, follow these steps:

  Clone the GitHub repository: ```git clone https://github.com/SaifSunny/Cardiovascular-Disease-Detection-using-Ensemble-Learning.git.```
  
  Install the required dependencies by running pip install -r requirements.txt in your Python environment.
  
  Run the Streamlit web application by executing streamlit run app.py.
  
  Access the live demo at https://cardiovascular-disease.streamlit.app/.
  
  Fill in the input attributes and click the "Submit" button to get the prediction result.
  
# Model Accuracy
The ensemble model used in this software has an accuracy of approximately 74% on the test dataset.

# Instructions for High Chance of Cardiovascular Disease
If the prediction result indicates a high chance of having cardiovascular disease, the software provides the following instructions:

**Consult a healthcare professional:** It is essential to consult with a healthcare professional for a thorough evaluation and diagnosis.

**Follow recommended lifestyle changes:** Implement positive lifestyle changes such as adopting a heart-healthy diet, engaging in regular physical activity, quitting smoking, managing stress levels, and maintaining a healthy weight.

**Take prescribed medications:** Follow your healthcare professional's instructions and take prescribed medications as directed.

**Monitor and track your health:** Keep track of your blood pressure, heart rate, and any symptoms you experience. Regularly check in with your healthcare professional.

#Instructions for Low Chance of Cardiovascular Disease
If the prediction result indicates a low chance of having cardiovascular disease, the software provides the following instructions:

**Maintain a healthy lifestyle:** Focus on balanced nutrition, regular physical activity, stress management, and maintaining a healthy weight.

**Regular check-ups:** Schedule regular check-ups with your healthcare professional to monitor your overall health.

**Stay informed:** Educate yourself about cardiovascular health and risk factors.

**Support others:** Encourage your loved ones to prioritize their cardiovascular health and promote healthy habits.

# Dataset
The software uses the Cardiovascular Disease dataset, which can be found in the file cardio_train.csv. The dataset is preprocessed and used to train the machine learning models.

# License
This project is licensed under the MIT License.

# Acknowledgments
The development of this software was inspired by the need to create a user-friendly tool for predicting cardiovascular disease risk. Special thanks to the open-source community for providing the necessary libraries and tools to develop this application.

Please feel free to contribute to this project by opening issues or submitting pull requests on GitHub.

For any questions or inquiries, please contact the project maintainer: saifsunny56@gmail.com.

Thank you for using our Cardiovascular Disease Prediction software!
