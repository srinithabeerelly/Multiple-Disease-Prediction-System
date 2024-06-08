
# 📝 Multiple-Disease-Prediction-webpage

🚨 Note: This code is provided as a prototype and may require further enhancements and optimizations for real-world scenarios.

---

## 📚 Description

The Multiple-Disease-Prediction-webpage is a web application developed using Python and the Streamlit library. The application is designed to predict the likelihood of three different diseases (diabetes, heart disease, and Parkinson's disease) using machine learning models.

### 📁 Repository Structure

- `main.py`: The main script for running the web application.
- `pred_systems/`: Directory containing scripts for the prediction systems.
- `saved_models/`: Directory containing the trained machine learning models.

## 🚀 Installation

Clone the repository to your local machine:

The application is developed in Python, and the following packages are required:

- Streamlit
- Pickle

## 🔧 Usage

To run the web application, navigate to the directory containing the cloned repository and run the following command in the terminal:

```bash
streamlit run main.py
```

This will start the Streamlit server, and the web application will be accessible at `localhost:8501`.

## 💡 Features

The application offers predictions for three different diseases:

### 1. Diabetes Prediction 🩺

The diabetes prediction page allows users to input various health parameters including:
- Number of Pregnancies
- Glucose Level
- Blood Pressure value
- Skin Thickness value
- Insulin Level
- BMI value
- Diabetes Pedigree Function value
- Age of the Person

Users can click the "Diabetes Test Result" button to get a prediction on whether they are diabetic or not. ✅❌

### 2. Heart Disease Prediction ❤️

The heart disease prediction page prompts users for the following health parameters:
- Age
- Sex
- Chest Pain types
- Resting Blood Pressure
- Serum Cholestoral in mg/dl
- Fasting Blood Sugar > 120 mg/dl
- Resting Electrocardiographic results
- Maximum Heart Rate achieved
- Exercise Induced Angina
- ST depression induced by exercise
- Slope of the peak exercise ST segment
- Major vessels colored by flourosopy
- Thal: 0 = normal; 1 = fixed defect; 2 = reversible defect

Users can press the "Heart Disease Test Result" button to get a prediction on whether they have heart disease or not. ✅❌

### 3. Parkinson's Disease Prediction 🎙️

The Parkinson's disease prediction page allows users to input various health parameters:
- MDVP:Fo(Hz)
- MDVP:Fhi(Hz)
- MDVP:Flo(Hz)
- MDVP:Jitter(%)
- MDVP:Jitter(Abs)
- MDVP:RAP
- MDVP:PPQ
- Jitter:DDP
- MDVP:Shimmer
- MDVP:Shimmer(dB)
- Shimmer:APQ3
- Shimmer:APQ5
- MDVP:APQ
- Shimmer:DDA
- NHR
- HNR
- RPDE
- DFA
- spread1
- spread2
- D2
- PPE

Users can press the "Parkinson's Test Result" button to get a prediction on whether they have Parkinson's disease or not. ✅❌

## 📄 License

This project is open source and available under the MIT License.

## 📧 Contact

For any inquiries, please reach out to the repository owner at

 [github.com/Cathorus](https://github.com/Cathorus).
