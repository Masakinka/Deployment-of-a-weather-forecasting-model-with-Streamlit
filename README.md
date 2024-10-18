# Deployment of a Weather Forecasting Model with Streamlit

# Weather Forecasting Model with Streamlit 🌦️

This project demonstrates the deployment of a **Machine Learning** model to predict whether it will rain tomorrow using weather data. The model has been trained on an Australian weather dataset and the web app is built with **Streamlit** to allow real-time predictions based on user input.

---

## 📂 Project Structure:
- **train_model.ipynb**: Jupyter notebook used for data analysis, preprocessing, and training the model.
- **app.py**: The main file for the Streamlit app, which loads the trained model and provides an interactive interface for weather prediction.
- **models/**: Directory containing the pre-trained machine learning model.
- **data/**: Directory containing the dataset used for training.
- **image/**: Includes media files such as the logo used in the Streamlit app.

---

## 🔧 Tools & Technologies:
- **Python**: Core language used in the project.
- **Scikit-learn**: Used for building and training the machine learning model.
- **Pandas**: For data manipulation and preprocessing.
- **Joblib**: For model saving and loading.
- **Streamlit**: For building an interactive web interface.

---

## 🚀 App Overview:
The app allows users to input various weather conditions such as temperature, rainfall, sunshine, and cloudiness to predict whether it will rain tomorrow. The model outputs:
- **Rain Prediction** (Yes/No)
- **Prediction Probability**: A confidence score indicating the likelihood of rain.

You can try the app [here](https://masakinka-deployment-of-a-weather-forecasting-model--app-cgf1jm.streamlit.app/)!

---

## 📊 Dataset:
The model is trained on the **Australian weather dataset** from Kaggle. It includes multiple weather features collected across Australia, such as temperature, rainfall, and humidity.

[Kaggle Dataset Link](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)

---

## 🌟 Highlights:
- **Interactive predictions**: The app allows real-time predictions by adjusting weather parameters.
- **Elegant UI**: Built with **Streamlit** for a clean and simple user experience.
- **Deployed**: The model is deployed online and accessible via Streamlit.

---

Feel free to fork this repository, try out the app, and improve the model for even more accurate predictions! 😎
