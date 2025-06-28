#  Model Pipelines and Deployment Basics

### Topic: Building End-to-End Pipelines

---

##  Summary

* Use **`Pipeline`** and **`ColumnTransformer`** from `sklearn` to streamline ML workflows
* Chain together **preprocessing** and **model training** steps
* Learn to **save** and **load models** with `joblib` or `pickle`
* Understand basics of **deployment** using **Streamlit** or **Flask APIs**

---

## 1. Why Use Pipelines?

In machine learning, it's easy to create a mess:

* You scale features manually
* Encode categories in a separate step
* Fit a model in yet another cell

This approach **breaks easily**, especially when:

* You switch datasets
* You retrain with new data
* You want to deploy the model

---

###  Analogy: Assembly Line in a Factory

> Imagine building a car.
> You wouldn’t ask a worker to install the engine, then send the car across the street for painting, and back again for wheels.
> Instead, you build a **production line** — a fixed sequence of steps.

> Pipelines are your **production line** for machine learning.

---

## 2. The `Pipeline` Class – Clean Workflow

A **Pipeline** allows you to chain **preprocessing + modeling** into a single object.

###  Example: Scaling + Logistic Regression

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Define pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression())
])

# Fit pipeline
pipe.fit(X_train, y_train)

# Predict
y_pred = pipe.predict(X_test)
```

---

###  Why Use a Pipeline?

* Keeps your code **modular and clean**
* Avoids **data leakage** (fit only on training data)
* Easy to **save**, **reuse**, and **deploy**

---

## 3. ColumnTransformer – Handling Mixed Data Types

Most real-world data has a mix of:

* **Numerical** features (e.g., age, income)
* **Categorical** features (e.g., gender, city)

You want to apply:

* **Scaling** to numerical columns
* **Encoding** to categorical columns

---

###  Analogy: Different Washing Machines for Clothes

> You don’t wash jeans and silk in the same way.
> ColumnTransformer lets you **process each type of feature** using a **different machine** — but all in **one laundry room**.

---

###  Code Example: ColumnTransformer + Pipeline

```python
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Sample data
df = pd.DataFrame({
    'age': [25, 32, 47, 51],
    'income': [40000, 60000, 80000, 100000],
    'gender': ['Male', 'Female', 'Female', 'Male'],
    'purchased': [0, 1, 1, 0]
})

X = df.drop('purchased', axis=1)
y = df['purchased']

# Columns
numeric_features = ['age', 'income']
categorical_features = ['gender']

# Preprocessing
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(), categorical_features)
])

# Final pipeline
pipe = Pipeline([
    ('prep', preprocessor),
    ('clf', LogisticRegression())
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Fit pipeline
pipe.fit(X_train, y_train)

# Predict
y_pred = pipe.predict(X_test)
```

---

## 4. Saving and Loading Models (joblib / pickle)

When your model is trained and ready, you’ll want to **save it** and **reload** it later (for use in APIs or batch predictions).

---

###  Analogy: Freezing Food

> After cooking (training), you can’t repeat everything from scratch every time.
> You **freeze** your model and reheat it when needed.

---

###  Code: Save with `joblib`

```python
import joblib

# Save the pipeline
joblib.dump(pipe, 'model_pipeline.pkl')

# Load the pipeline
model_loaded = joblib.load('model_pipeline.pkl')

# Use it
model_loaded.predict(X_test)
```

---

###  `joblib` vs `pickle`

| Tool     | Use Case                                 | Notes                                    |
| -------- | ---------------------------------------- | ---------------------------------------- |
| `pickle` | General-purpose object serialization     | Works for all Python objects             |
| `joblib` | Specifically for ML models, numpy arrays | Faster and more efficient for large data |

---

## 5. Intro to Deployment

Once your model is saved, the next step is **making it accessible** — turning it into an API or web app.

---

###  Deployment Options

| Tool          | Description                         | Use Case                     |
| ------------- | ----------------------------------- | ---------------------------- |
| **Flask**     | Lightweight Python web server       | REST API for backends        |
| **FastAPI**   | Modern async alternative to Flask   | Fast, production-ready APIs  |
| **Streamlit** | Quick dashboards for ML apps        | Prototyping & UI demos       |
| **Gradio**    | Simple web UI with inputs & outputs | Ideal for public model demos |

---

###  Analogy: Opening a Restaurant

> You cooked a great dish (trained model), froze it (saved it), now you want to **serve it** to customers through a **window (API)** or in a **restaurant (UI app)**.

---

###  Code: Minimal Flask App to Serve a Model

```python
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('model_pipeline.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
```

---

###  Code: Streamlit App

```python
import streamlit as st
import joblib
import pandas as pd

model = joblib.load('model_pipeline.pkl')

st.title("Purchase Prediction App")

age = st.slider("Age", 18, 70)
income = st.number_input("Income", 10000, 200000)
gender = st.selectbox("Gender", ["Male", "Female"])

input_df = pd.DataFrame({
    'age': [age],
    'income': [income],
    'gender': [gender]
})

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    st.success(f"Prediction: {'Purchased' if prediction == 1 else 'Not Purchased'}")
```

---

## 6. Summary Table

| Task                    | Tool                      | Purpose                                       |
| ----------------------- | ------------------------- | --------------------------------------------- |
| Build Clean ML Workflow | `Pipeline`                | Chain preprocessing + model                   |
| Handle Mixed Types      | `ColumnTransformer`       | Different preprocessing for different columns |
| Save Models             | `joblib`, `pickle`        | Reuse trained models                          |
| Serve Models            | Flask, Streamlit, FastAPI | Deploy as API or app                          |

---

## 7. Final Analogy Recap

| Analogy                    | Concept                      |
| -------------------------- | ---------------------------- |
| Assembly Line              | Pipeline                     |
| Different Washing Machines | ColumnTransformer            |
| Freezing Food              | Saving trained models        |
| Restaurant or Takeaway     | Deployment through UI or API |

---

