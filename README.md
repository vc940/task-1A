# Blood Donation Prediction

This project predicts whether a donor will donate blood in the future using a machine learning model trained on structured donor data (RFM and Time attributes).

---

## Approach

The goal of this task is to build a binary classification model that predicts whether a blood donor will donate again. The dataset includes:

- **Recency**: Months since the last donation  
- **Frequency**: Total number of donations  
- **Monetary**: Total blood donated (in c.c.)  
- **Time**: Time (in months) between the first donation and the last

Steps followed:

1. **Data Preprocessing**:
   - Loaded data from `blood_donation.csv`.
   - Renamed and selected relevant columns: `recency`, `frequency`, `monetary`, `time`, and `target`.
   - Applied `MinMaxScaler` to normalize input features.

2. **Model Building**:
   - Split data into training and testing sets (80/20).
   - Trained a **Logistic Regression** classifier using the preprocessed data.

3. **Evaluation**:
   - Evaluated model performance using accuracy, classification report, and confusion matrix.
   - Stored the trained model as `rfm_model.pkl`.

4. **Serving the Model**:
   - Built a lightweight **Flask API** (`app.py`) that:
     - Loads the saved model and scaler
     - Accepts JSON input with `recency`, `frequency`, `monetary`, and `time`
     - Returns a binary prediction (`0` or `1`)

---

## Models / Libraries Used

- **Scikit-learn (`sklearn`)**:
  - `LogisticRegression` (classifier)
  - `train_test_split`, `MinMaxScaler`, `classification_report`, etc.

- **Flask**: For creating REST API to serve predictions

- **pandas / numpy**: Data handling and preprocessing

- **pickle**: Model and scaler serialization

---

## How to Build and Run the Solution

> This section is for documentation purposes only. The actual execution must follow the **“Expected Execution”** instructions from the task.

### 1. Clone the repository

```bash
git clone https://github.com/vc940/task-1A
cd task-1A
```
### 2. Set up a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install pandas numpy scikit-learn flask
```
### 4. Train the model (if needed)

```bash
python model.py
```

### 5. Run the Flask server

```bash
python app.py
```
The API will start at: `http://127.0.0.1:5000/predict`

### 6. Send a POST request for prediction
Example using `curl`:

```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"recency": 2, "frequency": 50, "monetary": 12500, "time": 98}'
```

Response:
```json
{"prediction": 1}
```

## Expected Execution

- `python model.py`: Trains and saves the model and scaler  
- `python app.py`: Launches the Flask API  
- Use `POST /predict` endpoint to get predictions with input data

---

## Note

- Input JSON must include the fields: `recency`, `frequency`, `monetary`, and `time`
- Output will be a binary prediction:
  - `0` → Unlikely to donate again  
  - `1` → Likely to donate again
