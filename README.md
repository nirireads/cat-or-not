# 🐱 Cat or Not?  

A simple machine learning project that predicts whether an image contains a **cat** 🐾 or **not a cat** 🚫.  
It uses a **logistic regression model**, trained from scratch, and provides a **Streamlit web app** for easy image testing.  

---
````bash
## 📦 Setup Instructions  
- Clone the Repository
git clone https://github.com/your-username/cat-or-not.git
cd cat-or-not

- Create & Activate Virtual Environment
python -m venv venv
On Windows : venv\Scripts\activate
On macOS/Linux : source venv/bin/activate

- Install Dependencies
pip install -r requirements.txt

- Training the Model
python train.py

# Running the Program
streamlit run app.py

````Project Structure
cat-or-not/
├── app.py              # Streamlit web app: loads model, handles uploads, displays prediction.
├── train.py            # Core ML script: defines the Logistic Regression model, training loop, and optimization.
├── model_params.npz    # Saved model weights (w and b) after training.
├── cost_curve.png      # Visualization of the cost function vs. iterations (proof of learning).
├── requirements.txt    # Project dependencies.
└── README.md           # This documentation file.

