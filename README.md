# 🐱 Cat or Not?  

A simple machine learning project that predicts whether an image contains a **cat** 🐾 or **not a cat** 🚫.  
It uses a **logistic regression model**, trained from scratch, and provides a **Streamlit web app** for easy image testing.  

---

## 📦 Setup Instructions  

# 1️⃣ Clone the Repository
git clone https://github.com/your-username/cat-or-not.git
cd cat-or-not

# 2️⃣ Create & Activate Virtual Environment
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# 3️⃣ Install Dependencies
pip install -r requirements.txt

# Training the Model
python train.py

# Running the Program
streamlit run app.py


#Project Structure
cat-or-not/
├── app.py              # Streamlit web app
├── train.py            # Model training script
├── model_params.npz    # Saved model weights
├── cost_curve.png      # Training visualization
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation

