#Introduction
This project is a simple machine learning model that can tell whether an image contains a cat or not a cat.

It’s built completely from scratch in Python using NumPy (no TensorFlow, PyTorch, or scikit-learn for the model.
The goal is to demonstrate the math and logic behind logistic regression and apply it to image classification.

#Project Structure
cat-or-not/
│── app.py             # Streamlit UI to test images
│── train.py           # Train the model on dataset
│── models.py          # Logistic regression implementation
│── utils.py           # Helper functions (load dataset, preprocess)
│── data/              # Place dataset files here
│    ├── train_catvnoncat.h5
│    └── test_catvnoncat.h5
│── venv/              # Virtual environment (ignored in Git)
│── cost_curve.png     # Cost curve after training
│── model_params.npz   # Saved trained weights
│── README.md          # Project documentation

#Setup:
  - step: Clone the project
    command: |
      git clone https://github.com/your-username/cat-or-not.git
      cd cat-or-not

  - step: Create virtual environment
    command: |
      python -m venv venv

  - step: Install dependencies
    command: |
      pip install -r requirements.txt

  - step: Train the model
    command: |
      python train.py
    result: |
      - Trains logistic regression model
      - Saves weights to model_params.npz
      - Generates cost_curve.png

  - step: Run the Streamlit app
    command: |
      streamlit run app.py
    result: |
      Open the provided local URL in browser.
      Upload any image → get "Cat" or "Not a Cat" prediction.
