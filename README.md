# Introduction

This project is a simple machine learning model that can tell whether an image contains a cat or not a cat.

#steps:
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
