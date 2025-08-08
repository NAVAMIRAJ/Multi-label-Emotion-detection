# Multi-label Emotion Classification

This project performs multi-label classification of text into five emotion categories using a fine-tuned RoBERTa model:
**anger**, **fear**, **joy**, **sadness**, and **surprise**.

## Contents

- `ReadMe.md` — Instructions
- `NLP_milestone3_model_training.ipynb` — Training notebook
- `main.py` — Prediction script

## Setup Instructions

Install the required dependencies:

```bash
pip install torch transformers scikit-learn pandas numpy tqdm
```

## How to Run Predictions

Run the prediction script with a `.csv` file that contains a `text` column:

```bash
python main.py path_to_csv_file.csv
```

### Notes:
- Download the trained model weights from:  
  [bert_emotion_best.pt](https://drive.google.com/file/d/1JCmoug1TNjgUn9sWY_wYYPSbeMF6uxlH/view?usp=sharing)
- Place `bert_emotion_best.pt` in the **same directory** as `main.py`.
- The output will be:
  - Printed to the terminal
  - Saved to `test_result.csv` in the same directory

## Output Format

The output CSV will include:
- The original text
- One column per emotion: `anger`, `fear`, `joy`, `sadness`, `surprise`  
  Each with a binary value (1 = predicted, 0 = not predicted)