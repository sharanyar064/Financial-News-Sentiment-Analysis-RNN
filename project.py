import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow as tf  # For GPU/TPU optimization if available

# ==== Paths ====
DATA_PATH = "data/financial_sentiment.csv"   # your CSV file
RESULTS_DIR = "results"
MODEL_DIR = "saved_model"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ==== Load dataset safely ====
try:
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv(DATA_PATH, encoding="latin1")

# ==== Column cleanup ====
if len(df.columns) == 1:
    df = pd.read_csv(DATA_PATH, encoding="utf-8", header=None)
    df.columns = ["sentiment", "sentence"]
elif len(df.columns) >= 2:
    df = df.rename(columns={
        "text": "sentence",
        "Sentence": "sentence",
        "clean_text": "sentence",
        "message": "sentence",
        "label": "sentiment",
        "Sentiment": "sentiment",
        "category": "sentiment"
    })
    if "sentence" not in df.columns or "sentiment" not in df.columns:
        df = df.iloc[:, :2]
        df.columns = ["sentiment", "sentence"]

# ==== Data Cleaning Improvements ====
# Lowercase sentiments for consistency (in case of mixed case in Kaggle data)
df["sentiment"] = df["sentiment"].str.lower().str.strip()
# Remove any invalid labels (if any)
valid_labels = ['positive', 'negative', 'neutral']
df = df[df["sentiment"].isin(valid_labels)]
# Basic text cleaning: remove extra quotes, etc.
df["sentence"] = df["sentence"].str.replace(r'^"|"$', '', regex=True).str.strip()

# ==== Print Class Distribution (to understand imbalance) ====
print("\nClass Distribution:")
print(df["sentiment"].value_counts(normalize=True))

# ==== Encode labels (alphabetical: negative=0, neutral=1, positive=2) ====
encoder = LabelEncoder()
df["label"] = encoder.fit_transform(df["sentiment"])
print("\nLabel Mapping:", dict(zip(encoder.classes_, encoder.transform(encoder.classes_))))

# ==== Train/Test Split (stratified for balance) ====
X_train, X_test, y_train, y_test = train_test_split(
    df["sentence"], df["label"],
    test_size=0.2, random_state=42, stratify=df["label"]
)

# ==== Tokenization (increased max words for financial vocab) ====
MAX_WORDS = 15000  # Increased for better coverage of financial terms
MAX_SEQUENCE_LENGTH = 200  # Increased for longer financial sentences
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

# ==== Compute class weights (balanced, NO arbitrary boost to positive) ====
weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = {i: float(weights[i]) for i in range(len(weights))}
print("\nClass weights applied (balanced, no boost):", class_weights)

# ==== Model (added more regularization, reduced complexity to prevent overfitting) ====
model = Sequential([
    Embedding(MAX_WORDS, 128, input_length=MAX_SEQUENCE_LENGTH),
    Bidirectional(LSTM(64, return_sequences=True, dropout=0.4, recurrent_dropout=0.3)),  # Reduced units, increased dropout
    Bidirectional(LSTM(64, dropout=0.4, recurrent_dropout=0.3)),
    Dense(32, activation="relu"),  # Reduced dense units
    Dropout(0.5),  # Increased dropout
    Dense(len(encoder.classes_), activation="softmax")
])

# Explicitly build model
model.build(input_shape=(None, MAX_SEQUENCE_LENGTH))
model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(0.001), metrics=["accuracy"])  # Increased LR to 0.001 for better convergence
model.summary()

# ==== Training (increased epochs, added more patience) ====
es = EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True, min_delta=0.001)  # More patience, min_delta for better stopping
history = model.fit(
    X_train_seq, y_train,
    validation_split=0.2,
    epochs=50,  # Increased max epochs
    batch_size=64,  # Increased batch size for stability
    class_weight=class_weights,
    callbacks=[es],
    verbose=1
)

# ==== Evaluation ====
loss, acc = model.evaluate(X_test_seq, y_test, verbose=0)
print(f"\n✅ Test Accuracy: {acc*100:.2f}%")

y_pred = np.argmax(model.predict(X_test_seq), axis=1)
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

# ==== Graphs ====
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.title("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.title("Loss")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "training_history.png"))
plt.show()

# ==== Save model ====
model.save(os.path.join(MODEL_DIR, "financial_rnn_model.keras"))

# ==== Prediction Examples (corrected to handle financial context better) ====
sample_texts = [
    "Stock prices surged after the company announced excellent earnings results",  # Should be positive
    "The market remained stable with no major changes",  # Should be neutral
    "The company reported massive losses and stock prices plummeted"  # Should be negative (added for testing)
]
seqs = pad_sequences(tokenizer.texts_to_sequences(sample_texts), maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
preds = np.argmax(model.predict(seqs), axis=1)
pred_probs = model.predict(seqs)

for text, label, probs in zip(sample_texts, preds, pred_probs):
    print(f"\nText: {text}")
    print("Predicted Sentiment:", encoder.classes_[label])
    print("Confidence Scores:", {cls: f"{prob:.2%}" for cls, prob in zip(encoder.classes_, probs)})