# Context-Aware Neural Spellchecker (Norvig-LSTM Hybrid)

A custom-built natural language processing pipeline that corrects spelling mistakes by combining exhaustive edit-distance search with a PyTorch Long Short-Term Memory (LSTM) neural network for grammatical context evaluation.

## The Architecture

Traditional spellcheckers (like Peter Norvig's classic algorithm) rely entirely on Edit Distance (Damerau-Levenshtein). While effective at finding visually similar words, they fail when multiple words have the same edit distance (e.g., correcting "ta" to "to" vs "at"). 

This project solves that limitation using a Two-Round Hybrid approach:

*   **Round 1: The Typo Brain (Exhaustive Search):** The algorithm scans the entire 3,000-word dictionary and calculates the edit distance for every word. It returns a candidate list of the absolute mathematically closest words.
*   **Round 2: The Grammar Brain (Deep Learning):** If Round 1 returns a tie (multiple words with the same edit distance), the candidate list is passed to a custom-trained PyTorch LSTM. The LSTM reads the preceding context of the sentence and scores each candidate based on grammatical probability, effectively breaking the tie with human-like contextual awareness.

## How to Run

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```
**2. Generate the Dataset:**
Run the data preparation script to download Alice in Wonderland via NLTK, clean the punctuation (preserving internal apostrophes), and generate the JSON corpus.
```bash
python data_prep.py
```
**3. Train and Test the Model:**
Run the main script to train the LSTM on 15,000 contextual phrases and run the 100-sentence inference test.
```bash
python train_and_test.py
```
