# Context-Aware Neural Spellchecker (Norvig-LSTM Hybrid)

A custom-built natural language processing pipeline that corrects spelling mistakes by combining exhaustive edit-distance search with a PyTorch Long Short-Term Memory (LSTM) neural network for grammatical context evaluation.

## The Architecture & Algorithm Pipeline

Traditional spellcheckers (like Peter Norvig's classic algorithm) rely entirely on Edit Distance (Damerau-Levenshtein). While effective at finding visually similar words, they fail when multiple words have the same edit distance (e.g., correcting "ta" to "to" vs "at"). 

This project solves that limitation using a Two-Round Hybrid approach, supported by a custom end-to-end data processing pipeline:

### 1. Data Preparation & Tokenization
* **Corpus Processing:** The model ingests a cleaned JSON corpus (extracted from *Alice in Wonderland*), stripping special characters while preserving internal syntax (like apostrophes).
* **Vocabulary Limitation:** To optimize the neural network's learning efficiency, the vocabulary is restricted to the top 3,000 most frequent words. 
* **Token Mapping:** Rare words are replaced with an `<UNK>` (Unknown) token, and sequences are padded with a `<PAD>` token. A bidirectional dictionary (`word2idx` and `idx2word`) maps the text into numeric tensors for the PyTorch engine.

### 2. Dynamic Typo Generation (Data Augmentation)
Instead of relying on a static dataset of misspellings, the pipeline dynamically generates synthetic typos during training. A random 5-word context window is selected, and the 6th target word is mutated using one of four randomized algorithms:
* **Deletion:** Removing a random character (e.g., `apple` → `aple`)
* **Insertion:** Injecting a random alphabet character (e.g., `apple` → `axpple`)
* **Substitution:** Replacing a character randomly (e.g., `apple` → `bpple`)
* **Transposition (Mix):** Swapping adjacent characters (e.g., `apple` → `alppe`)

### 3. The PyTorch Neural Network (Context Engine)
The grammatical context is modeled using a Left-to-Right sequence model built in PyTorch, trained over 400 epochs using the Adam Optimizer and Cross-Entropy Loss.
* **Embedding Layer (`nn.Embedding`):** Maps the sparse 3,000-dimensional vocabulary IDs into dense 64-dimensional vectors, capturing semantic relationships between words.
* **Memory Layer (`nn.LSTM`):** A 128-unit Long Short-Term Memory network processes the 5 preceding context words. It maintains a hidden state that builds a grammatical understanding of the sentence up to the point of the typo.
* **Classification Layer (`nn.Linear`):** The final hidden state of the LSTM is sliced (`lstm_out[:, -1, :]`) and passed through a linear layer to output unnormalized mathematical confidence scores (logits) across the entire 3,000-word vocabulary.

### 4. Hybrid Inference Engine (The Two-Round Resolution)
When predicting a correction during testing, the model executes its signature hybrid logic:
* **Round 1 - The Typo Brain (Damerau-Levenshtein Search):** The misspelled word is compared against all 3,000 dictionary words using NLTK's edit distance algorithm (with transpositions enabled). The algorithm isolates a candidate list of the word(s) with the absolute lowest mathematical edit distance.
* **Round 2 - The Grammar Brain (LSTM Tie-Breaker):** If Round 1 yields a single match, it is returned immediately. If it yields multiple valid candidates (a tie), the candidate list is passed to the custom-trained PyTorch LSTM. The LSTM evaluates the preceding 5-word context, extracts the raw output logits specifically for the tied candidates, and selects the word with the highest grammatical probability.

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
