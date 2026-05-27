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
When predicting a correction during testing, the model executes its signature hybrid logic inside the `final()` function. This guarantees that the output is both visually accurate to the original typo and grammatically logical within the sentence context:

* **Round 1 - The Typo Brain (Damerau-Levenshtein Candidate Generation):** When an out-of-vocabulary word (typo) is detected, the engine bypasses traditional heuristic guessing and performs a brute-force exhaustive search across the entire 3,000-word vocabulary matrix. 
    * It calculates the Damerau-Levenshtein distance for every single word. (By enabling `transpositions=True`, the algorithm intelligently understands that adjacent swapped letters—like "hte" instead of "the"—count as a single human error rather than two distinct edits).
    * The engine dynamically filters this matrix, tracking the absolute `min_distance`. Any word that perfectly matches this minimum distance is appended to a highly restricted `candidate_words` array.

* **Round 2 - The Grammar Brain (Logit Extraction & Tie-Breaking):** If Round 1 yields a single match, it is returned immediately. However, if the `candidate_words` array contains a tie (e.g., "to" and "at" are both exactly 1 edit away from the typo "ta"), the engine triggers the neural network.
    * To conserve memory, the model is placed in `.eval()` mode with `torch.no_grad()` active.
    * The LSTM performs a forward pass on the preceding 5-word sequence (the `input_x` context window).
    * The model outputs a tensor containing 3,000 raw mathematical confidence scores (logits). The engine ignores the rest of the dictionary and extracts the specific logits *only* for the tied candidate words.
    * By iterating through the candidates and comparing these localized logits, the AI selects the word with the highest maximum grammatical probability (`best_score`). This allows the neural network to effectively act as the ultimate contextual tie-breaker for the deterministic string-matching algorithm.

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
