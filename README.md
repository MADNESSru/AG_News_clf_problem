# AG News Classification

## 📌 Task
The goal of this project is to build a **news classification model** on the **AG News** dataset into four categories:
1. World
2. Sports
3. Business
4. Sci/Tech

---

## 📂 Project Structure
- **Data Loading and Preprocessing** — data loading, cleaning, and vectorization.
- **Model Training** — training the Random Forest classifier on vectorized data.
- **Evaluation** — evaluating the model on the test set.

---

## 🚀 Implementation Steps
### 1️⃣ Data Loading
- The AG News dataset was loaded and split into training and test sets: `train_x`, `train_y`, `test_x`, `test_y`.
- Basic data analysis and missing value checks were performed.

### 2️⃣ Text Vectorization
- **CountVectorizer** was used to transform the text data into numerical format.
- The resulting vectorized data was stored in `train_arr` and `test_arr`.

### 3️⃣ Model Training
- A **RandomForestClassifier** model was selected for classification.
- The model was trained on vectorized data, with labels taken directly from the dataset.

### 4️⃣ Evaluation
- The following metrics were used for evaluation:
  - **Accuracy** — overall model accuracy.
  - **Precision** — accuracy of predictions for each class.
  - **Recall** — completeness of predictions.
  - **F1-score** — harmonic mean of Precision and Recall.
- The final results showed good accuracy on the test data.

---

## ⚙️ How to Run
To run the notebook and reproduce the results:
1. Clone the repository:
    ```bash
    git clone https://github.com/MADNESSru/AG_News_clf_problem.git
    ```
2. Navigate to the project directory:
    ```bash
    cd AG_News_clf_problem
    ```
3. Launch the notebook:
    ```bash
    jupyter notebook AG_news_classification.ipynb
    ```

---

## 📌 Dependencies
Below is the list of required libraries:
```bash
pip install nltk
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install wordcloud
pip install tqdm
pip install numpy
pip install pandas
```

### Full List of Libraries:
- **Text preprocessing and vectorization**
    - nltk
    - re
    - string
    - stopwords from nltk.corpus
    - PorterStemmer from nltk.stem
    - TfidfVectorizer, TfidfTransformer from sklearn.feature_extraction.text  

- **Visualization**
    - matplotlib
    - seaborn
    - WordCloud
    - tqdm  

- **Data Analysis**
    - numpy
    - pandas
    - os  

- **Models**
    - MultinomialNB from sklearn.naive_bayes
    - RandomForestClassifier from sklearn.ensemble  

- **Metrics**
    - accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay from sklearn.metrics  

---

## 💡 Ideas for Improvement
- Use neural network architectures such as **TextCNN** and **LSTM** to improve the results.
- Fine-tune LLM models like **BERT** or other transformer-based models for this classification task.
