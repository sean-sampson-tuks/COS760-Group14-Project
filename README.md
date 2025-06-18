# COS760-Group14-Project

# Exploring Semantic Relatedness in Amharic and Hausa Using Transfer Learning

## Project Overview
This project benchmarks the performance of several pre-trained multilingual models on semantic relatedness tasks for two low-resource African languages: Hausa and Amharic. The models are fine-tuned and evaluated against human-annotated scores from the [SemRel 2024 dataset](https://github.com/semantic-textual-relatedness/Semantic_Relatedness_SemEval2024). The analysis also explores the effectiveness of ensemble methods to improve performance.

## Methodology

### Data Sources and Preprocessing
**Data Source:** The project uses the SemRel dataset from the Semantic Textual Relatedness SemEval 2024 GitHub repository. The specific files used are `hau_train.csv` for Hausa and `amh_train.csv` for Amharic.

**Preprocessing:** For both languages, the raw 'Text' column, which contained sentence pairs separated by a newline, was split into 'Sentence1' and 'Sentence2' columns. The resulting dataframes were then partitioned into training (80%) and validation (20%) sets.

### Model Training and Evaluation
* **Models:** Four pre-trained multilingual models were selected for fine-tuning:
    * XLM-ROBERTa (XLM-R)
    * Multilingual BERT (mBERT)
    * AfroXLMR
    * AfriBERTa

* **Baseline:** A baseline was established using the `paraphrase-multilingual-MiniLM-L12-v2` model to compute cosine similarity scores without fine-tuning.
* **Training:** The models were fine-tuned for a maximum of 15 epochs using a `TransformerRegressor` architecture. The training process utilised the AdamW optimiser, Mean Squared Error (MSE) loss, a batch size of 8, and an early stopping mechanism with a patience of 3 to prevent overfitting.
* **Evaluation:** Model performance was measured against the human-annotated ground-truth scores using two metrics: Spearman's rank correlation coefficient and Mean Squared Error (MSE).

## Results
The fine-tuned models showed a significant improvement over the baseline for both languages. Models specifically trained on African languages, such as AfroXLMR and AfriBERTa, generally achieved the highest performance.

### Hausa Results
The Hausa dataset contained 1,736 sentence pairs. The baseline cosine similarity score was a Spearman correlation of 0.1726.

| Model | Spearman Correlation | MSE |
| :--- | :--- | :--- |
| **Ensemble (AfroXLMR + XLM-R)** | **0.7312**  | **0.0329**  |
| AfroXLMR | 0.7083  | 0.0372  |
| XLM-R | 0.7035  | 0.0376  |
| mBERT | 0.6143  | 0.0455  |
| AfriBERTa | 0.5095  | 0.0536  |

### Amharic Results
The Amharic dataset contained 992 sentence pairs. The baseline cosine similarity score was a Spearman correlation of 0.3825.

| Model | Spearman Correlation | MSE |
| :--- | :--- | :--- |
| **Ensemble (All Models)** | **0.8423**  | **0.0146**  |
| AfroXLMR | 0.8127  | 0.0223  |
| AfriBERTa | 0.7802  | 0.0188  |
| XLM-R | 0.7564  | 0.0216  |
| mBERT | 0.1951  | 0.0565  |

## How to Run the Code

### Prerequisites
Ensure you have Python installed with the following libraries:
`datasets`, `transformers`, `sentence-transformers`, `scipy`, `pandas`, `numpy`, `seaborn`, `matplotlib`.

You can install them using pip:
`!pip install -q datasets transformers sentence-transformers scipy pandas numpy seaborn matplotlib` 

### Setup and Execution

The project is contained within two Jupyter notebooks:
* `group14_semrel_complete_hausa_new_latest.ipynb` for the Hausa language analysis.
* `group14_semrel_complete_amharic (1).ipynb` for the Amharic language analysis.

To run the analysis, open either notebook in a compatible environment (like Google Colab) and execute the cells in order. The notebooks will automatically clone the required [`Semantic_Relatedness_SemEval2024`](https://github.com/semantic-textual-relatedness/Semantic_Relatedness_SemEval2024) repository to access the datasets.

## Reflections and Discussion

### Key Findings and Challenges

* **What Worked:** Fine-tuning pre-trained models, especially those with a focus on African languages (AfroXLMR, AfriBERTa), proved to be a highly effective strategy. Simple ensembling by averaging predictions further improved the results.
* **What Didn't Work:** The baseline cosine similarity approach was insufficient for this task. Multilingual BERT (mBERT) significantly underperformed the other models, particularly on the Amharic dataset, suggesting it may lack robust representations for these languages.
* **Challenges:** The primary challenge was the low-resource nature of the datasets, which limits model generalisation. The error analysis showed that all models struggled with linguistic nuance, such as shared keywords with different meanings.
