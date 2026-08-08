# Misinformation Detector for Immigration News in Spanish Digital Media

This repository contains the code and resources for the Misinformation Detector, a project focused on detecting and explaining misinformation about migration in Spanish political news [cite: 1]. The system is deployed as a functional Telegram chatbot called FakeNewsBot, as well as a Gradio web interface [cite: 1].

For full details, please refer to the project documentation: Report_G1.pdf.

## Features

*   **Multiclass Classification**: Categorizes Spanish news items into three labels: VERDADERO (true), FALSO (false), and CONTEXTO (misleading context) [cite: 1].
*   **Retrieval-Augmented Generation (RAG)**: Retrieves semantically similar verified documents from a ChromaDB vector database using Gemini Embedding 2 Preview [cite: 1].
*   **Forensic Explanations**: Generates evidence-based explanations in natural language using Llama 3.3 70B (via Groq API) [cite: 1].
*   **Ensemble Validation**: Utilizes a secondary model (Llama 3.1 8B) to validate or challenge the reasoning of the primary LLM, ensuring higher reliability [cite: 1].
*   **Continuous Improvement Pipeline**: An MLOps-inspired script (`pipeline_completo.py`) that automates scraping, synthetic data generation, RAG updates, and incremental model retraining [cite: 1].

## Architecture

The end-to-end system integrates several components:
1.  **Input Filter**: A lightweight keyword-based filter determines if the query relates to immigration [cite: 1].
2.  **Semantic Retrieval (RAG)**: Generates dense embeddings to query a vector database of 4,051 verified immigration-related fragments [cite: 1].
3.  **Classifier**: A fine-tuned XLM-ROBERTa model predicts the class probabilities and outputs a confidence score [cite: 1].
4.  **Confidence Threshold**: If the maximum probability falls below a calibrated threshold, the system abstains and returns INCIERTO [cite: 1].
5.  **LLM Generation**: Combines the retrieved documents, classifier prediction, and confidence distribution into a prompt for the LLMs to generate a final verdict and explanation [cite: 1].

## Dataset

The dataset was constructed by scraping news from four Spanish sources: El País, Cadena SER, Agencia EFE, and Maldita Migración [cite: 1]. To address class imbalance (the original real-only dataset had an 11:1 ratio of true to false/context news), synthetic false and contextual variants were generated using the Gemini 3.1 API [cite: 1]. The final balanced dataset contains 5,740 news items [cite: 1].

## Model Performance

Two transformer-based models were evaluated: BETO (Spanish-only) and XLM-ROBERTa (multilingual) [cite: 1]. XLM-ROBERTa was selected for deployment due to its superior generalization, particularly on the challenging CONTEXTO class [cite: 1].

**XLM-ROBERTa Results:**
*   **Accuracy**: 0.846 [cite: 1]
*   **Macro F1**: 0.782 [cite: 1]
*   **CONTEXTO F1**: 0.554 [cite: 1]

## Technologies Used

*   **Language**: Python [cite: 1]
*   **Machine Learning**: Hugging Face Transformers, PyTorch [cite: 1]
*   **Vector Database**: ChromaDB [cite: 1]
*   **APIs**: Gemini API, Groq API (Llama 3.3 and 3.1) [cite: 1]
*   **Deployment**: Telegram API, Gradio [cite: 1]

## Documentation

Detailed methodology, exploratory data analysis, and evaluation metrics can be found in others/Report_G1.pdf
