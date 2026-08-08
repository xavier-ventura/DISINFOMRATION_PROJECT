# Misinformation Detector for Immigration News in Spanish Digital Media

This repository contains the code and resources for the Misinformation Detector, a project focused on detecting and explaining misinformation about migration in Spanish political news. The system is deployed as a functional Telegram chatbot called FakeNewsBot.

For full details, please refer to the project documentation: OTHERS/Report_G1.pdf.

## Features

*   **Multiclass Classification**: Categorizes Spanish news items into three labels: VERDADERO (true), FALSO (false), and CONTEXTO (misleading context).
*   **Retrieval-Augmented Generation (RAG)**: Retrieves semantically similar verified documents from a ChromaDB vector database using Gemini Embedding 2 Preview.
*   **Forensic Explanations**: Generates evidence-based explanations in natural language using Llama 3.3 70B (via Groq API) [cite: 1].
*   **Ensemble Validation**: Utilizes a secondary model (Llama 3.1 8B) to validate or challenge the reasoning of the primary LLM, ensuring higher reliability.
*   **Continuous Improvement Pipeline**: An MLOps-inspired script (`pipeline_completo.py`) that automates scraping, synthetic data generation, RAG updates, and incremental model retraining.

## Architecture

The end-to-end system integrates several components:
1.  **Input Filter**: A lightweight keyword-based filter determines if the query relates to immigration.
2.  **Semantic Retrieval (RAG)**: Generates dense embeddings to query a vector database of 4,051 verified immigration-related fragments.
3.  **Classifier**: A fine-tuned XLM-ROBERTa model predicts the class probabilities and outputs a confidence score.
4.  **Confidence Threshold**: If the maximum probability falls below a calibrated threshold, the system abstains and returns INCIERTO.
5.  **LLM Generation**: Combines the retrieved documents, classifier prediction, and confidence distribution into a prompt for the LLMs to generate a final verdict and explanation.

## Dataset

The dataset was constructed by scraping news from four Spanish sources: El País, Cadena SER, Agencia EFE, and Maldita Migración. To address class imbalance (the original real-only dataset had an 11:1 ratio of true to false/context news), synthetic false and contextual variants were generated using the Gemini 3.1 API. The final balanced dataset contains 5,740 news items.

## Model Performance

Two transformer-based models were evaluated: BETO (Spanish-only) and XLM-ROBERTa (multilingual). XLM-ROBERTa was selected for deployment due to its superior generalization, particularly on the challenging CONTEXTO class.

**XLM-ROBERTa Results:**
*   **Accuracy**: 0.846
*   **Macro F1**: 0.782 
*   **CONTEXTO F1**: 0.554

## Technologies Used

*   **Language**: Python 
*   **Machine Learning**: Hugging Face Transformers, PyTorch
*   **Vector Database**: ChromaDB 
*   **APIs**: Gemini API, Groq API (Llama 3.3 and 3.1)
*   **Deployment**: Telegram API 

## Documentation

Detailed methodology, exploratory data analysis, and evaluation metrics can be found in OTHERS/Report_G1.pdf
