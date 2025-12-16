# **Emotion Detection from Keystroke Dynamics**

Developed a deep learning model to detect emotions based on typing behavior, leveraging keystroke dynamics.

---

## **Project Overview**

- Analyzed **keystroke dynamics data** to identify feature differences across emotions.  
- Implemented a **dual-attention Transformer model** capturing both timing and channel dependencies in typing patterns.  
- Validated the model for accurate emotion recognition, achieving **85% accuracy**.

---

## **Key Features**

- **Dual-Attention Transformer:** Simultaneously models temporal patterns (timing) and multi-channel keystroke data.  
- **Emotion Classification:** Predicts emotional states such as happiness, sadness, anger, etc., based on typing behavior.  
- **Data-Driven Insights:** Identifies keystroke features that correlate strongly with different emotional states.

---

## **Model architecture**
![Model Architecture](images/model_architecture.png)

- **Input:** Consecutive sequences of feature vectors
- **Positional Encoding:** Gaussian positional encoding applied to inputs
- **Transformer Encoder:** Incorporates dual-attention mechanism
  - **Temporal Attention Branch:** Captures timing dependencies
  - **Channel Attention Branch:** Captures feature/channel dependencies
- **Add & Normalize:** Outputs of both branches are summed and normalized
- **CNN Block:** Processes combined representation
- **Add & Normalize:** Another normalization after CNN
- **Feed-Forward Network:** Produces final emotion classification output
