# 20news-streamlit-app
## Why I Used BiLSTM Instead of Transformers

While transformer-based models like BERT or RoBERTa are state-of-the-art in text classification tasks such as 20 Newsgroups, I intentionally chose to build a **BiLSTM (Bidirectional LSTM) model from scratch**. This decision was based on both educational and practical motivations:

### Objective: Demonstrate End-to-End Model Building & Optimization

Rather than using pretrained models with millions of parameters, I wanted to showcase my ability to:
- Handle raw text data and perform robust preprocessing (tokenization, padding, vocabulary creation)
- Design and train a custom neural network architecture from scratch
- Optimize training through techniques like regularization, dropout, learning rate tuning, and evaluation metrics
- Balance generalization and performance without relying on pretrained contextual embeddings

### Why BiLSTM?

BiLSTMs are powerful for sequence modeling and can capture contextual dependencies in both forward and backward directions. They were widely used before transformers, and they still perform competitively when trained properly. BiLSTM allowed me to:

- Learn deep representations of text **without pretraining**
- Train all parameters **from scratch**, enabling full control over the model
- Show that with careful tuning, even non-transformer architectures can achieve strong results (e.g., ~87% accuracy on 20 Newsgroups)

### What I Achieved

- **Accuracy**: 86.7% on the 20-class classification task
- **Macro F1-score**: 0.865 — balanced performance across all categories
- Optimized from scratch using:
  - Custom tokenization and embedding handling
  - BiLSTM architecture with tuned hyperparameters
  - Regularization techniques (dropout, weight decay)
  - Learning rate scheduling and optimizer selection

# Why Not Transformers?

While transformers offer excellent performance, they come with trade-offs:
- Require extensive compute resources for fine-tuning
- Involve millions of pretrained parameters not learned from the task
- Obscure many of the *core optimization challenges* I aimed to solve manually

In contrast, my BiLSTM approach:
- Trains efficiently on a single GPU or CPU
- Provides interpretability and control over each layer
- Better reflects my understanding of model architecture and optimization

---

**In summary**, this project is about proving that strong NLP performance doesn’t have to come from a massive pretrained model. With BiLSTM and good engineering, you can still compete — and learn more deeply in the process.
