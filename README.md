## Background/Motivation:
Eating disorders are behavioral conditions characterized by severe and persistent disturbance in eating behaviors and associated distressing thoughts and emotions (American Psychiatric Association, 2023). They  are complex mental health conditions that can have medical and psychological impacts. Patients often turn to online communities such as Reddit to discuss and seek support for eating disorders. Subreddits like r/EatingDisorders and r/EDRecoverySnark are spaces where users can share recovery stories, seek advice, and raise concerns. It houses tens of thousands of messages on the subject of eating disorders. These forums, however, also present challenges as they can include content ranging from supportive dieting tips to potentially harmful discussions about dieting and eating behavior. The sometimes unmoderated and open nature of Reddit poses a challenge in effectively identifying and managing posts that may be harmful or trigger relapse in individuals with eating disorders. Here lies the opportunity for automated systems to aid in identifying concerning content.

## Results:
<img width="1632" alt="image" src="https://github.com/user-attachments/assets/b93cd983-f76e-4892-8e99-c3088a96331f" />

<img width="1632" alt="image" src="https://github.com/user-attachments/assets/4f1b8b4f-313e-4be9-8d52-b1b956be3c41" />

<img width="1632" alt="image" src="https://github.com/user-attachments/assets/01980321-a603-4127-8566-d97d8f00deac" />


## Methods:

Breakdown of objective:
- Accurately identify eating disorder-related content
- Distinguish it from general diet and nutrition discussions
- Provide transparent explanations for its classifications
- Help understand what linguistic patterns are associated with eating disorders

This project uses a BERT-based model to distinguish between general diet-related content and potentially concerning eating disorder content. The reasoning behind the architecture is due to the sensitivity of the content. BERT serves as our building block to understand specific nuances in language which might be very difficult to recreate from scratch with the time constraints. I used LIME explainable AI techniques to highlight why the model makes its decisions.

This project utilizes DistilBERT over BERT. For the size of the set it was likely to create an overfitted model with the heavy duty BERT (Devlin et al., 2018). DistilBERT uses 6 Transformer encoder layers (compared to BERT's 12). It mostly includes self-attention layers. This helps the model focus on relevant words regardless of their position in the sentence.

Dataset: https://paperswithcode.com/dataset/reddit-posts-related-to-eating-disorders-and
From description: “This dataset comprises 77,175 Reddit posts from 115 subreddit forums, annotated for the presence of 15 topics related to eating disorders and dieting. The dataset includes labels and scores on all 77,175 Reddit posts, determined by 5 Large Language Models. The dataset also includes a subset of 1,080 human-annotated posts for evaluation.” (Qiu, 2025)

Longest message length: 21613 characters
Shortest message length: 43 characters
Average length: 907.80

I utilized the key.csv containing a mix of human and generated ed or diet labels for each post and the topic_gpt4o.csv focusing on only text columns not the generated labels created by the LLM.

# Walkthrough:
`load_filtered_data()`
This preprocessing step converts labels to binary. It allows me to pose the question: is a post related to eating disorders (1) or just general diet content (0)?

BERTDataGenerator
To handle the large dataset efficiently, I implemented a custom data generator. Outputting the following return to match the expected inputs need for DistillBERT:

       return {
           'input_ids': encodings['input_ids'],
           'attention_mask': encodings['attention_mask']
       }, batch_labels

`create_bert_model()`
The key aspects of this architecture are:
Creating input layer for inputs from the generator
Creating input layer for the attention_mask from the generator
Leveraging DistilBERT layer for more contextual understanding of language
Taking the [CLS] token, which summarizes the representation for classification
Adding dropout for regularization
Passing regularized CLS token to a softmax output layer for binary classification

Similar to class examples and assignments, this generator processes data in batches, which is crucial for memory efficiency when working with large text datasets and transformer models.

`train_bert_model()`
The training process includes:
Splitting data into training and validation sets
Creating data generators
Setting up callbacks for early stopping and model checkpointing (great add for when session disconnects)
Training for 3 epochs with a small batch size. I chose 3 since I noticed the accuracy was more than ideal with more. 
Saving the model, tokenizer, and training history
Overall training took 3 hours running with T4-GPU. Callbacks were heavily relied on during experimentations. 
