# Fake News Detection and Visualization

This project focuses on building a fake news detection system using machine learning. The dataset used for training and testing contains labeled news articles, indicating whether they are "FAKE" or "REAL."

# Project Structure
- fake_news_detection.ipynb: Jupyter Notebook containing the complete code for the project.
- news.csv: Dataset file containing labeled news articles.

# Workflow
1. Data Loading and Exploration
   - Load the dataset using pandas.
   - Explore the dataset to understand its structure and characteristics.

2. Data Preprocessing
   - Split the dataset into training and testing sets.
   - Tokenize and vectorize the text data using TF-IDF.

3. Model Training
   - Use a Passive Aggressive Classifier for training.

4. Model Evaluation
   - Predict on the test set.
   - Evaluate the model's accuracy and create a confusion matrix.

5. Visualizations
   - Created visualizations using seaborn and matplotlib.
   - Visualized the distribution of labels, word cloud, confusion matrix, text lengths, and top N words.

## Credits
- Project developed by Chukwuemeka Eugene Obiyo.
- Dataset source: https://drive.google.com/file/d/1er9NJTLUA3qnRuyhfzuN0XUsoIC4a-_q/view
