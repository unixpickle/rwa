# Sentiment Analysis

This tests the RWA on a pretty routine task: Twitter sentiment analysis. The model is fed a Tweet byte-by-byte. The network's last output is used to classify the tweet as "positive" or "negative".

# Data

This uses the data from [Sentiment140](http://help.sentiment140.com/for-students/).

# Results

I trained an LSTM for 1.5 epochs on this task. It achieved a validation cross-entropy of 0.39 and a validation accuracy of 86.1%. Note that the original Sentiment140 paper only managed to achieve 82.7% accuracy, and that required a ton of feature engineering.

The LSTM did seem to learn faster than a plain RWA. Now, I want to see if an LSTM-RWA hybrid can outperform the classic LSTM. I am saving plots of both learning curves, so comparison will be possible.
