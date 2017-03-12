# Recurrent Weighted Average

This is a re-implementation of the architecture described in [Machine Learning on Sequential Data Using a Recurrent Weighted Average](https://arxiv.org/abs/1703.01253).

# Hypotheses

As the sequence gets longer and longer, the running average could become more and more "saturated" (i.e. new time-steps matter less and less). This might cause the network to have more and more trouble forming short-term memories as the sequence goes on. As a result, the network might do poorly at precise tasks like text character prediction.

If the above concern is actually an issue, perhaps the long-term benefits of RWAs could still be leveraged by stacking an RWA on top of an LSTM.

# Results

Here are the experiments I have run:

 * [char-rnn](experiments/char_rnn) - RWAs can learn to model language character-by-character, although LSTMs are faster and better.
 * [sentiment](experiments/sentiment) - A hybrid LSTM-RWA model learns to predict the sentiment of tweets faster than a plain LSTM.
