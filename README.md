# Recurrent Weighted Average

This is a re-implementation of the architecture described in [Machine Learning on Sequential Data Using a Recurrent Weighted Average](https://arxiv.org/abs/1703.01253).

# Hypotheses

As the sequence gets longer and longer, the running average could become more and more "saturated" (i.e. new time-steps matter less and less). This might cause the network to have more and more trouble forming short-term memories as the sequence goes on. As a result, the network might do poorly at precise task like text character prediction.

If the above concern is actually an issue, perhaps the long-term benefits of RWAs could still be leveraged by stacking an RWA on top of an LSTM.

# Results

I created a [char-rnn branch](https://github.com/unixpickle/char-rnn/tree/rwa) that uses RWA. The RWA did not learn to do better than random at English text prediction, even when allowed to run for way more epochs than an LSTM (which learned extremely quickly). This seems to confirm my hypothesis about the model's shortcomings.

# TODO

 * Perform operations in log domain to avoid floating-point overflow
