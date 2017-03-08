# Recurrent Weighted Average

This is a re-implementation of the architecture described in [Machine Learning on Sequential Data Using a Recurrent Weighted Average](https://arxiv.org/abs/1703.01253).

# Hypotheses

As the sequence gets longer and longer, the running average could become more and more "saturated" (new timesteps could matter less and less). This might cause the network to have more and more trouble forming short-term memories as the sequence goes on. This would greatly hinder the network's performance at things like character-level text prediction.

If the above concern is actually an issue, perhaps the long-term benefits of RWAs would still be leveraged by stacking an RWA on top of an LSTM.
