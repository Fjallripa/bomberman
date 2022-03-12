## Tested Ideas

1. `agent_vq10`: vanilla q-learning, simple 4D-feature, test on coin_heaven w/o opponents
   * agent folder called `agent_vq10/`  (stands for "vanilla q-learning version 1.0)
   * model stored as "model_vq10ch1.pt" (last part stands for "trained in coin-heaven scenario with only 1 player (itself))
   * constants:
     * feature.shape = (4), elements have 3 possible values
     * possible actions = 6
     * Q.shape = (3^4, 6) = (81, 6)
     * epsilon = 0.5
     * alpha   = 1
     * gamma   = 1
   * capped epsilon_greedy random action to only "walk" (prevents frequent suicides)
   1. `model_vq10_10k`: Trained model in almost 10,000 steps.
      * Doesn't move at all in tests. :(

2. `agent_vq11`: vanilla q-learning version 1.1
    * Testing with 
      * limited actions (no bombs or waiting),
      * some auxiliary rewards
    1. `model_vq11_ag-one` hyperparameters alpha and gamma set to 1 (as in vq10)
      * No test improvement vs. vq10
    2. `model_vq11_ag-half` alpha, gamma now 0.5
      * Moves one step when starting in lower right corner. :(

