## Tested Ideas

1. vanilla q-learning, simple 4D-feature, test on coin_heaven w/o opponents
   * agent folder called `agent_vq10/`  (stands for "vanilla q-learning version 1.0)
   * model stored as "model_vq10ch1.pt" (last part stands for "trained in coin-heaven scenario with only 1 player (itself))
   * constants:
     * feature.shape = (4), elements have 3 possible values
     * possible actions = 6
     * Q.shape = (3^4, 6) = (81, 6)
     * epsilon = 0.5
     * alpha   = 1
     * gamma   = 1