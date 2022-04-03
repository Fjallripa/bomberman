## History of the developed agents

### Development for Task 1: coin-heaven
Objective: Maximise coin-collecting speed
Approach: Develop a Q-learning algorithm to tackle that problem.

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
    3. `model_vq11_time`
      * Idea/Aim: 
        Log time of individual training functions to see what causes the training speed slowdown.
      * Setup/Changes: 
        Added a time taking functions stat log the processing times per step/round per function and store the in a .csv file under './logs/timing/'
      * Results/Observations:
        The slowdown was caused by a bug in end_of_round() which was promptly fixed, thus solving the issue.
    4. `model_vq11_10k`
      * Idea/Aim: 
        Check if slowdown issue is completely gone by performing a 10k rounds training
      * Results/Observations:
        The issue is gone, but the agent playing performance hasn't improved.

3. `agent_sq10`: symmetric q-learning version 1.0
    * Idea/Aim: 
      Reduce the size of the Q-matrix by exploiting symmetries in the feature design.
    * Results/Observations:
      by rewriting the state_to_features() function, the number of possible game states in Q were reduced from 81 to 15.
      The performance didn't improve noticably.
    * Trivia: 
      developed in new branch 'Symmetry-development' from agent_vq11

4. `agent_wq10`: working q-learning version 1.0
    * Idea/Aim:
      Find the bugs and design flaws in the q-learning by first making the debug logs much more detailed. Then they may be fixed.
    * Results/Observations:
      The improved log (`model_wq10_log`)uncovered a major design flaw where the rewards were one step out of sync with the actions and game states. This got fixed but sadly, even after a long training rund (`model_wq10_5k`) of 5,000 rounds, the tested performance didnt improve noticeably.
    * Trivia: 
      developed in new branch 'q-fixing' from agent_vq11
      locally stored analysis and log files got from now on collected centrally into model-specific subfolders of '/analysis/'
     
5. `agent_swq12`: symmetric working q-learning version 1.2
    * Idea/Aim:
      Merge agents sq10 and wq10 and further improve q-learning
    * Results/Observations:
      Despite multiple updates and experimenting, the playing performance didn't improve noticably
    * Trained Models:
      1. `sorted-features`
        * Idea/Aim:
          Try out the merged agent.
      2. `low-alpha`
        * Idea/Aim:
          Test if lowering the alpha to 0.01 improves anything.
      3. `zero-gamma`
        * Idea/Aim:
          Test if removing the influence of V_pi improves anything. In theory the agent should now be only concerned about wether the current move directly results in a coin or punishment.
      4. `round-based`
        * Idea/Aim:
          Tweaked the q-update to only update the Q-function after all step updates are complete instead of updating at every step.
        * Results/Observations:
          Besides not improving performance, we noticed that the round-wise update had the issue of only saving the latest step-update of every state-action-pair (S, a). In our setup with only actually 3 relevant states, this was a big issue.

6. `agent_swq13`: symmetric working q-learning version 1.3
    * Idea/Aim:
      Seriously improve q-learning by deep-diving into theory of q-update-algorithm
    * Results/Observations:
      
    * Trivia:
      * made a standard training analysis notebook
        `analysis_train.ipynb` that displays the model's Q-matrix with context, plots training performance as coins collected per round, and offers a function to plot the training evolution of certain Q-states. Used in all models of swq13.
      * automated the analysis data collection process
        in `file_handling.ipynb`. One function call now collects and renames every needed file into the model's respective folder unter '/analysis/'
    * Trained models:
      1. `better-gain`
        * Idea/Aim:
          Solving the issue of the last swq12 model, by averaging over all rewards of the same (S,a)
        * Setup/Changes: 
          Completely rebuilt the Q-update algorithm by collecting rewards the q-function by state instead of by step.
          The Q-update now consists of a mean value of rewards + expected gains V_pi for every (S, a).
        * Results/Observations:
          Somewhat improved performance. The agent now hops around but frequently gets stuck in infinite loops when landing in certain states.
      2. `test`
        * Idea/Aim:
          Tweaking the hyperparameters to see if this improves anything
        * Setup/Changes:
          Implemented epsilon annealing, tested for various final epsilon values, 
          tested to lower alpha,
          tested to set gamma to zero again
        * Results/Observations:
          Epsilon annealing: Observed reduction in training performance as training progressed. Suggests systematic error in how agent learns.
          Lower alpha and gamma to zero didn't seem to have any substantial impact on the learning here.
      3. `more-targets`
        * Idea/Aim: 
          Changed look_for_targets() so that it's capable of showing more than just one direction with closest coin. Hopefully these changes make the agent better distinguish between good and not so good directions.
        * Results/Observations: 
          No performance improvements here either. So this wasn't the issue. However, more game_states got utilized in the Q-matrix, not just 3 as before.
      4. `train-data`
        * Idea/Aim:
          Discovered bug in the collection of the training data where the agent got trained upon a state one step later than the one it acted upon. Solving this should fix the longstanding training issue.
        * Setup/Changes:
          Normal 100 rounds training
        * Results/Observations:
          This did actually improve training!!! Now, agent after about 20 rounds consistently collects all four coins. However, it's not perfect yet and there's a certain state where it still gets stuck in a loop.
      5. `10k`
        * Idea/Aim: 
          Verify the results of `train_data` by doing a very long training.
        * Setup/Changes:
          10,000 rounds training, same setup as in `train_data`.
        * Results/Observations:
          Sadly, after quick increase in performance with training round, it decreased again, resulting in an agent as bad as those before. Also the test performance of `train_data` is quite poor. Was it relying on random movements during training to get it unstuck?
      6. `reproduce`
        * Idea/Aim: 
          Try to reproduce performance of `train-data`. 
        * Setup/Changes:
          Set epsilon_last = 0.01, alpha = 0.01, gamma = 0
        * Results/Observations:
          Trained as bad as other models, maybe params were different.
      7. `reproduce2`
        * Idea/Aim: 
          Try again with different params
        * Setup/Changes:
          Set epsilon_last = 0.05, alpha = 0.1, gamma = 0
        * Results/Observations:
          Again super bad training. Looking at the Q-evolution, certain states seem to systematically learn the wrong action, never receiving reward for the right action.


7. `agent_swq14`: symmetric working q-learning version 1.4
    * Idea/Aim:
      Debug agent_swq13. There seem to be systematic issues in its training process.
    * Results/Observations:
      Found bug in appending actions to training data before using greedy epsilon and in analysis tool due to order of storing rounds in save_stats.
      Final result: practically optimal coin seeker (coin seeking rate = ca. 0.4), and Q-convergence after 1-10k rounds.

8. `agent_m1`: miner version 1
    * Idea/Aim:
      Expand features to be able train a crate-bombing coin-miner. Adapt Q-Learning and rest to fit the new features and see how well the agent can train on those and what issues arise.
    * Results/Observations:
      
    * Trivia:
      
    * Trained models:


### Template structure
0. `agent_template`:
    * Idea/Aim:
      
    * Results/Observations:
      
    * Trivia:
      
    * Trained models:
      1. `templa`
        * Idea/Aim:
        
        * Setup/Changes: 
        
        * Results/Observations:
        
      2. `templb`  


