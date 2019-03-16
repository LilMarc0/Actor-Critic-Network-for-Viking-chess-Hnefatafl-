# Actor-Critic-Network-for-Viking-chess-Hnefatafl-
* Work in progress * A board representation of the viking chess and an untrained ( enough ) actor-critic network that tries to play it better than a human

Packages needed: numpy, pickle, tensorflow, tflearn

-- Latest version:
  - run Adversarial.py to train 'atc' and 'def' actor-critic models.
  - run PlayVs.py to play as attackers versus the defending AI.
  - Models now output the 588 possible moves on the 7x7 board; moves7x7.pickle dict is now used to
    convert model output to board move input.
  - Various bugs, stability issues and complexity faults fixed in Board.py
