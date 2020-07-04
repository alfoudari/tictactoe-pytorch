# Tic-tac-toe RL Agent

This is a Tic-tac-toe game player trained with Reinforcement Learning.

The player depends on a simple [backend][1] and a reinforcement learning [library][2] that I made. It starts by knowing nothing about the game and gains knowledge by playing against an automated player that plays randomly.

Player X is Player 1 and Player O is Player 2, I use them interchangeably. Player X is the agent and Player O is whatever you want it to be.

My aim in creating this agent is to both understand Reinforcement Learning in terms of whats going on under the hood as well as train an agent that plays optimally. 

Considering the following aspects of Tic-tac-toe:
- It lives in a relatively small space (3x3) as opposed to Chess (8x8) or Go (19x19).
- It is a solved game, meaning that optimal strategies exist and can be coded with hard rules.

Therefore, it is reasonable to expect Reinforcement Learning to be able to learn an optimal strategy by getting better and better as it observes and learn from experience.

## Design

I have used a [DQN algorithm][4] with the following enhancements:
- Double DQN - calculate state values according to actions selected by the current behavior policy rather than the target network.
- Prioritized Replay Experience. [[paper]][5][[blog post]][6]

The prioritized replay experience in particular boosted learning in a big way and accelerated learning.

For the neural network used to predict Q-values, I have used a Convolutional Neural Network since there are spatial relationships between data points (two X's next to eachother matter as opposed to two X's far from each other).

The environment is based on OpenAI gym and has a reward structure as follows:
- Win: +1
- Lose: -1
- Draw: 0.5
- Step: -0.1

## Training and Performance

One of the metrics that measures the agent's winning stability is the standard deviation of evaluation episodes. For example, when I evaluate over 1000 episodes, I could get an average reward of 0.75 (meaning on average the agent is doing great and winning mostly), however outliers are not detected if the average is observed independently. Hence, the standard deviation is observed for the evaluation list indicating how far off values are from the mean of the list. During training, the standard deviation was getting minimized over time which means there are less outliers and more consistent wins.

Epsilon was decayed over time from an initial value near 1 (meaning explore almost all the time) to an epsilon limit of 0.1 (meaning act optimally 90% of the time and explore 10% of the time). This encourages exploration at the beginning to gather experience and slowly starts favoring optimal actions over random exploratory actions.

![Training](/docs/tictactoe_ddqn_prb.png)
*An agent that was trained for 34 minutes on an experience of around 100k games.*

The agent's performance currently is far from optimal. It does prioritize blocking the opponent from winning in certain states, however in analogous but different states it prioritizes winning rather than blocking, not realizing that the opponent is one step away from winning the game. Since the reward for intermediate steps is equal, this is almost definitely a problem with the state value.

There are a few improvements that I can make that should bring it closer to optimality. All improvements but the last one are from the [Rainbow DQN paper][3], namely:
- [ ] Adding noise to the neural network to further increase exploration.
- [ ] Dueling DQN.
- [ ] Periodically save the trained agent and use it *as an opponent* instead of playing against a random player all the time.

> Any suggestions or improvements are welcome. Please create a github issue or a pull request if you have something to contribute.

## Defining Player 2 policy

The environment `TicTacToeEnv` defined in `env/base.py` allows you to subclass it and define the behavior of player 2:

```python
@abc.abstractmethod
def player2_policy(self):
    """
    Define player 2 (non-agent) policy
    """
```

For example, this is an environment that makes player 2 play randomly:

```python
class Env(TicTacToeEnv):
    def player2_policy(self):
        random_action = random.choice(self._board.empty_cells)
        self._player2.mark(*random_action)
```

## Play

To play the game, you can run the following in `play/`:

### `agent_vs_random.py`

Plays a trained agent vs a random player for `num_games` times, default is 1000. Player X is the trained agent and Player O is the random player:

```console
$ python -m play.agent_vs_random -algorithm dqn -net_type cnn -policy dqn-cnn -num_games 1000
Game #1000 - Iteration duration: 0.0016622543334960938

X Won: 750
O Won: 172
Draw:  78
Win percentage: 75.0
Win+Draw percentage: 82.8
Loss percentage: 17.2
```

### `agent_vs_human.py`

Allows you to play against your trained agent. Player X (1) is the trained agent and Player O (2) is you. You can pass specify the first player in `fp` flag (example: `-fp=2`; default is player 1).

If you specify `-debug` then the neural network final layer values are printed, showing you what the agent thinks of each action on the board.

Cell coordinates start from (0,0) to (2,2):

```
$ python -m play.agent_vs_human -algorithm dqn -net_type cnn -policy dqn-cnn -fp 2 -debug
Enter cell coordinates (e.g. 1,2): 1,1
Player 2: Marking 1 1
   |   |
------------
   | O |
------------
   |   |
------------

action distribution:
tensor([[0.7591, 0.7115, 0.8669],
        [0.4905,   -inf, 0.5601],
        [0.7362, 0.5985, 0.7299]], grad_fn=<ViewBackward>)
action, max(action_dist): 2, 0.866879940032959

Player 1: Marking 0 2
   |   | X
------------
   | O |
------------
   |   |
------------

Enter cell coordinates (e.g. 1,2): 0,0
Player 2: Marking 0 0
 O |   | X
------------
   | O |
------------
   |   |
------------

action distribution:
tensor([[  -inf, 0.6825,   -inf],
        [0.5657,   -inf, 0.6621],
        [0.4554, 0.5861, 0.8073]], grad_fn=<ViewBackward>)
action, max(action_dist): 8, 0.8072749376296997

Player 1: Marking 2 2
 O |   | X
------------
   | O |
------------
   |   | X
------------
Enter cell coordinates (e.g. 1,2):
```

and so on.


[1]: https://github.com/abstractpaper/tictactoe
[2]: https://github.com/abstractpaper/prop
[3]: https://arxiv.org/abs/1710.02298
[4]: https://en.wikipedia.org/wiki/Q-learning#Deep_Q-learning
[5]: https://arxiv.org/abs/1511.05952
[6]: https://danieltakeshi.github.io/2019/07/14/per/