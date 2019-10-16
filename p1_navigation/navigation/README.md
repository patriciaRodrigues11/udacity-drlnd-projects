

# Project 1: Navigation

### Learning Algorithm
	
	The solution uses dqn and its varients (Double dqn, Duelling network and PER) and also shows the improvement in performance using fine tuned hyperparamters.

	Deep Q-Learning is a reinforcement learning algorithm that uses neural networks to approximate
	the optimal policy for achieving maximum reward in the given environment. Experiences (state,
	action, reward, and next state) are added to memory as the agent interacts with the environment
	to eventually be sampled and learned from. It was first proposed to evaluate states and actions on
	one network (local model) while the next state and epsilon-greed chosen action be evaluated on a
	separate network (target model) to maintain fixed Q-Targets and stabilize learning. The Double
	DQN method enhances this by choosing an action for the current state with the epsilon-greedy
	policy from the local model but evaluating it on the target model. This reduces overestimation of
	Q-Values by the online weights and allows further stabilization of learning. This Q-Value from
	the current state and the Q-Value from the next state are then used to compute the TD Error
	based on the Bellman Equation.
	
	

### Plot of Rewards
	 Please see the results section below for a detailed plot of all the appraoches.


### Ideas for Future Work

	Future work includes using the varients to get better performance and also training a dqn to use only raw pixels as input


# RESULTS

### RESULTS BY FINETUNING THE PARAMETERS (REFER FILE 168-eps.ipynb and 168-eps.pth)

1. default : start = 1.0, end = 0.01, decay = 0.995

	Episode 100	Average Score: 0.98
	Episode 200	Average Score: 4.24
	Episode 300	Average Score: 7.48
	Episode 400	Average Score: 10.19
	Episode 496	Average Score: 13.03
	Environment solved in 396 episodes!	Average Score: 13.03


2. start = 0.995, end = 0.01, decay = 0.995

	Episode 100	Average Score: 0.67
	Episode 200	Average Score: 4.21
	Episode 300	Average Score: 7.13
	Episode 400	Average Score: 10.16
	Episode 496	Average Score: 13.04
	Environment solved in 396 episodes!	Average Score: 13.04


3. max_t=1000, eps_start=0.995, eps_end=0.05, eps_decay=0.85

	Episode 100	Average Score: 4.28
	Episode 200	Average Score: 7.85
	Episode 300	Average Score: 12.12
	Episode 381	Average Score: 13.03
	Environment solved in 281 episodes!	Average Score: 13.03


4. max_t=1000, eps_start=0.995, eps_end=0.01, eps_decay=0.85

	Episode 100	Average Score: 2.99
	Episode 200	Average Score: 6.96
	Episode 300	Average Score: 12.20
	Episode 328	Average Score: 13.05
	Environment solved in 228 episodes!	Average Score: 13.05


5. max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.85

	Episode 100	Average Score: 4.08
	Episode 200	Average Score: 9.12
	Episode 300	Average Score: 12.66
	Episode 314	Average Score: 13.11
	Environment solved in 214 episodes!	Average Score: 13.11



6. max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.82

	Episode 100	Average Score: 4.71
	Episode 200	Average Score: 9.46
	Episode 264	Average Score: 13.07
	Environment solved in 164 episodes!	Average Score: 13.07




### RESULTS BY USING DIFFERENT DQN VARIENTS (REFER FILES dqn_agent.py, dqn_agent_PER.py , model.py (contains duelingQ network))

1. FIXED TARGETS
	Episode 100	Average Score: 0.98
	Episode 200	Average Score: 4.24
	Episode 300	Average Score: 7.48
	Episode 400	Average Score: 10.19
	Episode 496	Average Score: 13.03
	Environment solved in 396 episodes!	Average Score: 13.03


2. DOUBLE DQN

	Episode 100	Average Score: 1.69
	Episode 200	Average Score: 6.38
	Episode 300	Average Score: 10.19
	Episode 376	Average Score: 13.04
	Environment solved in 276 episodes!	Average Score: 13.04



3. DuelingQNetwork + Double DQN 

	Episode 100	Average Score: 0.70
	Episode 200	Average Score: 4.42
	Episode 300	Average Score: 5.76
	Episode 400	Average Score: 10.63
	Episode 457	Average Score: 13.03
	Environment solved in 357 episodes!	Average Score: 13.03



4. DuelingQNetwork + Double DQN + PER

	Episode 100	Average Score: 0.97
	Episode 200	Average Score: 4.14
	Episode 300	Average Score: 8.90
	Episode 400	Average Score: 11.03
	Episode 492	Average Score: 13.10
	Environment solved in 392 episodes!	Average Score: 13.10



