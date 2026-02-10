# Ball Balancing AI

## Problem description
  The game consists of a ramp and a ball in a two dimensional space. The objective of the game is to balance the ball on the ramp
  and keep it close to the center of it by changing its tilt. One round lasts ten seconds, the ball is randomly placed on the ramp 
  with no initial velocity and the ramp is flat. The player gets rewarded points for not dropping the ball off the edge and 
  gets more points the closer the ball is to the center.

  The inputs are the incline angle of the ramp, distance between the ball and the center of the ramp, the horiozntal and the vertical 
  velocity of the ball. The inputs vary in sizes which can have a significant impact on the performance and the training process of the 
  neural network, therefore all the inputs were normalized to a range of [-1, 1].
  The action space is discrete and it consists of three options: nothing, increase the incline angle of the ramp and decrease it.

## Solution
  ### Proximal Policiy Optimization
  Proximal Policy Optimization (PPO) is a reinforcement learning algorithm that optimizes a policy. It improves the stability of
  training using a clipped objective function, which prevents large updates to the policy. It works by collecting data from an episode
  with the current policy, estimating advantages and updating the policy with a loss function. The advantages are estimated by taking
  the difference of the discounted sum of rewards and the expected return value from the state predicted by a seperate value neral network.

  ### Neural Network Architecture
  The policy neural network is a fully connected neural network with an input layer of four neurons, two hidden layers and an output 
  layer with 3 neurons. The output layer uses the Softmax activation function while the others use the ReLu activation function. 
  The value neural network has the same architecture as the policy neural network except that the output layer has one neuron that has
  no activation funciton.

  ### Training
  The agent plays our the episode and records the states, actions, rewards and probabilities of the taken actions until it reaches the terminal state.
  Then the return at each time step is evaluated and is then used to calculate the advantage at each time step. From the experience a policy loss function
  and a value loss function is calculated and applied to the neural networks using the Adam optimizer.

  
