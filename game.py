#Part 0 - Installing the required packages and importing the libraries

#Installing Gymnasium environment for the game
!pip install gymnasium
!pip install "gymnasium[atari, accept-rom-license]"
!apt-get install -y swig
!pip install gymnasium[box2d]

#Import relevant libraries
import os # Used for operating system GPU and CPU
import random
import numpy as np # used to work with arrays and matrices in the CNN model
import torch # used for building and raining our agent pacman with pytorch
import torch.nn as nn #neural network module for buiding the architecture
import torch.optim as optim # optimizer used when training the agent
import torch.nn.functional as F # functions part of torch library that will be used during training
from collections import deque # used for storing memory of experiences during the training phase
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

#Part 1: Creating the architecture of the Neural Network (The eyes and the brain of the agent Pacman)
class Neural_Network(nn.Module):

  def __init__(self, action_size, seed = 42):
    super(Neural_Network, self).__init__()  # inheret all the features of the class nn
    self.seed = torch.manual_seed(seed) # a factor for random to ensure reproducability

    #Convolutional Neural Network(The Eyes) The state is an image of rgb pixels
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size = 8, stride = 4) #First convolution layer; The state is an image of rgb pixels, 3 color input channels
    self.bn1 = nn.BatchNorm2d(32) # Batch Normalization: Help in accelerating the training process faster and make it more stable. Takes the # of features in the previous conv layer.
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size = 4, stride = 2)
    self.bn2 = nn.BatchNorm2d(64)
    self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size = 3, stride = 1)
    self.bn3 = nn.BatchNorm2d(64)
    self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size = 3, stride = 1)
    self.bn4 = nn.BatchNorm2d(128)

    #Deep Q Network(The Brain)
    self.fullconnection1 = nn.Linear(10 * 10 * 128, 512) # First full connection layer; Flattening Layer:the output features resulting from flattening all previous convolutions
    self.fullconnection2 = nn.Linear(in_features=512, out_features=256) # Second full connection layer
    self.fullconnection3 = nn.Linear(in_features=256, out_features=action_size) # Third and last full connection layer,return the actions values based on Q values

    #Forward Propogation: forward signal states from the CNN archiecture through DQN of the agent
  def forward(self, state): # takes the state images as input
    x = F.relu(self.bn1(self.conv1(state))) # Forwarding signal from state to first conv layer, and then from conv to the batch normalization layer. Activated by Rectified Linear Unit(reLU)
    x = F.relu(self.bn2(self.conv2(x)))
    x = F.relu(self.bn3(self.conv3(x)))
    x = F.relu(self.bn4(self.conv4(x)))
    x = x.view(x.size(0), -1) #reshape to get flattening layer while keeping the first dimension of the batch

    #Forward signal through the DQN to the output actions
    x = F.relu(self.fullconnection1(x))
    x = F.relu(self.fullconnection2(x))
    return self.fullconnection3(x) # return the forwarded signal

#Part 2 - Training the AI

#Setting up the environment
import gymnasium as gym
env = gym.make('MsPacmanDeterministic-v0', full_action_space = False) # We used a deterministic variant and simplified action space for the game to reduce the complexity when training the agent
state_shape = env.observation_space.shape # rgb images dimensions
number_actions = env.action_space.n # The environment we used is MsPacman, which has 9. This can be found in the website.

#Initializing the hyperparameters
learning_rate = 5e-4
minibatch_size = 64 # The number of observations used in 1 step of the training to update the model parameters
discount_factor = 0.99 # a factor used to discount future rewards to encourage the agent to take different actions and learn faster

#Preprocessing the frames
from PIL import Image
from torchvision import transforms

#Preprocess the frames so that input images can be converted to Pytorch tensors that are accepted by the neural network of the agent
def preprocess_frame(frame):
  frame = Image.fromarray(frame) # The frame comes in the form of numpy array . So convert the numpy array into a PIL object
  preprocess = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()]) # Based on the observation space from Gymnasium environemnt. We also read a paper to help us in this process. We added it in the reference section
  return preprocess(frame).unsqueeze(0) # add a dimension of the batch(which is index 0) to keep track
#Implementing the DCQN class
class Agent():

  def __init__(self, action_size):
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # If couldn't find a GPU, then it will use CPU of the computer as a backup
    self.action_size = action_size

    #Create Q network and Target Q Network for backprop step, improving the process of learning
    self.qnetwork = Neural_Network(action_size).to(self.device)
    self.target_qnetwork = Neural_Network(action_size).to(self.device)
    self.optimizer = optim.Adam(self.qnetwork.parameters(), lr = learning_rate) # Optimizer method for a stochastic environment
    self.memory = deque(maxlen = 10000) # Replay Memory Buffer to train the Q network in predicting actions

  #Step method: Stores experiences and decides when to learn from them
  def step(self, state, action, reward, next_state, done):
    state = preprocess_frame(state) # Preprocess(convert to Torch sensors) the state and next state because they come in the form of numpy arrays
    next_state = preprocess_frame(next_state)
    self.memory.append((state, action, reward, next_state, done)) # Store the experience in the memory buffer of DQN in a quadruple form
    if len(self.memory) > minibatch_size:
      experiences = random.sample(self.memory, k = minibatch_size)
      self.learn(experiences, discount_factor)

  def act(self, state, epsilon = 0.):
    state = preprocess_frame(state).to(self.device)
    self.qnetwork.eval() # Start evaluating and predicting from the batch of experiences
    with torch.no_grad(): # If the loss function(or gradient descent is disabled)
      action_values = self.qnetwork(state)
    self.qnetwork.train() # Training commence; have the agent explore first to guarantee optimal behavior in the long run. So set epsilon(exploration factor to 1 at first)
    if random.random() > epsilon:
      return np.argmax(action_values.cpu().data.numpy())
    else:
      return random.choice(np.arange(self.action_size))

  def learn(self, experiences, discount_factor):
    states, actions, rewards, next_states, dones = zip(*experiences) # Unzip experiences from the buffer

    #convert into PyTorch tensors
    states = torch.from_numpy(np.vstack(states)).float().to(self.device)
    actions = torch.from_numpy(np.vstack(actions)).long().to(self.device)
    rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
    next_states = torch.from_numpy(np.vstack(next_states)).float().to(self.device)
    dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(self.device)
    next_q_targets = self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
    q_targets = rewards + discount_factor * next_q_targets * (1 - dones)
    q_expected = self.qnetwork(states).gather(1, actions)
    loss = F.mse_loss(q_expected, q_targets)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
#Initializing the DCQN agent
agent = Agent(number_actions)

#Training the DCQN agent
number_episodes = 100
maximum_number_timesteps_per_episode = 10000
epsilon_starting_value  = 1.0
epsilon_ending_value  = 0.01
epsilon_decay_value  = 0.995
epsilon = epsilon_starting_value
scores_on_100_episodes = deque(maxlen = 100)
average_episode = []

for episode in range(1, number_episodes + 1):
  state, _ = env.reset()
  score = 0
  for t in range(maximum_number_timesteps_per_episode):
    action = agent.act(state, epsilon)
    next_state, reward, done, _, _ = env.step(action)
    agent.step(state, action, reward, next_state, done)
    state = next_state
    score += reward
    if done:
      break
  scores_on_100_episodes.append(score)
  epsilon = max(epsilon_ending_value, epsilon_decay_value * epsilon)
  print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)), end = "")
  ave = np.mean(scores_on_100_episodes)
  average_episode.append(ave)
  if episode % 100 == 0:
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)))
  if np.mean(scores_on_100_episodes) >= 500.0: # We've set a value of 500 whuch is enough for the agent to win the game while also eating the ghosts
    print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode - 100, np.mean(scores_on_100_episodes)))
    torch.save(agent.qnetwork.state_dict(), 'checkpoint.pth') # Update Q network with some successful paths
    break
#Plot the performance of the training
plt.plot(average_episode, label='Score per Episode')
plt.xlabel('Episodes')
plt.ylabel('Score')
plt.title('Scores for 100 Episodes Epsilon Greedy')
plt.legend()
plt.show()

#Part 3 - Visualizing the results
import glob
import io
import base64
import imageio
from IPython.display import HTML, display
from gym.wrappers.monitoring.video_recorder import VideoRecorder

def show_video_of_model(agent, env_name):
    env = gym.make(env_name, render_mode='rgb_array')
    state, _ = env.reset()
    done = False
    frames = []
    while not done:
        frame = env.render()
        frames.append(frame)
        action = agent.act(state)
        state, reward, done, _, _ = env.step(action)
    env.close()
    imageio.mimsave('video.mp4', frames, fps=30)

show_video_of_model(agent, 'MsPacmanDeterministic-v0')

def show_video():
    mp4list = glob.glob('*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        display(HTML(data='''<video alt="test" autoplay
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")

show_video()
