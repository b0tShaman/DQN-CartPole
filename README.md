# Deep Q-Network (DQN) Implementation in Java

This repository contains an implementation of a **Deep Q-Network (DQN)** algorithm for solving the CartPole problem. Additionally, it includes a visualization of the CartPole environment created using **JPanel** for animation.

---

## Features
- Implementation of the DQN algorithm in Java.
- Training a neural network to balance a pole on a cart.
- **JPanel-based animation** to visualize the CartPole in action.

---

## Prerequisites

Before running the project, ensure you have the following installed:

1. **Java Development Kit (JDK)** (Version 8 or higher)
2. **Maven** (for managing dependencies, optional but recommended)
3. A compatible **IDE** (e.g., IntelliJ IDEA, Eclipse, or VS Code with Java extensions).

---

## How to Run

### Clone the Repository
```bash
git clone https://github.com/b0tShaman/DQN-CartPole.git
cd DQN-CartPole
```

### Compile and Run

#### Option 1: Using an IDE
1. Open the project in your preferred IDE.
2. Build the project to resolve dependencies.
3. Run the `DQN.java` class to train and visualize the CartPole animation.

#### Option 2: Using the Command Line
1. Navigate to the project root.
2. Compile the Java files:
   ```bash
   javac -d bin src/**/*.java
   ```
3. Run the main class:
   ```bash
   java -cp bin DQN.DQN
   ```

---

## File Structure
```
DQN-CartPole/
├── src/
│   ├── Common/
│   │   ├── InputNeuron.java            # Represents an input neuron in the neural network
│   │   ├── Matrix.java                 # Utility for matrix operations
│   │   ├── Neuron.java                 # Core neuron implementation
│   │   ├── NeuronBase.java             # Base class for neurons
│   ├── DQN/
│       ├── CartPole.java               # JPanel-based animation for CartPole
│       ├── DQN.java                    # Entry point of the application, DQN implementation
│       ├── Environment_CartPole.java   # CartPole simulation environment
│       ├── StateSpaceQuantization.java # Handles state space quantization for DQN
├── README.md                           # Project documentation
└── .gitignore                          # Ignored files (e.g., build outputs, IDE settings)
```

---

## Visualization

The CartPole animation uses Java's **Swing** and **JPanel** to render the environment. After training, the policy learned by the DQN is applied to control the CartPole in real-time.

---

## How It Works

### DQN Overview
- The **DQN algorithm** is a reinforcement learning method that uses a neural network to approximate the Q-value function.
- The agent observes the state of the environment, selects actions, and receives rewards to learn an optimal policy.

### Key Components
1. **State**: The current position and velocity of the cart and the angle and angular velocity of the pole.
2. **Action**: The force applied to the cart (left or right).
3. **Reward**: Positive reward for keeping the pole balanced and negative reward when it falls.
4. **Neural Network**: Approximates the Q-values for each state-action pair.

---

## Key Variables

Here are the main configurable variables in the code (DQN.java) that control the behavior of the simulation and the training process:

- `train` (boolean):  
  Set this to `true` to enable training. Once the training is complete, set it to `false` to stop training and begin using the trained model.

- `Length_Of_Stick` (double):  
  The length of the pole in meters. Default value is `0.326`.

- `Mass_Of_Cart` (double):  
  The mass of the cart in kilograms. Default value is `0.711`.

- `Mass_Of_Stick` (double):  
  The mass of the pole in kilograms. Default value is `0.209`.

- `g` (double):  
  The acceleration due to gravity (m/s²). Default value is `9.8`.

- `initial_angle_of_stick` (double):  
  The initial angle the pole makes with the cart in radians. A value of `0` means the pole is upright (balanced)

- `fps` (int):  
  Frames per second for the pendulum animation. Default value is `10`.

- `M` (int):  
  The number of times the agent interacts with the environment to create the `trainingSet.csv` file. Default value is `100`.

- `maxEpisodeRewardRequired` (int):  
  The reward threshold that, when achieved, will stop the training and save the neural network weights. Default value is `0`.

- `EPISODES` (int):  
  The number of data points in `trainingSet.csv` is calculated as `EPISODES / 0.02`. Default value is `30`.

- `discount` (double):  
  The discount factor for future rewards. Default value is `0.99`.

- `Filepath` (string):  
  The path to store Training Data and Neural Network weights.


