# Snake RL

This project is a reinforcement learning agent that learns to play the game of Snake.

## Project Structure

- `main.py`: The main script to run the training.
- `requirements.txt`: Python dependencies.
- `snake_rl/`: Source code for the project.
  - `__init__.py`: Makes the `snake_rl` directory a Python package.
  - `agent.py`: Defines the RL agent.
  - `environment.py`: Defines the Snake game environment.
  - `model.py`: Defines the neural network model for the agent.
  - `train.py`: Contains the training loop and logic.
- `utils/`: For helper functions and utilities.

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd snake-rl
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To train the agent, run:

```bash
python main.py
``` 