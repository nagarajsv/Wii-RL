import sys
sys.path.append("C:\\Users\\nagar\\.pyenv\\pyenv-win\\versions\\3.12.10\\Lib\\site-packages")
sys.path.append("C:\\Users\\nagar\\dolphin-training\\python-stubs")

from dolphin import event #type: ignore
from batting_env import BattingEnv
from rainbow_batting_agent import Agent

await event.frameadvance()

env = BattingEnv()

agent = Agent(
            env=env,
            state_dim=(4, 84, 84),
            action_dim=2,
            model_name="rainbow_batting_agent",
            n_atoms=51,          # Number of atoms in value distribution
            v_min=-10,           # Minimum value
            v_max=10,            # Maximum value
            n_steps=3,           # Multi-step returns
            alpha=0.6,           # Prioritized replay alpha
            beta=0.4,            # Prioritized replay beta
            beta_increment=0.001, # Beta increment per step
            gamma=0.99,          # Discount factor
            lr=0.00025,          # Learning rate
            batch_size=32
        )

agent.load_model("C:\\Users\\nagar\\dolphin-training\\python-stubs\\models\\batting\\rainbow_batting_agent_rainbow_ep2350.pth")

await agent.play()