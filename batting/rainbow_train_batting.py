import sys
sys.path.append("C:\\Users\\nagar\\.pyenv\\pyenv-win\\versions\\3.12.10\\Lib\\site-packages")
sys.path.append("C:\\Users\\nagar\\dolphin-training\\python-stubs")
import locale
locale.setlocale(locale.LC_ALL, 'en_US')

from dolphin import event # type: ignore
from batting_env import BattingEnv
from rainbow_agent_log import Agent

await event.frameadvance()

env = BattingEnv(frame_skip=2)

agent = Agent(
            env=env,
            state_dim=(4, 84, 84),
            action_dim=2,
            model_name="rainbow_batting_agent",
            eps_steps=30_000,
            total_steps=50_000
        )

await agent.learn()