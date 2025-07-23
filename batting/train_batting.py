import sys
sys.path.append("C:\\Users\\nagar\\.pyenv\\pyenv-win\\versions\\3.12.10\\Lib\\site-packages")
sys.path.append("C:\\Users\\nagar\\dolphin-training\\python-stubs")

from dolphin import event # type: ignore
from batting_env import BattingEnv
from batting_agent import BattingAgent

await event.frameadvance()

env = BattingEnv()

agent = BattingAgent(env, state_dim=(4, 84, 84), action_dim=2, model_name="batting-3")

await agent.learn()

print(f"complete")