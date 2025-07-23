import sys
sys.path.append("C:\\Users\\nagar\\.pyenv\\pyenv-win\\versions\\3.12.10\\Lib\\site-packages")
sys.path.append("C:\\Users\\nagar\\dolphin-training\\python-stubs")

from dolphin import event
from pitching_env import PitchingEnv
from pitching_agent import Agent

await event.frameadvance()

env = PitchingEnv()

agent = Agent(env, n_obs=8, n_act=env.action_space.n, model_name="test")

agent.load_model("C:\\Users\\nagar\\dolphin-training\\python-stubs\\models\\pitching\\pitchingv2_699.pth")

await agent.play()
# await agent.learn()
print(f'Complete')