import sys
sys.path.append("C:\\Users\\nagar\\.pyenv\\pyenv-win\\versions\\3.12.10\\Lib\\site-packages")
sys.path.append("C:\\Users\\nagar\\dolphin-training\\python-stubs")

import locale
locale.setlocale(locale.LC_ALL, 'en_US')

import gymnasium as gym
import numpy as np
from dolphin import event, savestate, memory, controller, emulation
import time
import random
from enum import Enum
from networks.dqn import DQN
from networks.replay_memory import ReplayMemory, Transition
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.optim as optim
import math
from itertools import count

class PitchingActions(Enum):
    SLOW_OVERHAND_LEFT_FASTBALL = 0
    SLOW_OVERHAND_CENTER_FASTBALL = 1
    SLOW_OVERHAND_RIGHT_FASTBALL = 2
    SLOW_UNDERHAND_LEFT_FASTBALL = 3
    SLOW_UNDERHAND_CENTER_FASTBALL = 4
    SLOW_UNDERHAND_RIGHT_FASTBALL = 5
    FAST_OVERHAND_LEFT_FASTBALL = 6
    FAST_OVERHAND_CENTER_FASTBALL = 7
    FAST_OVERHAND_RIGHT_FASTBALL = 8
    FAST_UNDERHAND_LEFT_FASTBALL = 9
    FAST_UNDERHAND_CENTER_FASTBALL = 10
    FAST_UNDERHAND_RIGHT_FASTBALL = 11
    SLOW_OVERHAND_LEFT_CURVEBALL = 12
    SLOW_OVERHAND_CENTER_CURVEBALL = 13
    SLOW_OVERHAND_RIGHT_CURVEBALL = 14
    SLOW_UNDERHAND_LEFT_CURVEBALL = 15
    SLOW_UNDERHAND_CENTER_CURVEBALL = 16
    SLOW_UNDERHAND_RIGHT_CURVEBALL = 17
    FAST_OVERHAND_LEFT_CURVEBALL = 18
    FAST_OVERHAND_CENTER_CURVEBALL = 19
    FAST_OVERHAND_RIGHT_CURVEBALL = 20
    FAST_UNDERHAND_LEFT_CURVEBALL = 21
    FAST_UNDERHAND_CENTER_CURVEBALL = 22
    FAST_UNDERHAND_RIGHT_CURVEBALL = 23
    SLOW_OVERHAND_LEFT_SCREWBALL = 24
    SLOW_OVERHAND_CENTER_SCREWBALL = 25
    SLOW_OVERHAND_RIGHT_SCREWBALL = 26
    SLOW_UNDERHAND_LEFT_SCREWBALL = 27
    SLOW_UNDERHAND_CENTER_SCREWBALL = 28
    SLOW_UNDERHAND_RIGHT_SCREWBALL = 29
    FAST_OVERHAND_LEFT_SCREWBALL = 30
    FAST_OVERHAND_CENTER_SCREWBALL = 31
    FAST_OVERHAND_RIGHT_SCREWBALL = 32
    FAST_UNDERHAND_LEFT_SCREWBALL = 33
    FAST_UNDERHAND_CENTER_SCREWBALL = 34
    FAST_UNDERHAND_RIGHT_SCREWBALL = 35
    SLOW_OVERHAND_LEFT_SPLITTER = 36
    SLOW_OVERHAND_CENTER_SPLITTER = 37
    SLOW_OVERHAND_RIGHT_SPLITTER = 38
    SLOW_UNDERHAND_LEFT_SPLITTER = 39
    SLOW_UNDERHAND_CENTER_SPLITTER = 40
    SLOW_UNDERHAND_RIGHT_SPLITTER = 41
    FAST_OVERHAND_LEFT_SPLITTER = 42
    FAST_OVERHAND_CENTER_SPLITTER = 43
    FAST_OVERHAND_RIGHT_SPLITTER = 44
    FAST_UNDERHAND_LEFT_SPLITTER = 45
    FAST_UNDERHAND_CENTER_SPLITTER = 46
    FAST_UNDERHAND_RIGHT_SPLITTER = 47

class BattingActions(Enum):
    NONE = 0
    SWING = 1

class PitchingOutcomes(Enum):
    STRIKE = 0
    SINGLE = 1
    DOUBLE = 2
    TRIPLE = 3
    HOME_RUN = 4
    BALL = 5
    OUT = 6
    FOUL = 7
    
BUTTONS = {
    "A": False,
    "B": False,
    "Left": False,
    "Right": False,
    "Up": False,
    "Down": False,
    "One": False,
    "Two": False,
    "Plus": False,
    "Minus": False,
    "Home": False
}

PITCHING_TYPES = [
    "slow_overhand_left_fastball",
    "slow_overhand_center_fastball",
    "slow_overhand_right_fastball",
    "slow_underhand_left_fastball",
    "slow_underhand_center_fastball",
    "slow_underhand_right_fastball",
    "fast_overhand_left_fastball",
    "fast_overhand_center_fastball",
    "fast_overhand_right_fastball",
    "fast_underhand_left_fastball",
    "fast_underhand_center_fastball",
    "fast_underhand_right_fastball",
    "slow_overhand_left_curveball",
    "slow_overhand_center_curveball",
    "slow_overhand_right_curveball",
    "slow_underhand_left_curveball",
    "slow_underhand_center_curveball",
    "slow_underhand_right_curveball",
    "fast_overhand_left_curveball",
    "fast_overhand_center_curveball",
    "fast_overhand_right_curveball",
    "fast_underhand_left_curveball",
    "fast_underhand_center_curveball",
    "fast_underhand_right_curveball",
    "slow_overhand_left_screwball",
    "slow_overhand_center_screwball",
    "slow_overhand_right_screwball",
    "slow_underhand_left_screwball",
    "slow_underhand_center_screwball",
    "slow_underhand_right_screwball",
    "fast_overhand_left_screwball",
    "fast_overhand_center_screwball",
    "fast_overhand_right_screwball",
    "fast_underhand_left_screwball",
    "fast_underhand_center_screwball",
    "fast_underhand_right_screwball",
    "slow_overhand_left_splitter",
    "slow_overhand_center_splitter",
    "slow_overhand_right_splitter",
    "slow_underhand_left_splitter",
    "slow_underhand_center_splitter",
    "slow_underhand_right_splitter",
    "fast_overhand_left_splitter",
    "fast_overhand_center_splitter",
    "fast_overhand_right_splitter",
    "fast_underhand_left_splitter",
    "fast_underhand_center_splitter",
    "fast_underhand_right_splitter"
]

def release_buttons():
    controller.set_wiimote_buttons(0, BUTTONS)
    
def pitch(pitch_type: str):
    release_buttons()
    split = pitch_type.split("_")

    speed = 16 # slow speed

    match(split[0]):
        case "fast":
            speed = 75

    match(split[1]):
        case "overhand":
            controller.set_wiimote_buttons(0, {"One": True})
        case "underhand":
            controller.set_wiimote_buttons(0, {"Two": True})

    match(split[2]):
        case "right":
            controller.set_wiimote_buttons(0, {"Right": True})
        case "left": 
            controller.set_wiimote_buttons(0, {"Left": True})
        # center is default, no button pressed

    match(split[3]):
        case "splitter":
            controller.set_wiimote_buttons(0, {"A": True, "B": True})
        case "screwball":
            controller.set_wiimote_buttons(0, {"A": True})
        case "curveball":
            controller.set_wiimote_buttons(0, {"B": True})
        # fastball is default, no button pressed

    time.sleep(0.1)
    controller.set_wiimote_swing(0, 0, -3, 0, 0.5, speed, speed, 1.5708)
    
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
    
episode_durations = []
    
def optimize_model():
    if len(rmemory) < BATCH_SIZE:
        return
    transitions = rmemory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = torch.nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

STRIKES_ADDRESS = 0x91BFE784 # word
BALLS_ADDRESS = 0x91BFE788 # word
OUTS_ADDRESS = 0x91BFE78C # word
BAT_SWING_ADDRESS = 0x803d08FF # byte
SIDE_ADDRESS = 0x91BFEA67 # (1 batting, 0 pitching), byte
OUR_SCORE_ADDRESS = 0x91BFE794 # word
ENEMY_SCORE_ADDRESS = 0x91BFE7BC # word
INNING_NUMBER_ADDRESS = 0x91BFE7E0 # word
OUR_NUMBER_BASES_ADDRESS = 0x91BFE75C # word
CPU_NUMBER_BASES_ADDRESS = 0x91BFE760 # word
PITCHING_READY_ADDRESS = 0x91BFEA08 # (1 not ready, 0 ready), byte
BATTING_READY_ADDRESS = 0x91BFEEA5 # (1 not ready, 0 ready), byte
BATTER_HANDEDNESS_ADDRESS = 0x804A6FB0 # (0 right, 128 left), byte
VELOCITY_BALL_ADDRESS = 0x9207EAE4 # float (0 when batting/pitching ready)

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

class BaseballPitchingEnv():
    
    def __init__(self):

        self.action_space = gym.spaces.Discrete(len(PitchingActions))

        # [strikes, balls, outs, batter_handedness, last_pitch, second_last_pitch, last_pitch_outcome, second_last_pitch_outcome]
        self.observation_space = gym.spaces.Box(
            low = 0,
            high = np.array([3, 4, 3, 1, len(PitchingActions) - 1, len(PitchingActions) - 1, len(PitchingOutcomes) - 1, len(PitchingOutcomes) -1]),
            dtype=np.int32
        )
        
        self.state = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        
        self.last_outcome = 0
        self.second_last_outcome = 0
        self.last_pitch = 0
        self.second_last_pitch = 0

    def reset(self):
        # savestate.load_from_file("sakura_pitching.sav")
        # savestate.load_from_slot(1)
        
        time.sleep(5)
        
        self.last_pitch = 0
        self.second_last_pitch = 0
        self.last_outcome = 0
        self.second_last_outcome = 0
        
        obs = np.array([0, 0, 0, 1 if memory.read_u8(BATTER_HANDEDNESS_ADDRESS) == 128 else 0, 0, 0, 0, 0])
        
        info = {}

        return obs, info

    def step(self, action):
        self.prev_cpu_bases_ran = memory.read_u32(CPU_NUMBER_BASES_ADDRESS)
        self.prev_outs = memory.read_u32(OUTS_ADDRESS)
        self.prev_strikes = memory.read_u32(STRIKES_ADDRESS)
        self.prev_balls = memory.read_u32(BALLS_ADDRESS)
        self.prev_score = memory.read_u32(ENEMY_SCORE_ADDRESS)
        pitch(PITCHING_TYPES[action])
        self.second_last_pitch = self.last_pitch
        self.last_pitch = action
    
    def calculate_reward(self) -> tuple[float, int, bool]:
        cur_bases_ran = memory.read_u32(CPU_NUMBER_BASES_ADDRESS)
        cur_outs = memory.read_u32(OUTS_ADDRESS)
        cur_strikes = memory.read_u32(STRIKES_ADDRESS)
        cur_balls = memory.read_u32(BALLS_ADDRESS)
        side = memory.read_u8(SIDE_ADDRESS)
        cur_score = memory.read_u32(ENEMY_SCORE_ADDRESS)
        difference = cur_bases_ran - self.prev_cpu_bases_ran

        cpu_score = memory.read_u32(ENEMY_SCORE_ADDRESS)

        reward = self.prev_score - cur_score
        
        self.second_last_outcome = self.last_outcome
        
        memory.write_u32(OUTS_ADDRESS, 0)

        if(side == 1 or cur_outs - self.prev_outs == 1):
            self.last_outcome = PitchingOutcomes.OUT.value
            # side switched, meaning an out was thrown
            return 1, self.last_outcome, cpu_score == 99
        elif(difference == 4):
            self.last_outcome = PitchingOutcomes.HOME_RUN.value
            # home run or out of the park
            return -1, self.last_outcome, cpu_score == 99
        elif(difference == 3):
            self.last_outcome = PitchingOutcomes.TRIPLE.value
            # triple
            return -0.75, self.last_outcome, cpu_score == 99
        elif(difference == 2):
            self.last_outcome = PitchingOutcomes.DOUBLE.value
            # double
            return -0.5, self.last_outcome, cpu_score == 99
        elif(difference == 1):
            self.last_outcome = PitchingOutcomes.SINGLE.value
            # single or walk
            return -0.25, self.last_outcome, cpu_score == 99
        elif(cur_strikes - self.prev_strikes == 1 and cur_strikes <= 2):
            self.last_outcome = PitchingOutcomes.STRIKE.value
            return 0.5, self.last_outcome, cpu_score == 99
        elif(cur_balls - self.prev_balls == 1):
            self.last_outcome = PitchingOutcomes.BALL.value
            return -0.15, self.last_outcome, cpu_score == 99
        else:
            # probably a foul but strikes are already at 3
            self.last_outcome = PitchingOutcomes.FOUL.value
            return 0, self.last_outcome, cpu_score == 99
        
    def get_obs(self) -> np.ndarray:
        return np.array([
            memory.read_u32(STRIKES_ADDRESS),
            memory.read_u32(BALLS_ADDRESS),
            memory.read_u32(OUTS_ADDRESS),
            1 if memory.read_u8(BATTER_HANDEDNESS_ADDRESS) == 128 else 0,
            self.last_pitch,
            self.second_last_pitch,
            self.last_outcome,
            self.second_last_outcome
        ])
    
env = BaseballPitchingEnv()
    
policy_net = DQN(8, env.action_space.n).to(device)
target_net = DQN(8, env.action_space.n).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

rmemory = ReplayMemory(10000)

steps_done = 0

total_reward = 0

if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 600
else:
    num_episodes = 50

for i_episode in range(num_episodes):
    for _ in range(4):   
        await event.frameadvance()
    savestate.load_from_file("sakura_pitching.sav")
    savestate.load_from_slot(1)
    for _ in range(4):
        await event.frameadvance()
    # Initialize the environment and get its state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = select_action(state)
        print(f"action: {action}")
        env.step(action.item())
        for _ in range(10):
            await event.frameadvance()
        
        ball_velocity = memory.read_f32(VELOCITY_BALL_ADDRESS)
        while(ball_velocity != 0.0):
            # print(f"ball velocity: ", ball_velocity)
            await event.frameadvance()
            ball_velocity = memory.read_f32(VELOCITY_BALL_ADDRESS)
            
        pitching_ready = memory.read_u8(PITCHING_READY_ADDRESS)
        while(pitching_ready != 0):
            # print(f"pitchiing ready :", pitching_ready)
            await event.frameadvance()
            pitching_ready = memory.read_u8(PITCHING_READY_ADDRESS)
            
        reward, outcome, done = env.calculate_reward()
        total_reward += reward
        print(f"total reward: {total_reward}")
        observation = env.get_obs()
        reward = torch.tensor([reward], device=device)

        if(done):
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        rmemory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            torch.save(policy_net.state_dict(), f"pitching_model_episode_{i_episode}.pth")
            print(f"saved episode {i_episode}")
            break

print(f'Complete')