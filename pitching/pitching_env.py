import numpy as np
from dolphin import event, memory, savestate # type: ignore
import gymnasium as gym
import baseball

class PitchingEnv():
    
    def __init__(self):

        self.action_space = gym.spaces.Discrete(len(baseball.PitchingActions))

        # [strikes, balls, outs, batter_handedness, last_pitch, second_last_pitch, last_pitch_outcome, second_last_pitch_outcome]
        self.observation_space = gym.spaces.Box(
            low = 0,
            high = np.array([3, 
                             4, 
                             3, 
                             1, 
                             len(baseball.PitchingActions) - 1, 
                             len(baseball.PitchingActions) - 1, 
                             len(baseball.Outcomes) - 1, 
                             len(baseball.Outcomes) -1]),
            dtype=np.int32
        )
        
        self.state = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        
        self.last_outcome = 0
        self.second_last_outcome = 0
        self.last_pitch = 0
        self.second_last_pitch = 0
        self.outs = 0

    async def reset(self) -> tuple[np.ndarray, dict]:
        await event.frameadvance()
        savestate.load_from_file("sakura_pitching.sav")
        savestate.load_from_slot(1)
        await event.frameadvance()
        
        self.last_pitch = 0
        self.second_last_pitch = 0
        self.last_outcome = 0
        self.second_last_outcome = 0
        
        obs = np.array([0, 0, 0, 1 if memory.read_u8(baseball.BATTER_HANDEDNESS_ADDRESS) == 128 else 0, 0, 0, 0, 0])
        
        info = {}

        return obs, info
    
    async def skip_frames(self) -> None:
        ball_velocity = memory.read_f32(baseball.VELOCITY_BALL_ADDRESS)
        while(ball_velocity != 0.0):
            # wait for the ball to be thrown
            await event.frameadvance()
            ball_velocity = memory.read_f32(baseball.VELOCITY_BALL_ADDRESS)
        pitching_ready = memory.read_u8(baseball.PITCHING_READY_ADDRESS)
        while(pitching_ready != 0):
            # wait for the game to be ready for the next pitch
            await event.frameadvance()
            pitching_ready = memory.read_u8(baseball.PITCHING_READY_ADDRESS)

    async def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.prev_cpu_bases_ran = memory.read_u32(baseball.CPU_NUMBER_BASES_ADDRESS)
        self.prev_outs = memory.read_u32(baseball.OUTS_ADDRESS)
        self.prev_strikes = memory.read_u32(baseball.STRIKES_ADDRESS)
        self.prev_balls = memory.read_u32(baseball.BALLS_ADDRESS)
        self.prev_score = memory.read_u32(baseball.ENEMY_SCORE_ADDRESS)
        await baseball.pitch(baseball.PITCHING_TYPES[action])
        # wait for game to be ready for next pitch
        await self.skip_frames()
        
        self.second_last_pitch = self.last_pitch
        self.last_pitch = action
        reward, outcome, done = self.calculate_reward()
        if done:
            self.outs = 0
        self.second_last_outcome = self.last_outcome
        self.last_outcome = outcome
        
        info = {}
        
        return self.get_obs(), reward, done, False, info
    
    def calculate_reward(self) -> tuple[float, int, bool]:
        cur_bases_ran = memory.read_u32(baseball.CPU_NUMBER_BASES_ADDRESS)
        cur_outs = memory.read_u32(baseball.OUTS_ADDRESS)
        cur_strikes = memory.read_u32(baseball.STRIKES_ADDRESS)
        cur_balls = memory.read_u32(baseball.BALLS_ADDRESS)
        side = memory.read_u8(baseball.SIDE_ADDRESS)
        cur_score = memory.read_u32(baseball.ENEMY_SCORE_ADDRESS)
        difference = cur_bases_ran - self.prev_cpu_bases_ran

        cpu_score = memory.read_u32(baseball.ENEMY_SCORE_ADDRESS)

        if(side == 1 or cur_outs - self.prev_outs == 1):
            self.outs += 1
            # side switched, meaning an out was thrown
            return 0.75, baseball.Outcomes.OUT.value, self.outs == 9
        elif(difference == 4):
            self.outs += 1
            # home run or out of the park
            return -1, baseball.Outcomes.HOME_RUN.value, self.outs == 9
        elif(difference == 3):
            self.outs += 1
            # triple
            return -0.75, baseball.Outcomes.TRIPLE.value, self.outs == 9
        elif(difference == 2):
            self.outs += 1
            # double
            return -0.5, baseball.Outcomes.DOUBLE.value, self.outs == 9
        elif(difference == 1):
            self.outs += 1
            # single or walk
            return -0.25, baseball.Outcomes.SINGLE.value, self.outs == 9
        elif(cur_strikes - self.prev_strikes == 1 and cur_strikes <= 2):
            return 1, baseball.Outcomes.STRIKE.value, self.outs == 9
        elif(cur_balls - self.prev_balls == 1):
            return -0.15, baseball.Outcomes.BALL.value, self.outs == 9
        else:
            # probably a foul but strikes are already at 3
            return 0, baseball.Outcomes.FOUL.value, self.outs == 9
        
    def get_obs(self) -> np.ndarray:
        return np.array([
            memory.read_u32(baseball.STRIKES_ADDRESS),
            memory.read_u32(baseball.BALLS_ADDRESS),
            memory.read_u32(baseball.OUTS_ADDRESS),
            1 if memory.read_u8(baseball.BATTER_HANDEDNESS_ADDRESS) == 128 else 0,
            self.last_pitch,
            self.second_last_pitch,
            self.last_outcome,
            self.second_last_outcome
        ])