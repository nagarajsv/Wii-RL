import numpy as np
from dolphin import event, memory, savestate, emulation  # type: ignore
import gymnasium as gym
import baseball
from PIL import Image
import cv2

class BattingEnv():
    
    def __init__(self, x_window: int = 84, y_window: int = 84, frame_skip: int = 4, frame_stack: int = 4) -> None:
        self.action_space = gym.spaces.Discrete(len(baseball.BattingActions))
        
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        
        self.x_window = x_window
        self.y_window = y_window
        
        self.observation_space = gym.spaces.Box(
            low = 0,
            high = 255,
            shape = (self.frame_stack, x_window, y_window),
            dtype=np.uint8
        )
        
        self.frames = np.zeros((frame_stack, x_window, y_window), dtype=np.uint8)
        
        self.prev_swinging = False
        self.prev_ball_contact = False
        self.steps = 0
                
    async def reset(self) -> tuple[np.ndarray, dict]:
        await event.framedrawn()
        emulation.reset()
        await event.framedrawn()
        rand = np.random.randint(1)
        # if rand == 4:
        #     rand = 5
        savestate.load_from_slot(rand + 3)
        print(f"LOADING STATE {rand + 3}")
        memory.write_u32(baseball.OUTS_ADDRESS, 0)
        memory.write_u32(baseball.OUR_SCORE_ADDRESS, 0)
        memory.write_u32(baseball.STRIKES_ADDRESS, 0)
        
        info = {}
        width, height, data = await event.framedrawn() 
        self.frames = np.zeros((self.frame_stack, self.x_window, self.y_window), dtype=np.uint8)
        self.send_new_frame(data, width, height)
        # self.send_new_frame(data)
        
        return self.frames, info
    
    async def test_state(self) -> tuple[np.ndarray, dict]:
        await event.framedrawn()
        savestate.load_from_slot(3)
        memory.write_u32(baseball.OUTS_ADDRESS, 0)
        memory.write_u32(baseball.OUR_SCORE_ADDRESS, 0)
        memory.write_u32(baseball.STRIKES_ADDRESS, 0)
        
        info = {}
        width, height, data = await event.framedrawn() 
        self.frames = np.zeros((self.frame_stack, self.x_window, self.y_window), dtype=np.uint8)
        self.send_new_frame(data, width, height)
        # self.send_new_frame(data)
        
        return self.frames, info
    
    async def step(self, action:int) -> tuple[np.ndarray, float, bool, bool, dict]:
        info = {}
        
        self.prev_outs = memory.read_u32(baseball.OUTS_ADDRESS)
        self.prev_strikes = memory.read_u32(baseball.STRIKES_ADDRESS)
        self.prev_balls = memory.read_u32(baseball.BALLS_ADDRESS)
        self.prev_bases_ran = memory.read_u32(baseball.OUR_NUMBER_BASES_ADDRESS)
        num_pitches_thrown = memory.read_u32(baseball.NUMBER_CPU_PITCHES_ADDRESS)
        bat_swinging = memory.read_u8(baseball.BAT_SWING_ADDRESS)
        ball_state = memory.read_u8(baseball.BALL_STATE_ADDRESS)
        self.prev_ball_contact = True if ball_state == 1 else False
        self.prev_swinging = True if bat_swinging == 1 else False
        ball_hit = False
        
        # print(f"sending action")
        # print(f"action: {action}")
        # action = 0
        if(action == 1 and bat_swinging == 0):
            await baseball.swing()
            bat_swinging = memory.read_u8(baseball.BAT_SWING_ADDRESS)
            # print(f"initla swing check")
            # while bat_swinging == 0:
            #     # print(f"swinging")
            #     await event.frameadvance()
            #     bat_swinging = memory.read_u8(baseball.BAT_SWING_ADDRESS)
            # print(f"swing check")
            while bat_swinging == 1:
                # print(f"swinging")
                await event.framedrawn()
                bat_swinging = memory.read_u8(baseball.BAT_SWING_ADDRESS)
                if memory.read_u8(baseball.BALL_STATE_ADDRESS) == 1:
                    ball_hit = True
                    # print(f"ball hit")
            # print(f"swing checks done")
        else:
            for _ in range(self.frame_skip):
                await event.framedrawn()
            
        camera_angle = memory.read_u8(baseball.CAMERA_ANGLE_ADDRESS)
        prev_angle = 3
        # print(f"camera angle check")
        while(camera_angle != 3):
            self.prev_swinging = 0
            await event.framedrawn()
            prev_angle = camera_angle
            camera_angle = memory.read_u8(baseball.CAMERA_ANGLE_ADDRESS)
        if prev_angle == 13:
            for _ in range(30):
                await event.framedrawn()
        # print(f"camera angle check done")
            
        # print(f"event.frame draw")
        ###########
        width, height, data = await event.framedrawn()
        # 
        # self.send_empty_frame()
        # await event.framedrawn()         
        # print(f"event.frame draw done")
        self.send_new_frame(data, width, height)
        # print(f"send new frame done")
        
        reward = self.calculate_reward(action, ball_hit)
        # print(f"reward calculated")
        self.steps += 1
        
        return self.get_obs(), reward, num_pitches_thrown > 10, False, info
        
    def calculate_reward(self, action, ball_hit: False) ->float:
        outs = memory.read_u32(baseball.OUTS_ADDRESS)
        strikes = memory.read_u32(baseball.STRIKES_ADDRESS)
        balls = memory.read_u32(baseball.BALLS_ADDRESS)
        bases_ran = memory.read_u32(baseball.OUR_NUMBER_BASES_ADDRESS)
        difference = bases_ran - self.prev_bases_ran
        side = memory.read_u8(baseball.SIDE_ADDRESS)
        swinging = memory.read_u8(baseball.BAT_SWING_ADDRESS)
        reward = 0
        # if action == 1 and not ball_hit:
        #     reward -= 1
        # elif action == 1 and ball_hit:
        #     reward += 1
            # print(f"swing miss")
            
        if action == 1 and ball_hit:
            reward += (1 + difference)  # Good hit
        elif action == 1 and not ball_hit:
            reward += -0.01  # Missed swing (single penalty)
        elif strikes - self.prev_strikes == 1 and action != 1:
            reward += -0.02
        # elif balls - self.prev_balls == 1 and action != 1:
        #     reward += 0.02
            
        # if action == 1 and not ball_hit:
        #     reward += -0.05
            
        # if strikes - self.prev_strikes == 1:
        #     reward += -0.05
        # elif action == 1 and ball_hit:
        #     reward += (10 + (difference * 0.25))
        # elif balls - self.prev_balls == 1:
        #     reward += 0.2
            
        # if(difference == 4):
        #     # home run or out of the park
        #     reward += 1
        # elif(difference == 3):
        #     # triple
        #     reward += 0.75
        # elif(difference == 2):
        #     # double
        #     reward += 0.625
        # elif(difference == 1):
        #     # single
        #     reward += 0.5
        # elif(balls - self.prev_balls == 1):
        #     # ball
        #     reward += 0.2
        # elif(strikes - self.prev_strikes == 1 and strikes <= 2):
        #     # strike
        #     # print(f"STRIKE STRIKE STRIKE")
        #     reward += -0.5
        # elif(outs - self.prev_outs == 1 or side == 0):
        #     # out
        #     reward += -0.5
        # else:
        #     reward += 0
            
        return reward
        
    def preprocess_img(self, width, height, data: bytes) -> np.ndarray:
        img = Image.frombytes("RGB", (width, height), data)
        # img.save(f"C:\\Users\\nagar\\dolphin-training\\python-stubs\\img\\img{self.steps}_color.png")
        # print(f"image craeted")
        crop_left = 316
        crop_size = 317 #456
        crop_right = crop_left + crop_size
        # img = self.black_and_white(img)
        img = img.convert("L")
        
        img = img.crop((crop_left, height-crop_size, crop_right, height))
        
        # print(f"img cropped")
        img = img.resize((self.x_window, self.y_window))
        # print(f"img resized")
        # img = img.convert("L", dither=None)
        
        # print(f"img grayscaled")
        # img.save(f"C:\\Users\\nagar\\dolphin-training\\python-stubs\\img\\img{self.steps}.png")
        return np.array(img, dtype=np.uint8)
    
    def black_and_white(self, img: Image) -> Image:
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Mask bright white-ish regions
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 50, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white).astype(np.float32) / 255.0

        # Stronger boost for masked regions: scale [0.5, 2.0]
        enhanced = gray * (0.15 + 2.0 * mask)

        # Apply gamma correction to increase contrast
        gamma = 1.8  # higher = more aggressive contrast
        enhanced = 255 * ((enhanced / 255) ** gamma)

        # Clip and convert back
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

        return Image.fromarray(enhanced, mode='L')
    
    def get_obs(self) -> np.ndarray:
        return self.frames
    
    def send_new_frame(self, data: bytes, width: int, height: int):
        img = self.preprocess_img(width, height, data)
        
        # print(f"preprocess img done")
        self.frames[:-1] = self.frames[1:]
        self.frames[-1] = img
        # self.frames.append(img)
        
    def send_empty_frame(self) -> None:
        img = np.zeros((1, self.x_window, self.y_window), dtype=np.uint8)
        self.frames[:-1] = self.frames[1:]
        self.frames[-1] = img