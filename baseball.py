from enum import Enum
from dolphin import controller, event, memory # type: ignore

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
BALL_STATE_ADDRESS = 0x91BFEEA5 # byte (1 ball is hit, 0 otherwise)
BATTER_NUMBER_ADDRESS = 0x91BFF0F4 # word
NUMBER_CPU_PITCHES_ADDRESS = 0x91BFEAA0 # word
CAMERA_ANGLE_ADDRESS = 0x91bfdea3 # byte

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
    NOOP = 0
    SWING = 1

class Outcomes(Enum):
    NONE = 0
    STRIKE = 1
    SINGLE = 2
    DOUBLE = 3
    TRIPLE = 4
    HOME_RUN = 5
    BALL = 6
    OUT = 7
    FOUL = 8
    
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

def release_buttons() -> None:
    controller.set_wiimote_buttons(0, BUTTONS)
    
async def pitch(pitch_type: str) -> None:
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

    # time.sleep(0.1)
    controller.set_wiimote_swing(0, 0, -3, 0, 0.5, speed, speed, 1.5708)
    ball_velocity = memory.read_f32(VELOCITY_BALL_ADDRESS)
    while(ball_velocity == 0):
        await event.frameadvance()
        ball_velocity = memory.read_f32(VELOCITY_BALL_ADDRESS)
    pitching_ready = memory.read_u8(PITCHING_READY_ADDRESS)
    while(pitching_ready == 0):
        await event.frameadvance()
        pitching_ready = memory.read_u8(PITCHING_READY_ADDRESS)
        
async def swing() -> None:
    release_buttons()
    controller.set_wiimote_swing(0, 0, -3, 0, 0.5, 40, 40, 1.5708)
    # bat_swinging = memory.read_u8(BAT_SWING_ADDRESS)
    # print(f"beginning loop")
    # while(bat_swinging == 0):
    #     print(f"in loop")
    #     await event.frameadvance()
    #     bat_swinging = memory.read_u8(BAT_SWING_ADDRESS)
    # print(f"done")
    for _ in range(5):
        await event.frameadvance()