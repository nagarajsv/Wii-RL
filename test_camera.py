from dolphin import event, savestate, memory, gui, emulation #type: ignore
import time

await event.frameadvance()
savestate.load_from_slot(8)
camera_angle = memory.read_u8(0x91bfdea3)
prev_angle = 3

while True:
    
    gui.draw_text((10, 10), 0xffff0000, f"{camera_angle}")
    if prev_angle == 13 and camera_angle == 3:
        for _ in range(30):
            await event.frameadvance()
        time.sleep(5)
    
    prev_angle = camera_angle
    camera_angle = memory.read_u8(0x91bfdea3)
    await event.frameadvance()
    