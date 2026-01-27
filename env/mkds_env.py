import numpy as np
import os
from desmume.emulator import DeSmuME
from desmume.controls import keymask, Keys
from PIL import Image
from src.utils import config

class MKDSEnv:
    def __init__(self):
        # Initializing emulator and SDL window for visualization
        self.emu = DeSmuME()
        self.emu.open(config.ROM_PATH)
        self.window = self.emu.create_sdl_window()
        
        # discrete RL actions mapped to emulator keymasks
        self.action_map = self._setup_actions()
        
        # Checkpoint angles from Lua script for Figure-8 Circuit
        self.chkpnt_ang = [
            16380, 16380, 20805, 24812, 28650, -30861, -23773, -17975, -15386, -11925,
            -8233, -8233, -8233, -8233, -8233, -8233, -14276, -18273, -27352, 29514, 
            26495, 18165, 16380, 16380, 16380, 16380
        ]

        # Episode tracking
        self.prev_checkpoint = 0
        self.prev_lap = 0

    def _setup_actions(self):
        """Standard 6-action space for solo Time Trials."""
        ACCEL = keymask(Keys.KEY_A)  # X button
        LEFT = keymask(Keys.KEY_LEFT)
        RIGHT = keymask(Keys.KEY_RIGHT)
        DRIFT = keymask(Keys.KEY_R)  # W key in your original config
        
        return {
            0: [ACCEL],                         # Straight
            1: [ACCEL, LEFT],                   # Left
            2: [ACCEL, RIGHT],                  # Right
            3: [ACCEL, DRIFT],                  # Drift Straight
            4: [ACCEL, DRIFT, LEFT],            # Drift Left
            5: [ACCEL, DRIFT, RIGHT]            # Drift Right
        }

    def _get_state(self):
        """Captures and processes the top screen into 84x84 grayscale."""
        img = self.emu.screenshot()
        # Crop Top Screen (256x192)
        top_screen = img.crop((0, 0, 256, 192))
        # Resize to 84x84 and convert to grayscale
        gray = top_screen.resize((config.STATE_W, config.STATE_H), Image.Resampling.LANCZOS).convert('L')
        return np.array(gray, dtype=np.uint8)

    def _read_ram(self):
        """Direct RAM reads using your established offsets."""
        base_ptr = int.from_bytes(self.emu.memory.unsigned[config.ADDR_BASE_POINTER:config.ADDR_BASE_POINTER+4], 'little')
        race_ptr = int.from_bytes(self.emu.memory.unsigned[config.ADDR_RACE_INFO_POINTER:config.ADDR_RACE_INFO_POINTER+4], 'little')
        
        if base_ptr == 0 or race_ptr == 0:
            return 0.0, 0, 0, 0, 1.0
            
        # Physics and progress data
        speed = int.from_bytes(self.emu.memory.unsigned[base_ptr + config.OFFSET_SPEED:base_ptr + config.OFFSET_SPEED + 4], 'little', signed=True) / 4096.0
        angle = int.from_bytes(self.emu.memory.unsigned[base_ptr + config.OFFSET_ANGLE:base_ptr + config.OFFSET_ANGLE + 2], 'little', signed=True)
        checkpoint = self.emu.memory.unsigned[race_ptr + config.OFFSET_CHECKPOINT]
        lap = self.emu.memory.unsigned[race_ptr + config.OFFSET_LAP]
        offroad_mod = int.from_bytes(self.emu.memory.unsigned[base_ptr + config.OFFSET_OFFROAD:base_ptr + config.OFFSET_OFFROAD + 4], 'little', signed=True) / 4096.0
        
        return speed, angle, checkpoint, lap, offroad_mod

    def calculate_reward(self, speed, angle, checkpoint, lap, offroad_mod):
        """Rewards speed and correct direction, penalizes offroad."""
        # Base reward is speed
        reward = speed * 2.0
        
        # Wrong direction determination from Lua logic
        if checkpoint < len(self.chkpnt_ang):
            ref_angle = self.chkpnt_ang[checkpoint]
            # Normalize angles for calculation
            reangle = 65520 + angle if angle < 0 else angle
            reref_angle = 65520 + ref_angle if ref_angle < 0 else ref_angle
            
            angle_diff = min(abs(reref_angle - reangle) % 65520, (65520 - abs(reref_angle - reangle) % 65520))
            if angle_diff > 16380: # Over 90 degrees deviation
                reward = -abs(reward) # Penalize going backwards
        
        # Progression bonuses
        if checkpoint > self.prev_checkpoint:
            reward += 15.0
        if lap > self.prev_lap:
            reward += 100.0
            
        # Offroad penalty
        if offroad_mod < 0.9:
            reward *= 0.5
            
        self.prev_checkpoint = checkpoint
        self.prev_lap = lap
        return reward

    def step(self, action_idx):
        """Applies action, advances 4 frames, and returns observation."""
        self.emu.input.keypad_update(0)
        for key in self.action_map[action_idx]:
            self.emu.input.keypad_add_key(key)
        
        # Advance emulator (Frame Skip 4)
        for _ in range(4):
            self.emu.cycle()
        
        self.window.draw()
        
        next_raw_frame = self._get_state()
        speed, angle, cp, lap, offroad = self._read_ram()
        reward = self.calculate_reward(speed, angle, cp, lap, offroad)
        
        # Done if race is finished (3 laps)
        done = True if lap >= 3 else False
        
        return next_raw_frame, reward, done

    def reset(self):
        """Loads boot savestate and returns initial state."""
        if os.path.exists(config.SAVE_FILE_NAME):
            self.emu.savestate.load_file(config.SAVE_FILE_NAME)
        
        self.prev_checkpoint = 0
        self.prev_lap = 0
        return self._get_state()