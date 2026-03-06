import sys
import time
import os
import csv
import argparse

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

# Allow importing hardware modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hardware.serial_handler import STM32_SerialHandler
from hardware.imu_sensor import IMUSensor

class ManualRecorder:
    def __init__(self, use_imu=True):
        self.use_imu = use_imu
        self.recording = False
        self.playing = False
        self.recorded_data = [] # List of tuples: (dt, steer, pwm, yaw)
        
        self.base_speed_pwm = 50.0
        self.max_steer = 25.0
        
        self.current_steer = 0.0
        self.current_pwm = 0.0
        
        self.csv_filename = "telemetry_run.csv"
        
        print("\n--- Initializing Hardware ---")
        self.serial = STM32_SerialHandler()
        if not self.serial.connect():
            print("WARNING: STM32 not connected. Running in simulation mode.")
            
        if self.use_imu:
            self.imu = IMUSensor()
            self.imu.start()
        else:
            self.imu = None
            
        pygame.init()
        self.screen = pygame.display.set_mode((400, 300))
        pygame.display.set_caption("BFMC Manual Recorder")
        self.font = pygame.font.SysFont("monospace", 15)
        self.clock = pygame.time.Clock()
        
    def _draw_hud(self, fps):
        self.screen.fill((30, 30, 30))
        
        text_lines = [
            f"--- BFMC Telemetry Console ---",
            f"FPS: {fps:.1f}",
            "",
            f"Steer (A/D): {self.current_steer:>5.1f} deg",
            f"PWM   (W/S): {self.current_pwm:>5.1f}",
            f"Yaw        : {self.imu.get_yaw() if self.imu else 0.0:>5.1f} deg",
            "",
            "Controls:",
            "  [R] - Start/Stop Recording",
            "  [P] - Playback Telemetry CSV",
            "  [ESC] - Quit",
            ""
        ]
        
        if self.recording:
            text_lines.append(f"STATUS: [REC] {len(self.recorded_data)} samples")
            color = (255, 50, 50)
        elif self.playing:
            text_lines.append("STATUS: [PLAYING]")
            color = (50, 255, 50)
        else:
            text_lines.append("STATUS: IDLE")
            color = (200, 200, 200)
            
        for i, line in enumerate(text_lines):
            text_surface = self.font.render(line, True, color if "STATUS" in line else (255,255,255))
            self.screen.blit(text_surface, (20, 20 + i * 20))
            
        pygame.display.flip()

    def load_csv(self):
        data = []
        if not os.path.exists(self.csv_filename):
            print(f"Error: {self.csv_filename} not found.")
            return []
            
        with open(self.csv_filename, 'r') as f:
            reader = csv.reader(f)
            next(reader) # skip header
            for row in reader:
                if len(row) == 4:
                    data.append((float(row[0]), float(row[1]), float(row[2]), float(row[3])))
        print(f"Loaded {len(data)} samples from {self.csv_filename}")
        return data

    def save_csv(self):
        if not self.recorded_data:
            print("No data to save.")
            return
            
        with open(self.csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['dt', 'steer_deg', 'speed_pwm', 'yaw_deg'])
            for row in self.recorded_data:
                writer.writerow(row)
        print(f"Saved {len(self.recorded_data)} samples to {self.csv_filename}")

    def play_sequence(self):
        data = self.load_csv()
        if not data:
            return
            
        print("\n--- PLAYBACK STARTED ---")
        self.playing = True
        
        for dt, steer, pwm, yaw in data:
            # Handle quit events during playback
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.playing = False
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    self.playing = False
                    print("Playback aborted by user.")
                    self.serial.set_speed(0.0)
                    self.serial.set_steering(0.0)
                    return
            
            self.current_steer = steer
            self.current_pwm = pwm
            
            # Send to hardware
            self.serial.set_speed(self.current_pwm)
            self.serial.set_steering(self.current_steer)
            
            self._draw_hud(self.clock.get_fps())
            time.sleep(dt)
            
        self.serial.set_speed(0.0)
        self.serial.set_steering(0.0)
        self.current_steer = 0.0
        self.current_pwm = 0.0
        self.playing = False
        print("--- PLAYBACK FINISHED ---")

    def run(self):
        running = True
        last_time = time.time()
        
        while running:
            dt = self.clock.tick(30) / 1000.0 # Target 30 FPS
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r and not self.playing:
                        if self.recording:
                            self.recording = False
                            self.save_csv()
                        else:
                            print("\n--- RECORDING STARTED ---")
                            self.recorded_data = []
                            self.recording = True
                    elif event.key == pygame.K_p and not self.recording:
                        loaded = self.load_csv()
                        if loaded:
                            self.playing = True
                            self.play_data = loaded
                            self.play_idx = 0
                            print("\n--- PLAYBACK STARTED ---")
            
            if self.playing:
                if self.play_idx < len(self.play_data):
                    dt, steer, pwm, yaw = self.play_data[self.play_idx]
                    self.current_steer = steer
                    self.current_pwm = pwm
                    
                    self.serial.set_speed(self.current_pwm)
                    self.serial.set_steering(self.current_steer)
                    self._draw_hud(self.clock.get_fps())
                    
                    time.sleep(dt)
                    self.play_idx += 1
                else:
                    print("--- PLAYBACK FINISHED ---")
                    self.serial.set_speed(0.0)
                    self.serial.set_steering(0.0)
                    self.current_steer = 0.0
                    self.current_pwm = 0.0
                    self.playing = False
                    last_time = time.time()
                continue
                
            keys = pygame.key.get_pressed()
            
            # Keyboard Logic
            self.current_pwm = 0.0
            if keys[pygame.K_w] or keys[pygame.K_UP]:
                self.current_pwm = self.base_speed_pwm
            elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
                self.current_pwm = -self.base_speed_pwm # Reverse
                
            self.current_steer = 0.0
            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                self.current_steer = -self.max_steer
            elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                self.current_steer = self.max_steer
                
            # Send to hardware
            self.serial.set_speed(self.current_pwm)
            self.serial.set_steering(self.current_steer)
            
            # Record
            if self.recording:
                now = time.time()
                real_dt = now - last_time
                yaw = self.imu.get_yaw() if self.imu else 0.0
                self.recorded_data.append((real_dt, self.current_steer, self.current_pwm, yaw))
                last_time = now
            else:
                last_time = time.time()
                
            self._draw_hud(self.clock.get_fps())
            
        print("\nShutting down Manual Recorder...")
        if self.recording:
            self.save_csv()
        self.serial.set_speed(0.0)
        self.serial.set_steering(0.0)
        if self.imu:
            self.imu.stop()
        pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BFMC Manual Recording & Playback")
    parser.add_argument("--no-imu", action="store_true", help="Disable IMU background thread")
    args = parser.parse_args()
    
    app = ManualRecorder(use_imu=not args.no_imu)
    app.run()
