import sys
import time
import os
import csv
import argparse
import curses
import threading

# Allow importing hardware modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hardware.serial_handler import STM32_SerialHandler
try:
    from hardware.imu_sensor import IMUSensor
    _IMU_AVAILABLE = True
except ImportError:
    _IMU_AVAILABLE = False


class ManualRecorder:
    def __init__(self, stdscr, use_imu=True):
        self.stdscr = stdscr
        self.use_imu = use_imu
        self.recording = False
        self.playing = False
        self.recorded_data = [] # List of tuples: (dt, steer, pwm, yaw)
        
        self.base_speed_pwm = 50.0
        self.max_steer = 25.0
        
        self.current_steer = 0.0
        self.current_pwm = 0.0
        
        self.csv_filename = "telemetry_run.csv"
        self.message_log = "--- Initializing Hardware ---"
        
        # Make getch non-blocking
        self.stdscr.nodelay(True)
        curses.curs_set(0) # Hide cursor
        
        self.serial = STM32_SerialHandler()
        if not self.serial.connect():
            self.message_log = "WARNING: STM32 not connected. Simulating."
            
        if self.use_imu and _IMU_AVAILABLE:
            self.imu = IMUSensor()
            self.imu.start()
        else:
            self.imu = None
            
        self.target_fps = 30
        self.frame_period = 1.0 / self.target_fps

    def _draw_hud(self, actual_fps):
        self.stdscr.erase()
        
        header = "=== BFMC Terminal Telemetry Console ==="
        self.stdscr.addstr(0, 0, header, curses.A_BOLD)
        self.stdscr.addstr(1, 0, f"FPS: {actual_fps:.1f}")
        
        self.stdscr.addstr(3, 0, f"Steer (A/D): {self.current_steer:>5.1f} deg")
        self.stdscr.addstr(4, 0, f"PWM   (W/S): {self.current_pwm:>5.1f}")
        y_yaw = self.imu.get_yaw() if self.imu else 0.0
        self.stdscr.addstr(5, 0, f"Yaw        : {y_yaw:>5.1f} deg")
        
        self.stdscr.addstr(7, 0, "Controls:")
        self.stdscr.addstr(8, 2, "[R] - Start/Stop Recording")
        self.stdscr.addstr(9, 2, "[P] - Playback Telemetry CSV")
        self.stdscr.addstr(10, 2, "[ESC] or [Q] - Quit")
        
        # Status Line
        if self.recording:
            stat_str = f"STATUS: [REC] {len(self.recorded_data)} samples"
            self.stdscr.addstr(12, 0, stat_str, curses.A_STANDOUT)
        elif self.playing:
            self.stdscr.addstr(12, 0, "STATUS: [PLAYING]", curses.A_BOLD)
        else:
            self.stdscr.addstr(12, 0, "STATUS: IDLE")
            
        self.stdscr.addstr(14, 0, f"Log: {self.message_log}")
            
        self.stdscr.refresh()

    def load_csv(self):
        data = []
        if not os.path.exists(self.csv_filename):
            self.message_log = f"Error: {self.csv_filename} not found."
            return []
            
        with open(self.csv_filename, 'r') as f:
            reader = csv.reader(f)
            next(reader) # skip header
            for row in reader:
                if len(row) == 4:
                    data.append((float(row[0]), float(row[1]), float(row[2]), float(row[3])))
        self.message_log = f"Loaded {len(data)} samples."
        return data

    def save_csv(self):
        if not self.recorded_data:
            self.message_log = "No data to save."
            return
            
        with open(self.csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['dt', 'steer_deg', 'speed_pwm', 'yaw_deg'])
            for row in self.recorded_data:
                writer.writerow(row)
        self.message_log = f"Saved {len(self.recorded_data)} samples."

    def run(self):
        running = True
        last_time = time.time()
        fps_time = last_time
        frames_this_sec = 0
        actual_fps = 0.0
        
        play_data = []
        play_idx = 0
        
        while running:
            now = time.time()
            loop_dt = now - last_time
            
            # FPS Calculation
            frames_this_sec += 1
            if now - fps_time >= 1.0:
                actual_fps = frames_this_sec / (now - fps_time)
                frames_this_sec = 0
                fps_time = now
            
            # Handle Keyboard Input
            c = self.stdscr.getch()
            
            if c != -1:
                if c == 27 or c == ord('q') or c == ord('Q'): # ESC or Q
                    running = False
                    break
                
                # Command Toggles
                if not self.playing:
                    if c == ord('r') or c == ord('R'):
                        if self.recording:
                            self.recording = False
                            self.save_csv()
                        else:
                            self.message_log = "RECORDING STARTED"
                            self.recorded_data = []
                            self.recording = True
                    
                    elif c == ord('p') or c == ord('P'):
                        if not self.recording:
                            loaded = self.load_csv()
                            if loaded:
                                self.playing = True
                                play_data = loaded
                                play_idx = 0
                                self.message_log = "PLAYBACK STARTED"
                                
                else: # Allow cancelling playback
                    if c == ord('c') or c == ord('C'):
                        self.playing = False
                        self.message_log = "Playback aborted by user."
                        self.serial.set_speed(0.0)
                        self.serial.set_steering(0.0)
            
            # Subsystem Logic
            if self.playing:
                if play_idx < len(play_data):
                    dt, steer, pwm, yaw = play_data[play_idx]
                    self.current_steer = steer
                    self.current_pwm = pwm
                    
                    self.serial.set_speed(self.current_pwm)
                    self.serial.set_steering(self.current_steer)
                    self._draw_hud(actual_fps)
                    
                    time.sleep(dt) # Emulate exact recorded timeframe
                    play_idx += 1
                else:
                    self.message_log = "PLAYBACK FINISHED"
                    self.serial.set_speed(0.0)
                    self.serial.set_steering(0.0)
                    self.current_steer = 0.0
                    self.current_pwm = 0.0
                    self.playing = False
                    last_time = time.time()
                continue
                
            # Manual Control state
            self.current_pwm = 0.0
            self.current_steer = 0.0
            
            if c == ord('w') or c == curses.KEY_UP:
                self.current_pwm = self.base_speed_pwm
            elif c == ord('s') or c == curses.KEY_DOWN:
                self.current_pwm = -self.base_speed_pwm
            elif c == ord('a') or c == curses.KEY_LEFT:
                self.current_steer = -self.max_steer
            elif c == ord('d') or c == curses.KEY_RIGHT:
                self.current_steer = self.max_steer
                
            self.serial.set_speed(self.current_pwm)
            self.serial.set_steering(self.current_steer)
            
            # Record
            if self.recording:
                real_dt = now - last_time
                yaw = self.imu.get_yaw() if self.imu else 0.0
                self.recorded_data.append((real_dt, self.current_steer, self.current_pwm, yaw))
                
            last_time = now
            self._draw_hud(actual_fps)
            
            # Sleep to maintain frame rate
            elapsed_loop = time.time() - now
            sleep_time = self.frame_period - elapsed_loop
            if sleep_time > 0:
                time.sleep(sleep_time)
                
        # Shutdown
        if self.recording:
            self.save_csv()
        self.serial.set_speed(0.0)
        self.serial.set_steering(0.0)
        if self.imu:
            self.imu.stop()

def main(stdscr):
    parser = argparse.ArgumentParser(description="BFMC Manual Recording & Playback")
    parser.add_argument("--no-imu", action="store_true", help="Disable IMU background thread")
    # Parse args manually because curses wrap steals sys.argv slightly weirdly if not careful
    args = parser.parse_args()
    
    app = ManualRecorder(stdscr, use_imu=not args.no_imu)
    app.run()

if __name__ == "__main__":
    curses.wrapper(main)
