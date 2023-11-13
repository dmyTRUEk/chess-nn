# this program evals move positions

import time
import subprocess
import os
from sys import argv as cli_args
import datetime

# import pyautogui 

FEN_INPUT_FIELD_XY = 1200, 1000
EMPTY_SPACE_XY = 1880, 540
SCORE_COPY_POS_XY = 1490, 85

ANALYZE_TIME = 10
DELAY_BETWEEN_INPUTS = 0.2


def main():
    # filename = input("Input filename: ")
    filename = cli_args[1]
    with open(filename, "r") as file_in:
        now = datetime.datetime.now()
        now = f"{now.year}-{now.month}-{now.day}_{now.hour}-{now.minute}-{now.second}"
        for i in range(10)[:0:-1]:
            print(f"Starting in {i} seconds...")
            time.sleep(1)
        print("Starting!")
        with open(filename+"_evaluated_"+now, "w") as file_out:
            n = 0
            for line in file_in:
                n += 1
                # if n >= 10: break
                fen = line.strip()
                score = analyze_position(fen)
                score_fen = f"{score} {fen}"
                print(score_fen)
                file_out.write(score_fen+'\n')
                if n % 100 == 0:
                    file_out.flush()
            print("Done!")


def analyze_position(fen: str) -> str:
    time.sleep(DELAY_BETWEEN_INPUTS)
    mouse_move_to(*FEN_INPUT_FIELD_XY)
    time.sleep(DELAY_BETWEEN_INPUTS)
    mouse_click_primary()
    time.sleep(DELAY_BETWEEN_INPUTS)
    set_copy_paste_buffer(fen)
    time.sleep(DELAY_BETWEEN_INPUTS)
    keyboard_paste()
    time.sleep(DELAY_BETWEEN_INPUTS)
    mouse_move_to(*EMPTY_SPACE_XY)
    time.sleep(DELAY_BETWEEN_INPUTS)
    mouse_click_primary()
    time.sleep(ANALYZE_TIME)
    mouse_move_to(*SCORE_COPY_POS_XY)
    time.sleep(DELAY_BETWEEN_INPUTS)
    mouse_click_primary(3)
    time.sleep(DELAY_BETWEEN_INPUTS)
    keyboard_copy()
    time.sleep(DELAY_BETWEEN_INPUTS)
    score = get_copy_paste_buffer().strip()
    time.sleep(DELAY_BETWEEN_INPUTS)
    return score


XDG_SEAT = os.environ["XDG_SEAT"]

def mouse_move_to(x, y):
    subprocess.run(f"swaymsg seat {XDG_SEAT} cursor set {x} {y}".split())

def mouse_click_primary(n=1):
    for _ in range(n):
        subprocess.run(f"swaymsg seat {XDG_SEAT} cursor press button1".split())
        subprocess.run(f"swaymsg seat {XDG_SEAT} cursor release button1".split())
        time.sleep(0.05)

# def mouse_click_secondary(n=1):
#     for _ in range(n):
#         subprocess.run(f"swaymsg seat {XDG_SEAT} cursor press button2".split())
#         subprocess.run(f"swaymsg seat {XDG_SEAT} cursor release button2".split())
#         time.sleep(0.01)

# def keyboard_press(key):
#     raise NotImplemented()

def keyboard_paste():
    subprocess.run("wtype -M ctrl v -m ctrl".split())

# def keyboard_write(text):
#     raise NotImplemented()

def keyboard_copy():
    subprocess.run("wtype -M ctrl c -m ctrl".split())

def get_copy_paste_buffer() -> str:
    return subprocess.check_output(["wl-paste"]).decode("utf-8")

def set_copy_paste_buffer(text: str):
    return subprocess.run(["wl-copy", text])




if __name__ == '__main__':
    main()

