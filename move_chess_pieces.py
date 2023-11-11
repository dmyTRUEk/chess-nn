# this program moves chess pieces on board by clicking coords

import time
import os
import subprocess

X_MIN = 523
X_MAX = 523+864
Y_MIN = 60
Y_MAX = 60+864

LETTERS = 'abcdefgh'

SPEED = 250


def main():
    while True:
        play_game()


def play_game():
    moves = str(input('input game moves: '))
    moves = moves.strip()
    moves = moves.split(' ')
    # print(f'{moves = }')

    time.sleep(7)

    is_whites_move = True
    # is_whites_move = False

    for move in moves:
        cx1, cy1, cx2, cy2, create_fig = move[0], move[1], move[2], move[3], ''
        if len(move) > 4:
            create_fig = move[4]
        # print(cx1, cy1, cx2, cy2)

        x1, y1, x2, y2 = LETTERS.find(cx1), int(cy1)-1, LETTERS.find(cx2), int(cy2)-1
        # print(x1, y1, x2, y2)

        mouse_click_at(
            X_MIN + (X_MAX-X_MIN)*(x1+0.5)/8,
            Y_MAX - (Y_MAX-Y_MIN)*(y1+0.5)/8,
        )
        time.sleep(50/SPEED)

        mouse_click_at(
            X_MIN + (X_MAX-X_MIN)*(x2+0.5)/8,
            Y_MAX - (Y_MAX-Y_MIN)*(y2+0.5)/8,
        )
        time.sleep(50/SPEED)

        if create_fig != '':
            x3 = x2
            if is_whites_move:
                if create_fig == 'q':
                    y3 = 7
                elif create_fig == 'n':
                    y3 = 6
                elif create_fig == 'r':
                    y3 = 5
                elif create_fig == 'b':
                    y3 = 4
            else:
                if create_fig == 'q':
                    y3 = 0
                elif create_fig == 'n':
                    y3 = 1
                elif create_fig == 'r':
                    y3 = 2
                elif create_fig == 'b':
                    y3 = 3

            mouse_click_at(
                X_MIN + (X_MAX-X_MIN)*(x3+0.5)/8,
                Y_MAX - (Y_MAX-Y_MIN)*(y3+0.5)/8,
            )
            time.sleep(50/SPEED)

        is_whites_move = not is_whites_move
    print()


XDG_SEAT = os.environ["XDG_SEAT"]

def mouse_move_to(x, y):
    subprocess.run(f"swaymsg seat {XDG_SEAT} cursor set {x} {y}".split())

def mouse_click_primary(n=1):
    for _ in range(n):
        subprocess.run(f"swaymsg seat {XDG_SEAT} cursor press button1".split())
        subprocess.run(f"swaymsg seat {XDG_SEAT} cursor release button1".split())
        time.sleep(0.05)

def mouse_click_at(x, y, n=1):
    mouse_move_to(x, y)
    mouse_click_primary(n)


if __name__ == '__main__':
    main()

