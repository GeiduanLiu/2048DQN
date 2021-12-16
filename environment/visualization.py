import tkinter as tk
import numpy as np


class GameBoard:
    def __init__(self, game):
        self.game = game

        self.win_width = 800
        self.win_height = 600
        self.grid_num = 4
        self.grid_size = (self.win_height * 0.8) / self.grid_num
        self.grid_color_dict = {
            '0': '#C7BDB1',
            '2': '#E4DACE',
            '4': '#E4D9C7',
            '8': '#E5B489',
            '16': '#E7A176',
            '32': '#E88C73',
            '64': '#E77654',
            '128': '#E1CB82',
            '256': '#E2CA74',
            '512': '#E3C66A',
            '1024': '#E3C35C',
            '2048': '#E2C14E',
            '4096': '#E485C9',
            '8192': '#E179C4',
            '16384': '#E16DC5',
        }

        self.win = tk.Tk()
        self.win.title('Game 2048')
        self.win.geometry('%dx%d' % (self.win_width, self.win_height))

        self.keypress_lb = tk.Label(self.win, text='Press', font=('Arial', 20))
        self.keypress_lb.bind('<Key>', self.key_response)
        self.keypress_lb.focus_set()
        self.keypress_lb.place(relx=0.75, rely=0.1)

        self.score_lb = tk.Label(self.win, text='Score: 0', font=('Arial', 20))
        self.score_lb.place(relx=0.75, rely=0.2)

        self.done_lb = tk.Label(self.win, text='', font=('Arial', 20), fg='red')
        self.done_lb.place(relx=0.75, rely=0.3)

        self.canvas = tk.Canvas(self.win, width=self.grid_size * self.grid_num, height=self.grid_size * self.grid_num,
                                bg='#BBACA0')
        self.canvas.place(relx=0.1, rely=0.1)

        # self.show_board(data)

    def start(self, data):
        self.show_board(data)
        self.win.mainloop()

    def show_board(self, data):
        # clear canvas
        self.canvas.delete(tk.ALL)

        start_offset = 0.02
        gap = 0.02
        for i in range(self.grid_num):
            for j in range(self.grid_num):
                text = str(int(data[i, j]))

                y1 = (i + gap + start_offset) * self.grid_size
                y2 = (i + 1 - gap + start_offset) * self.grid_size
                x1 = (j + gap + start_offset) * self.grid_size
                x2 = (j + 1 - gap + start_offset) * self.grid_size
                self.canvas.create_rectangle(x1, y1, x2, y2,
                                             fill=self.grid_color_dict.get(text, self.grid_color_dict['16384']))

                x = (x1 + x2) / 2
                y = (y1 + y2) / 2
                color = 'black' if text in ['2', '4'] else 'white'
                fontsize = int(self.grid_size * 0.6 / np.power(len(text), 0.35))
                if text != '0':
                    self.canvas.create_text(x, y, text=text, fill=color, font=('Arial', fontsize))

    def key_response(self, event):
        s = event.keysym
        self.keypress_lb.config(text='Operation: %s' % s)
        # print("event.char=", event.char)
        # print("event.keycode=", event.keycode)
        done = False
        if s in self.game.actions:
            ret = self.game.step(s)
            if ret == 'op error':
                return
            _, _, done = ret
        elif s == 'q':
            exit()
        elif s == 'p':
            self.game.reset()
            self.done_lb.config(text='')

        self.show_board(data=self.game.board)
        self.score_lb.config(text='Score: %s' % self.game.get_score())
        if done:
            self.done_lb.config(text='Game Over!')
