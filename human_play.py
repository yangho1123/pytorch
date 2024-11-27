from game import State
from pytorch.pv_mcts_old import pv_mcts_action
import torch
from pathlib import Path
import tkinter as tk
from tkinter import messagebox
import time

class GameUI(tk.Frame):
    def __init__(self, master=None, model=None):
        tk.Frame.__init__(self, master)
        self.master.title('Othello_three-player')
        
        self.unit_x = 30
        self.unit_y = 17.320508
        self.position = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (6, 2), (8, 2), (10, 2), (12, 2), (14, 2), (16, 2),
                 (0, 0), (0, 0), (0, 0), (0, 0), (5, 5), (7, 5), (9, 5), (11, 5), (13, 5), (15, 5), (17, 5),
                 (0, 0), (0, 0), (0, 0), (4, 8), (6, 8), (8, 8), (10, 8), (12, 8), (14, 8), (16, 8), (18, 8),
                 (0, 0), (0, 0), (3, 11), (5, 11), (7, 11), (9, 11), (11, 11), (13, 11), (15, 11), (17, 11), (19, 11),
                 (0, 0), (2, 14), (4, 14), (6, 14), (8, 14), (10, 14), (12, 14), (14, 14), (16, 14), (18, 14), (20, 14),
                 (1, 17), (3, 17), (5, 17), (7, 17), (9, 17), (11, 17), (13, 17), (15, 17), (17, 17), (19, 17), (21, 17),
                 (2, 20), (4, 20), (6, 20), (8, 20), (10, 20), (12, 20), (14, 20), (16, 20), (18, 20), (20, 20), (0, 0),
                 (3, 23), (5, 23), (7, 23), (9, 23), (11, 23), (13, 23), (15, 23), (17, 23), (19, 23), (0, 0), (0, 0),
                 (4, 26), (6, 26), (8, 26), (10, 26), (12, 26), (14, 26), (16, 26), (18, 26), (0, 0), (0, 0), (0, 0),
                 (5, 29), (7, 29), (9, 29), (11, 29), (13, 29), (15, 29), (17, 29), (0, 0), (0, 0), (0, 0), (0, 0),
                 (6, 32), (8, 32), (10, 32), (12, 32), (14, 32), (16, 32), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]

        self.state = State()
        self.next_action = pv_mcts_action(model, 0.0) # 產生用pv_mcts產生動作的function，呼叫就會產生action
        
        self.c = tk.Canvas(self, width=720, height=624, highlightthickness=0)
        self.c.bind('<Button-1>', self.turn_of_human) # 第一手都為人類玩家
        
        self.c.pack()
        self.on_draw()

    def turn_of_human(self, event):
        # game over
        if self.state.is_done(): # 結束時點一下會回到初始盤面
            self.print_end()
            self.state = State()
            self.master.after(3000, self.on_draw())
            return
        
        # AI turn
        if not self.state.get_player_order() == 0: # 這邊可能要改一下，我把原本的first_player函式改成下面那個
            return
        
        # 將點擊的位置座標轉換成格子編號
        click_x = float(event.x/self.unit_x)
        click_y = float(event.y/self.unit_y)
        print(f"{click_x}, {click_y}")
        action = -1
        for i in range(121):
            if click_x < 1 or click_x > 22 or click_y < 1 or click_y > 35:
                break
            x, y = self.position[i]
            if click_x < x or click_x > x+2:
                continue
            if click_y < y-1 or click_y > y+3:
                continue
            temp_x = click_x - x
            temp_y = click_y - y
            if temp_y >= 0 and temp_y <= 2:
                action = i
                break
            if temp_x < 1:
                if temp_y < 0:
                    if temp_y + temp_x > 0:
                        action = i
                        break
                else: # temp_y = 2
                    if temp_y-2 < temp_x:
                        action = i
                        break
            else:
                if temp_y < 0:
                    if temp_y + 2 > temp_x:
                        action = i
                        break
                else:
                    if temp_y + temp_x < 4:
                        action = i
                        break
        
        # 點擊位置是否合法
        legal_actions = self.state.legal_actions()
        if legal_actions == [121]:
            action = 121
        if action == -1 or action not in legal_actions: # 直接return表示動作無效
            return
        
        self.state = self.state.next(action)
        self.on_draw()

        self.master.after(1, self.turn_of_ai) # 利用after延遲1ms等待更新完成在做玩家腳色交換

    def turn_of_ai(self):
        if self.state.is_done():
            return
        
        action = self.next_action(self.state)
        self.state = self.state.next(action)
        self.on_draw()

        if self.state.get_player_order() == 2:
            self.master.after(1, self.turn_of_ai)

    def draw_piece(self, index, player_order):
        x, y = self.position[index]
        x *= self.unit_x
        y *= self.unit_y
        if player_order == 0:
            if self.state.mine_pieces[index] == 1:
                self.c.create_oval(x+10, y-2.7, x+50, y+37.3, width=1.0, outline='#000000', fill='#FF0000')
            elif self.state.next_pieces[index] == 1:
                self.c.create_oval(x+10, y-2.7, x+50, y+37.3, width=1.0, outline='#000000', fill='#00FF00')
            else:
                self.c.create_oval(x+10, y-2.7, x+50, y+37.3, width=1.0, outline='#000000', fill='#0000FF')
        elif player_order == 1:
            if self.state.prev_pieces[index] == 1:
                self.c.create_oval(x+10, y-2.7, x+50, y+37.3, width=1.0, outline='#000000', fill='#FF0000')
            elif self.state.mine_pieces[index] == 1:
                self.c.create_oval(x+10, y-2.7, x+50, y+37.3, width=1.0, outline='#000000', fill='#00FF00')
            else:
                self.c.create_oval(x+10, y-2.7, x+50, y+37.3, width=1.0, outline='#000000', fill='#0000FF')
        else:
            if self.state.next_pieces[index] == 1:
                self.c.create_oval(x+10, y-2.7, x+50, y+37.3, width=1.0, outline='#000000', fill='#FF0000')
            elif self.state.prev_pieces[index] == 1:
                self.c.create_oval(x+10, y-2.7, x+50, y+37.3, width=1.0, outline='#000000', fill='#00FF00')
            else:
                self.c.create_oval(x+10, y-2.7, x+50, y+37.3, width=1.0, outline='#000000', fill='#0000FF')


    def on_draw(self):
        # 每個格子寬60，邊長34.641016 -> (30, 17.320508)，共十一格。左右留30、上下留17.320508
        self.c.delete('all')
        for i in range(121):
            x, y = self.position[i]
            if x == 0 and y == 0:
                continue
            x *= self.unit_x
            y *= self.unit_y
            # 背景
            points1 = [x, y, x + self.unit_x, y - self.unit_y, x + 2*self.unit_x, y]
            points2 = [x, y + 2*self.unit_y, x + self.unit_x, y + 3*self.unit_y, x + 2*self.unit_x, y + 2*self.unit_y]
            self.c.create_polygon(points1, fill='#C69C6C')
            # 格子線
            self.c.create_rectangle(x, y, x + 2*self.unit_x, y + 2*self.unit_y, width=0.0, fill='#C69C6C')
            self.c.create_polygon(points2, fill='#C69C6C')
            self.c.create_line(x, y, x + self.unit_x, y - self.unit_y, width=1.0, fill='#000000')
            self.c.create_line(x + self.unit_x, y + 3*self.unit_y, x + 2*self.unit_x, y + 2*self.unit_y, width=1.0, fill='#000000')
            self.c.create_line(x + self.unit_x, y - self.unit_y, x + 2*self.unit_x, y, width=1.0, fill='#000000')
            self.c.create_line(x, y + 2*self.unit_y, x + self.unit_x, y + 3*self.unit_y, width=1.0, fill='#000000')
            self.c.create_line(x, y, x, y + 2*self.unit_y, width=1.0, fill='#000000')
            self.c.create_line(x + 2*self.unit_x, y, x + 2*self.unit_x, y + 2*self.unit_y, width=1.0, fill='#000000')
            

        for i in range(121):
            if self.state.mine_pieces[i] == -1 or (self.state.mine_pieces[i] == 0 and self.state.next_pieces[i] == 0 and self.state.prev_pieces[i] == 0):
                continue
            self.draw_piece(i, self.state.get_player_order())

    def print_end(self):
        self.c.delete('all')
        if self.state.get_player_order() == 0:
            messagebox.showinfo("result", f"Red: {self.state.piece_count(self.state.mine_pieces)}, Green: {self.state.piece_count(self.state.next_pieces)}, Blue: {self.state.piece_count(self.state.prev_pieces)}")
        elif self.state.get_player_order() == 1:
            messagebox.showinfo("result", f"Red: {self.state.piece_count(self.state.prev_pieces)}, Green: {self.state.piece_count(self.state.mine_pieces)}, Blue: {self.state.piece_count(self.state.next_pieces)}")
        else:
            messagebox.showinfo("result", f"Red: {self.state.piece_count(self.state.next_pieces)}, Green: {self.state.piece_count(self.state.prev_pieces)}, Blue: {self.state.piece_count(self.state.mine_pieces)}")

#start = time.time()
model = torch.jit.load('./model/best.pt')
#end = time.time()
#print("加載模型：%f 秒" % (end - start))
f = GameUI(model=model)
f.pack()
f.mainloop()