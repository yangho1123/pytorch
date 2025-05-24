# ====================
# 黑白棋
# ====================

# 匯入套件
import random
from collections import deque
import numpy as np
import copy, sys
# 貼目數
red_komi = 0
green_komi = 1
blue_komi = 2
TEMPERATURE = 0
Bound = 100
SUPER =100
HIGH = 10
LOW = 0.1
DIRECTIONS = [-11, -10, -1, 1, 10, 11]  # 六個方向
outer_corner = [5,10,55,65,110,115] # 六個角落
centers = [49,50,59,60,61,70,71]
middle = []
PathTable = [
    -1, -1, -1, -1, -1, 4, 4, 4, 4, 4, 4,
    -1, -1, -1, -1, 4, 3, 3, 3, 3, 3, 4,
    -1, -1, -1, 4, 3, 2, 2, 2, 2, 3, 4,
    -1, -1, 4, 3, 2, 1, 1, 1, 2, 3, 4,
    -1, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4,
    4, 3, 2, 1, 0, 0, 0, 1, 2, 3, 4,
    4, 3, 2, 1, 0, 0, 1, 2, 3, 4, -1,
    4, 3, 2, 1, 1, 1, 2, 3, 4, -1, -1,
    4, 3, 2, 2, 2, 2, 3, 4, -1, -1, -1,
    4, 3, 3, 3, 3, 3, 4, -1, -1, -1, -1,
    4, 4, 4, 4, 4, 4, -1, -1, -1, -1, -1
]

FlipTable = [
    0, 0, 0, 0, 0, 1, 1.6, 1.7, 1.8, 1.7, 1,
    0, 0, 0, 0, 1.6, 3, 3.2, 3.2, 3.7, 3, 1.6,
    0, 0, 0, 1.7, 3.3, 4, 4, 4, 3.8, 3.2, 1.7,
    0, 0, 1.7, 3.3, 4, 4.7, 4.7, 4.5, 4, 3.3, 1.7,
    0, 1.6, 3.2, 4, 4.8, 4.7, 4.8, 4.8, 4, 3.3, 1.6,
    1, 3, 4, 4.8, 4.9, 3.2, 4.9, 4.8, 4.1, 3, 1,
    1.6, 3.2, 4.1, 4.8, 4.8, 4.7, 4.7, 4, 3.2, 1.6, 0,
    1.7, 3.2, 4, 4.4, 4.7, 4.7, 4, 3.2, 1.7, 0, 0,
    1.7, 3.2, 3.9, 4.1, 4.1, 4, 3.2, 1.7, 0, 0, 0,
    1.5, 3, 3.2, 3.2, 3.2, 3, 1.6, 0, 0, 0, 0,
    1, 1.5, 1.7, 1.7, 1.6, 1, 0, 0, 0, 0, 0
]
# 遊戲狀態
class State:
    # 初期化
    def __init__(self, pieces=None, next_pieces=None, prev_pieces=None, depth=0, pass_turn=0, WeightTable=None): # 有棄權規則，因此必須另外紀錄回合數
        self.dxy = ((-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0))
        self.pass_end = False # 三方棄權的結束
        # 棋子的配置
        self.mine_pieces = pieces
        self.next_pieces = next_pieces
        self.prev_pieces = prev_pieces
        self.depth = depth
        self.pass_turn = pass_turn
        if WeightTable == None:
            self.WeightTable = [1] * 121
        else:
            self.WeightTable = WeightTable
        self.directions = [+1, -1, +10, -10, +11, -11]
        
        if pieces == None and prev_pieces == None and next_pieces == None:
            self.mine_pieces = [0] * 121
            self.mine_pieces[49] = self.mine_pieces[71] = 1 # 紅子
            self.next_pieces = [0] * 121
            self.next_pieces[50] = self.next_pieces[60] = self.next_pieces[70] = 1 # 綠子
            self.prev_pieces = [0] * 121
            self.prev_pieces[59] = self.prev_pieces[61] = 1 # 藍子
            for i in range(11):
                for j in range(11):
                    if (i + j <= 4) or (i + j >= 16):
                        self.mine_pieces[i*11+j] = -1
                        self.prev_pieces[i*11+j] = -1
                        self.next_pieces[i*11+j] = -1

    # 取得棋子數量
    def piece_count(self, pieces):
        count = 0
        for i in pieces:
            if i == 1:
                count += 1
        return count

    # 判斷輸贏
    def finish(self):
        mine_count = self.piece_count(self.mine_pieces)
        next_count = self.piece_count(self.next_pieces)
        prev_count = self.piece_count(self.prev_pieces)
        if self.get_player_order() == 0: # red
            mine_count -= red_komi
            next_count -= green_komi
            prev_count -= blue_komi
            #print(f"{mine_count}, {next_count}, {prev_count}")
            if mine_count >= prev_count and mine_count >= next_count:
                if next_count >= prev_count:
                    return [1, 0, -1]
                else:
                    return [1, -1, 0]
            elif next_count >= prev_count:
                if mine_count >= prev_count:
                    return [0, 1, -1]
                else:
                    return [-1, 1, 0]
            else:
                if mine_count >= next_count:
                    return [0, -1, 1]
                else:
                    return [-1, 0, 1]
        elif self.get_player_order() == 1: # green
            mine_count -= green_komi
            next_count -= blue_komi
            prev_count -= red_komi
            #print(f"{prev_count}, {mine_count}, {next_count}")
            if mine_count > prev_count and mine_count >= next_count:
                if next_count > prev_count:
                    return [-1, 1, 0]
                else:
                    return [0, 1, -1]
            elif next_count > prev_count:
                if mine_count > prev_count:
                    return [-1, 0, 1]
                else:
                    return [0, -1, 1]
            else:
                if mine_count >= next_count:
                    return [1, 0, -1]
                else:
                    return [1, -1, 0]
        else: # blue
            mine_count -= blue_komi
            next_count -= red_komi
            prev_count -= green_komi
            #print(f"{next_count}, {prev_count}, {mine_count}")
            if mine_count > prev_count and mine_count > next_count:
                if next_count >= prev_count:
                    return [0, -1, 1]
                else:
                    return [-1, 0, 1]
            elif next_count >= prev_count:
                if mine_count > prev_count:
                    return [1, -1, 0]
                else:
                    return [1, 0, -1]
            else:
                if mine_count > next_count:
                    return [-1, 1, 0]
                else:
                    return [0, 1, -1]
        
    # 判斷遊戲是否結束
    def is_done(self):
        return self.piece_count(self.mine_pieces)+self.piece_count(self.prev_pieces)+self.piece_count(self.next_pieces) == 91 or self.pass_end==True

    # 取得下一個狀態
    def next(self, action):
        state = State(self.mine_pieces.copy(), self.next_pieces.copy(), self.prev_pieces.copy(), self.depth+1, self.pass_turn, WeightTable=self.WeightTable.copy())
        if action != 121:
            state.is_legal_action_xy(action%11, int(action/11), True) # x, y, flip -> 下棋
        temp = state.mine_pieces
        state.mine_pieces = state.next_pieces
        state.next_pieces = state.prev_pieces
        state.prev_pieces = temp
        if action == 121:
            state.pass_turn += 1
        else:
            state.pass_turn = 0
        if state.pass_turn == 2 and state.legal_actions() == [121]: #三方皆pass，結束比賽
            state.pass_end = True
        return state
    
    # 取得合法著手的list(有分組)
    def legal_actions(self):    # 根據權重表將合法走步分組
        actions = []       
        for y in range(11):
            for x in range(11):
                if(self.mine_pieces[x+y*11] == -1):
                    continue
                if self.is_legal_action_xy(x, y, False): # 判斷是否可以下子而已還不用做翻面的動作
                    actions.append(x+y*11)
        if len(actions) == 0:
            actions.append(121)
        high_act, normal_act, low_act = self.weight_group(actions)  # 將合法走步分成三組
        if high_act:
            return high_act
        elif normal_act:
            return normal_act
        else:
            return low_act
    # (原本的無分組)
    def all_legal_actions(self):
        actions = []       
        for y in range(11):
            for x in range(11):
                if(self.mine_pieces[x+y*11] == -1):
                    continue
                if self.is_legal_action_xy(x, y, False): # 判斷是否可以下子而已還不用做翻面的動作
                    actions.append(x+y*11)
        if len(actions) == 0:
            actions.append(121)
        return actions
    
    # 判斷下的位置是否能夠夾住對方的棋子
    def is_legal_action_xy(self, x, y, flip=False):
        def is_legal_action_xy_dxy(x, y, dx, dy):
            x, y = x+dx, y+dy
            if x < 0 or y < 0 or x > 10 or y > 10 or self.mine_pieces[x+y*11] == -1 or (self.prev_pieces[x+y*11] != 1 and self.next_pieces[x+y*11] != 1):
                return False # 該格的該方向沒有敵方棋子 -> 沒有辦法夾
            enemy = 1 if self.prev_pieces[x+y*11] == 1 else 2
            for _ in range(11):
                if x < 0 or y < 0 or x > 10 or y > 10 or self.mine_pieces[x+y*11] == -1 or (self.mine_pieces[x+y*11] == 0 and self.prev_pieces[x+y*11] == 0 and self.next_pieces[x+y*11] == 0):
                    return False # 中間斷掉or超過邊界了，沒有辦法
                if self.mine_pieces[x+y*11] == 1 or (enemy == 1 and self.next_pieces[x+y*11] == 1) or (enemy == 2 and self.prev_pieces[x+y*11] == 1):
                    if flip: # 該直線上有可以與原始x,y相夾的點存在(有己方或第三方的點)
                        for i in range(11):
                            x, y = x-dx, y-dy # 前進尋找有無可以形成夾子的己方或第三方 -> 後退翻轉棋子
                            if self.mine_pieces[x+y*11] == 1: 
                                return True # 到達己方的點 -> 翻轉結束
                            self.mine_pieces[x+y*11] = 1
                            if enemy == 1:
                                self.prev_pieces[x+y*11] = 0
                            else:
                                self.next_pieces[x+y*11] = 0
                    return True
                x, y = x+dx, y+dy
            return False
                 
        if self.mine_pieces[x+y*11] == 1 or self.prev_pieces[x+y*11] == 1 or self.next_pieces[x+y*11] == 1:
            return False # 該位置已經有子
        
        if flip:
            self.mine_pieces[x+y*11] = 1

        flag = False
        for dx, dy in self.dxy:
            if is_legal_action_xy_dxy(x, y, dx, dy):
                flag = True
        return flag
    # 判斷是否為高權重或低權重並分組
    def weight_group(self, actions):
        global outer_corner
        high = []
        normal = []
        low = []
        self.updateWeight()
        for action in actions:
            if action == 121:
                high.append(action)
            elif self.WeightTable[action] == HIGH:
                high.append(action)
            elif self.WeightTable[action] == LOW:
                low.append(action)
            else:
                normal.append(action)
            
        return high, normal, low
    
    # 更新權重表
    def updateWeight(self):
        self.WeightTable = [1] * 121        
        global outer_corner, centers  
        dir = [[1, 10],[-1,11],[-10,11],[-11,10],[-11,1],[-1,-10]] 
        # 初始化權重表        
        for pos in centers:            # 中央棋格高權重
            self.WeightTable[pos] = HIGH              
        for i, pos in enumerate(outer_corner):  # 六角落高權重
            self.WeightTable[pos] = HIGH
        # 六個角落或延伸點設為高權重
        # for i, pos in enumerate(outer_corner):
        #     if i == 0:                
        #         for j in range(2):  #分別往2個方向找，若該格有子則將下一格設為high
        #             meet = 0
        #             ifValue = True
        #             # 該格有任一方的子則進入迴圈
        #             while self.mine_pieces[pos]==1 or self.next_pieces[pos]==1 or self.prev_pieces[pos]==1:
        #                 if meet == 2 or meet == 3:
        #                     if self.mine_pieces[pos] == 1:  # 前面已有別人的棋又遇到我的棋
        #                         ifValue = False
        #                         break
        #                 if self.mine_pieces[pos] == 0:    # 不是我方的子
        #                     if self.next_pieces[pos]==1:  # meet紀錄是第一個遇到的非我方子是哪一家的子
        #                         newmeet = 2
        #                     else:
        #                         newmeet = 3
        #                     if (pos==outer_corner[i]) or (meet!=0 and newmeet!=meet):   
        #                         ifValue = False # 還在角落或是遇到第三種顏色(這次的newmeet和上次的不同)
        #                         break
        #                     meet = newmeet      # 下一次迴圈meet就變成2或3
        #                 if (pos+dir[i][j] == 10)or(pos+dir[i][j] == 55):    # 到另一角落的前一格
        #                     break    
        #                 pos += dir[i][j]
        #             if ifValue and (self.mine_pieces[pos]==0 and self.next_pieces[pos]==0 and self.prev_pieces[pos]==0):     
        #                 self.WeightTable[pos] = HIGH
        #             pos = outer_corner[i]

        #     if i == 1:
        #         for j in range(2):  #分別往2個方向找，若該格有子則將pos設為下一格
        #             meet = 0
        #             ifValue = True
        #             while self.mine_pieces[pos]==1 or self.next_pieces[pos]==1 or self.prev_pieces[pos]==1:
        #                 if meet == 2 or meet == 3:
        #                     if self.mine_pieces[pos] == 1:  # 前面已有別人的棋又遇到我的棋
        #                         ifValue = False
        #                         break
        #                 if self.mine_pieces[pos] == 0:    # 不是我方的子
        #                     if self.next_pieces[pos]==1:  # meet紀錄是第一個遇到的非我方子是哪一家的子
        #                         newmeet = 2
        #                     else:
        #                         newmeet = 3
        #                     if (pos==outer_corner[i]) or (meet!=0 and newmeet!=meet):   
        #                         ifValue = False # 還在角落或是遇到第三種顏色(這次的newmeet和上次的不同)
        #                         break
        #                     meet = newmeet      # 下一次迴圈meet就變成2或3
        #                 if (pos+dir[i][j] == 5)or(pos+dir[i][j] == 65):    # 到另一角落的前一格
        #                     break   
        #                 pos += dir[i][j]
        #             if ifValue and (self.mine_pieces[pos]==0 and self.next_pieces[pos]==0 and self.prev_pieces[pos]==0):     
        #                 self.WeightTable[pos] = HIGH
        #             pos = outer_corner[i]
        #     if i == 2:
        #         for j in range(2):  #分別往2個方向找，若該格有子則將下一格設為high
        #             meet = 0
        #             ifValue = True
        #             while self.mine_pieces[pos]==1 or self.next_pieces[pos]==1 or self.prev_pieces[pos]==1:
        #                 if meet == 2 or meet == 3:
        #                     if self.mine_pieces[pos] == 1:  # 前面已有別人的棋又遇到我的棋
        #                         ifValue = False
        #                         break
        #                 if self.mine_pieces[pos] == 0:    # 不是我方的子
        #                     if self.next_pieces[pos]==1:  # meet紀錄是第一個遇到的非我方子是哪一家的子
        #                         newmeet = 2
        #                     else:
        #                         newmeet = 3
        #                     if (pos==outer_corner[i]) or (meet!=0 and newmeet!=meet):   
        #                         ifValue = False # 還在角落或是遇到第三種顏色(這次的newmeet和上次的不同)
        #                         break
        #                     meet = newmeet      # 下一次迴圈meet就變成2或3
        #                 if (pos+dir[i][j] == 5)or(pos+dir[i][j] == 110):    # 到另一角落的前一格
        #                     break
        #                 pos += dir[i][j]
        #             if ifValue and (self.mine_pieces[pos]==0 and self.next_pieces[pos]==0 and self.prev_pieces[pos]==0):     
        #                 self.WeightTable[pos] = HIGH
        #             pos = outer_corner[i]
        #     if i == 3:
        #         for j in range(2):  #分別往2個方向找，若該格有子則將下一格設為high
        #             meet = 0
        #             ifValue = True
        #             while self.mine_pieces[pos]==1 or self.next_pieces[pos]==1 or self.prev_pieces[pos]==1:
        #                 if meet == 2 or meet == 3:
        #                     if self.mine_pieces[pos] == 1:  # 前面已有別人的棋又遇到我的棋
        #                         ifValue = False
        #                         break
        #                 if self.mine_pieces[pos] == 0:    # 不是我方的子
        #                     if self.next_pieces[pos]==1:  # meet紀錄是第一個遇到的非我方子是哪一家的子
        #                         newmeet = 2
        #                     else:
        #                         newmeet = 3
        #                     if (pos==outer_corner[i]) or (meet!=0 and newmeet!=meet):   
        #                         ifValue = False # 還在角落或是遇到第三種顏色(這次的newmeet和上次的不同)
        #                         break
        #                     meet = newmeet      # 下一次迴圈meet就變成2或3
                            
        #                 if (pos+dir[i][j] == 115)or(pos+dir[i][j] == 10):    # 到另一角落的前一格
        #                     break
        #                 pos += dir[i][j]
        #             if ifValue and (self.mine_pieces[pos]==0 and self.next_pieces[pos]==0 and self.prev_pieces[pos]==0):     
        #                 self.WeightTable[pos] = HIGH
        #             pos = outer_corner[i]
        #     if i == 4:
        #         for j in range(2):  #分別往2個方向找，若該格有子則將下一格設為high
        #             meet = 0
        #             ifValue = True
        #             while self.mine_pieces[pos]==1 or self.next_pieces[pos]==1 or self.prev_pieces[pos]==1:
        #                 if meet == 2 or meet == 3:
        #                     if self.mine_pieces[pos] == 1:  # 前面已有別人的棋又遇到我的棋
        #                         ifValue = False
        #                         break
        #                 if self.mine_pieces[pos] == 0:    # 不是我方的子
        #                     if self.next_pieces[pos]==1:  # meet紀錄是第一個遇到的非我方子是哪一家的子
        #                         newmeet = 2
        #                     else:
        #                         newmeet = 3
        #                     if (pos==outer_corner[i]) or (meet!=0 and newmeet!=meet):   
        #                         ifValue = False # 還在角落或是遇到第三種顏色(這次的newmeet和上次的不同)
        #                         break
        #                     meet = newmeet      # 下一次迴圈meet就變成2或3
        #                 if (pos+dir[i][j] == 55)or(pos+dir[i][j] == 115):    # 到另一角落的前一格
        #                     break    
        #                 pos += dir[i][j]
        #             if ifValue and (self.mine_pieces[pos]==0 and self.next_pieces[pos]==0 and self.prev_pieces[pos]==0):     
        #                 self.WeightTable[pos] = HIGH
        #             pos = outer_corner[i]
        #     if i == 5:
        #         for j in range(2):  #分別往2個方向找，若該格有子則將下一格設為high
        #             meet = 0
        #             ifValue = True
        #             while self.mine_pieces[pos]==1 or self.next_pieces[pos]==1 or self.prev_pieces[pos]==1:
        #                 if meet == 2 or meet == 3:
        #                     if self.mine_pieces[pos] == 1:  # 前面已有別人的棋又遇到我的棋
        #                         ifValue = False
        #                         break
        #                 if self.mine_pieces[pos] == 0:    # 不是我方的子
        #                     if self.next_pieces[pos]==1:  # meet紀錄是第一個遇到的非我方子是哪一家的子
        #                         newmeet = 2
        #                     else:
        #                         newmeet = 3
        #                     if (pos==outer_corner[i]) or (meet!=0 and newmeet!=meet):   
        #                         ifValue = False # 還在角落或是遇到第三種顏色(這次的newmeet和上次的不同)
        #                         break
        #                     meet = newmeet      # 下一次迴圈meet就變成2或3
        #                 if (pos+dir[i][j] == 110)or(pos+dir[i][j] == 65):    # 到另一角落的前一格
        #                     break    
        #                 pos += dir[i][j]
        #             if ifValue and (self.mine_pieces[pos]==0 and self.next_pieces[pos]==0 and self.prev_pieces[pos]==0):     
        #                 self.WeightTable[pos] = HIGH
        #             pos = outer_corner[i]
        border_ranges_dside = [range(5, 10), range(10, 5, -1),
                           range(10, 65, 11), range(65, 10, -11),
                           range(65, 115, 10), range(115, 65, -10),
                           range(115, 110, -1), range(110, 115, 1),
                           range(110, 55, -11), range(55, 110, 11),
                           range(55, 5, -10), range(5, 55, 10)]  
        for border_range in border_ranges_dside:    #5 6 7 8 9
            border_range_list = list(border_range)
            if self.mine_pieces[border_range_list[0]]==0:
                continue  # 如果該格子不是自己的，就換考慮接下來的角落格子(可能是空的或是別人的)
            # 角落是自己的則繼續
            # 把目前此角落上的棋子種類記錄下來
            check_piece = self.get_piece(border_range_list[0])  # 1:red 2:green 3:blue
            changed = False
            not_match = False
            for pos in border_range_list[1:]:
                if self.is_blank(pos):
                    break   # 如果該格子是空的就離開迴圈
                else:
                    if self.get_piece(pos) != check_piece:
                        if changed:
                            not_match = True
                            break
                        else:
                            changed = True
                            check_piece = self.get_piece(pos)
            if not_match:
                continue
            # 檢查這個格子是否需要調整 
            if self.is_blank(pos):
                # 檢查格子的週遭是否沒有HIGH
                need_to_adjust = True
                for d in self.directions:
                    if self.in_board(pos+d) and self.is_blank(pos+d) and self.WeightTable[pos+d]>=HIGH and (pos+d) not in outer_corner:
                        need_to_adjust = False
                        break
                # 如果週遭沒有HIGH，就把自己設為HIGH
                if need_to_adjust:
                    self.WeightTable[pos] = HIGH

        # 6個邊的唯一空位高權重
        border_ranges = [range(5, 10), range(10, 65, 11), 
                      range(65, 115, 10), range(115, 110, -1),
                      range(110, 55, -11), range(55, 5, -10)]
        for border_range in border_ranges:
            blank_counts, found_pos = 0,0
            for pos in border_range:
                if self.is_blank(pos):
                    blank_counts += 1
                    if blank_counts > 1:
                        break
                    found_pos = pos
                if blank_counts == 1 and found_pos not in outer_corner:
                    self.WeightTable[found_pos] = HIGH
        # 若邊邊的中間點還沒被佔，沒有被設為高權重的話且(旁邊有其他方的子or角落被佔)則設為低權重。
        EdgeMiddle = [7,8,32,43,85,95,112,113,77,88,25,35]  # 上(左右)、右上(上下)、右下(左下右上)、下(左右)、左下(上下)、左上(右上右下)
        CheckDir = [[-1,1],[-11,11],[-10,10],[-1,1],[-11,11],[-10,10]]
        # for i,pos in enumerate(EdgeMiddle):
        #     if i%2==0:
        #         d = CheckDir[i//2][0]
        #     else:
        #         d = CheckDir[i//2][1]
        #     if self.is_blank(pos)==False or self.WeightTable[pos]==HIGH or self.WeightTable[pos]==SUPER:
        #         continue    #有子、高權重跳過
        #     if self.is_blank(pos+d) and self.in_board(pos+d):
        #         continue    # 邊的中間兩格的旁邊格子為空跳過
        #     # 角落是否為空
        #     if self.is_blank(pos+d*2) and self.in_board(pos+d*2):
        #         if self.mine_pieces[pos+d] == 0 and (self.next_pieces[pos+d]==1 or self.prev_pieces[pos+d]==1):
        #             self.WeightTable[pos] = LOW    # 相鄰點有子但不是己方設低權重
        #         elif self.mine_pieces[pos+d]==1 and self.mine_pieces[pos-d]==0 and (self.next_pieces[pos-d]==1 or self.prev_pieces[pos-d]==1):
        #             self.WeightTable[pos] = LOW    # 相鄰點雖是己方，但另一邊不是己方設低權重
        #     else:
        #         if self.next_pieces[pos-d]==1 or self.prev_pieces[pos-d]==1:
        #             continue                        # 另一邊相鄰點有子但不是己方
        #         if self.next_pieces[pos+d*2]==1 or self.prev_pieces[pos+d*2]==1:
        #             self.WeightTable[pos] = LOW    # 若該方向的角落已被另兩方佔據則設為低權重

        # 把權重值高的點的鄰居權重值設為低權重
        for pos in range(121):
            if self.mine_pieces[pos]!=-1 and self.WeightTable[pos] == HIGH:
                for d in DIRECTIONS:
                    if (pos == 55) and ((d == 10)or(d == -1)):
                        continue
                    if (pos == 65) and ((d == -10)or(d == 1)):
                        continue
                    if (pos == 66) and ((d == -1)):
                        continue
                    if (0 <= pos+d < 121) and (self.mine_pieces[pos+d] != -1) and (self.WeightTable[pos+d] == 1):
                        self.WeightTable[pos+d] = LOW
        for pos in range(121):
            if self.in_board(pos) and self.is_blank(pos) and self.WeightTable[pos] >= HIGH:
                for d in self.directions:
                    if self.in_board(pos+d) and self.is_blank(pos+d) and ((pos+d) not in outer_corner):
                        self.WeightTable[pos+d] = LOW

    
    def is_blank(self, pos):
        if self.mine_pieces[pos]<=0 and self.next_pieces[pos]<=0 and self.prev_pieces[pos]<=0:
            return True
        else:
            return False
    def in_board(self, pos):
        if pos > 4 and pos < 116 and self.mine_pieces[pos] != -1:
            return True
        else:
            return False
    # 判斷是否為先手  判斷現在輪到哪個玩家
    def get_player_order(self):
        return (self.depth % 3)

    def get_piece(self, pos):   # 1: red 2: green 3: blue
        if self.get_player_order() == 0:
            # mine==red
            if self.mine_pieces[pos]==1:
                return 1
            elif self.next_pieces[pos]==1:
                return 2
            elif self.prev_pieces[pos]==1:
                return 3
            else:
                return None
        if self.get_player_order() == 1:
            # mine==green
            if self.mine_pieces[pos]==1:
                return 2
            elif self.next_pieces[pos]==1:
                return 3
            elif self.prev_pieces[pos]==1:
                return 1
            else:
                return None
        if self.get_player_order() == 2:
            # mine==blue
            if self.mine_pieces[pos]==1:
                return 3
            elif self.next_pieces[pos]==1:
                return 1
            elif self.prev_pieces[pos]==1:
                return 2
            else:
                return None
            
    # 顯示文字列字串
    def __str__(self):
        rgb = ('r', 'g', 'b')
        if self.get_player_order() == 1:
            rgb = ('g', 'b', 'r')
        elif self.get_player_order() == 2:
            rgb = ('b', 'r', 'g')
        str = ''
        for i in range(121):
            if self.mine_pieces[i] == -1:
                str += '|'
            elif self.mine_pieces[i] == 1:
                str += rgb[0]
            elif self.next_pieces[i] == 1:
                str += rgb[1]
            elif self.prev_pieces[i] == 1:
                str += rgb[2]
            else:
                str += '-'
            if i % 11 == 10:
                str += '\n'
        return str


# 隨機選擇動作
def random_action(state):
    legal_actions = state.all_legal_actions()
    return legal_actions[random.randint(0, len(legal_actions)-1)]

# maxn方法產生走步
def maxn_action(depth):
    def maxn_action(state):
        # updatePath(state)
        state.updateWeight()
        printWeightTable(state)
        save_state_to_file(state)
        if state.depth < 8: #前8步隨機 
            return random_action(state)        
        if state.is_done():
            return 121  # 終局時沒有可選動作
        best_scores = [float('-inf')] * 3  # 初始化為負無窮
        best_action = 121
        current_player = state.get_player_order()
        # Normal_version
        for action in state.legal_actions():    # 有分組則用state.legal_actions()             
            # print("depth:", state.depth)
            # print(state.WeightTable)
            # if state.depth > 10:
            #     sys.exit()
            new_state = state.next(action)
            scores = maxn(new_state, depth - 1, bound=100)
            # print("scores:", scores)
            # 對於當前玩家，找到最高分的走步
            if scores[current_player] > best_scores[current_player]:
                best_scores = scores
                best_action = action            
        updatePath(state)
        return best_action
    return maxn_action
# 無權重
def maxn_actionnw(depth):
    def maxn_actionnw(state):
        # updatePath(state)
        if state.depth < 8: #前8步隨機 
            return random_action(state)        
        if state.is_done():
            return 121  # 終局時沒有可選動作
        best_scores = [float('-inf')] * 3  # 初始化為負無窮
        best_action = 121
        current_player = state.get_player_order()
        # 探索所有合法的動作
        for action in state.all_legal_actions():    # 有分組則用state.legal_actions() 
            new_state = state.next(action)
            scores = maxnnw(new_state, depth - 1, bound=100)
            # print("scores:", scores)
            # 對於當前玩家，找到最高分的走步
            if scores[current_player] > best_scores[current_player]:
                best_scores = scores
                best_action = action            
        updatePath(state)
        return best_action
    return maxn_actionnw

def maxn(state, depth, bound):    
    if state.is_done() or depth == 0:
        if state.is_done():
            s = state.finish()
            ns = np.array(s)+1                      
            s = list(ns * Bound / sum(ns))
        return s if state.is_done() else ScoreCount(state)
    state.updateWeight()    # 更新權重表
    best_scores = [float('-inf')] * 3  # 分數初始化
    current_player = state.get_player_order()
    # 探索所有可能的合法動作
    for action in state.legal_actions():       # 有分組則用state.legal_actions() 
        new_state = state.next(action)
        if best_scores[current_player] >= bound:   # 剪枝
            return best_scores
        scores = maxn(new_state, depth-1, bound-best_scores[current_player])
        
        # 找到當前玩家的最佳分數
        if scores[current_player] >= best_scores[current_player]:
            best_scores = scores             

    return best_scores

def maxnnw(state, depth, bound):    
    if state.is_done() or depth == 0:
        if state.is_done():
            s = state.finish()
            ns = np.array(s)+1                      
            s = list(ns * Bound / sum(ns))
        return s if state.is_done() else ScoreCountnw(state)
        
    best_scores = [float('-inf')] * 3  # 分數初始化
    current_player = state.get_player_order()
     # 探索所有可能的合法動作
    for action in state.all_legal_actions():       # 有分組則用state.legal_actions() 
        new_state = state.next(action)
        if best_scores[current_player] >= bound:   # 剪枝
            return best_scores
        scores = maxnnw(new_state, depth-1, bound-best_scores[current_player])
        
        # 找到當前玩家的最佳分數
        if scores[current_player] >= best_scores[current_player]:
            best_scores = scores             

    return best_scores

# 計算當前盤面審局分數
def ScoreCount(state):
    
    score = [0, 0, 0]           # 紅、綠、藍
    cp = state.get_player_order()
    ncp = (cp + 1) % 3
    pcp = (cp + 2) % 3
    # 有權重
    for i in range(121):
        if state.mine_pieces[i] == 1:     
            score[cp] += state.WeightTable[i]/(FlipTable[i]+PathTable[i])
        if state.next_pieces[i] == 1:
            score[ncp] += state.WeightTable[i]/(FlipTable[i]+PathTable[i])
        if state.prev_pieces[i] == 1:
            score[pcp] += state.WeightTable[i]/(FlipTable[i]+PathTable[i])
    # for i in range(121):
    #     if state.mine_pieces[i] == 1:     
    #         score[cp] += 1/(FlipTable[i]+PathTable[i])
    #     if state.next_pieces[i] == 1:
    #         score[ncp] += 1/(FlipTable[i]+PathTable[i])
    #     if state.prev_pieces[i] == 1:
    #         score[pcp] += 1/(FlipTable[i]+PathTable[i])
    for i in range(3):
        score[i] = score[i] * Bound / sum(score)
    return score

# 計算當前盤面審局分數
def ScoreCountnw(state):
    score = [0, 0, 0]           # 紅、綠、藍
    cp = state.get_player_order()
    ncp = (cp + 1) % 3
    pcp = (cp + 2) % 3
    
    # for i in range(121):
    #     if state.mine_pieces[i] == 1:     
    #         score[cp] += state.WeightTable[i]/(FlipTable[i]+PathTable[i])
    #     if state.next_pieces[i] == 1:
    #         score[ncp] += state.WeightTable[i]/(FlipTable[i]+PathTable[i])
    #     if state.prev_pieces[i] == 1:
    #         score[pcp] += state.WeightTable[i]/(FlipTable[i]+PathTable[i])
    # 無權重
    for i in range(121):
        if state.mine_pieces[i] == 1:     
            score[cp] += 1/(FlipTable[i]+PathTable[i])
        if state.next_pieces[i] == 1:
            score[ncp] += 1/(FlipTable[i]+PathTable[i])
        if state.prev_pieces[i] == 1:
            score[pcp] += 1/(FlipTable[i]+PathTable[i])
    for i in range(3):
        score[i] = score[i] * Bound / sum(score)
    return score

def printPathTable(PathTable):
    # 確保 PathTable 是 121 個元素
    if len(PathTable) != 121:
        print("PathTable 長度不正確。")
        return

    # 每 11 個元素換行
    for i in range(0, 121, 11):
        print(' '.join(map(str, PathTable[i:i+11])))

def printWeightTable(state):
    filename = "WeightTable.txt"    
    depth_str = str(state.depth)
    if len(state.WeightTable) != 121:
        print("WeightTable 長度不正確。")
        return
    # 打開檔案準備寫入
    with open(filename, 'a') as file:
        file.write('\n')
        # 每 11 個元素換行，並寫入檔案
        file.write('depth: ' + depth_str + '\n')
        for i in range(0, 121, 11):
            line = ' '.join(map(str, state.WeightTable[i:i+11]))
            file.write(line + '\n')  # 寫入一行數據並換行

def save_state_to_file(state, filename="state_board.txt"):
    board_str = str(state)  # 生成棋盤的字符串表示
    depth_str = str(state.depth)
    with open(filename, 'a') as file:  # 以追加模式打開文件
        file.write('depth: ' + depth_str + '\n')
        file.write(board_str + '\n')  # 寫入棋盤字符串並在末尾添加一個換行符
        file.write('\n')  # 添加一個額外的空行作為分隔符

# 更新路徑表(三位玩家共用)
def updatePath(state):
    global PathTable
    queue = deque()
     # 初始化 PathTable，將有棋子的位置設置為 0，其它有效位置設為一個大數值
    for i in range(121):
        if state.mine_pieces[i] != -1:  # 只處理棋盤有效的位置
            if state.mine_pieces[i] == 1 or state.next_pieces[i] == 1 or state.prev_pieces[i] == 1:
                PathTable[i] = 0
                queue.append(i)
            else:
                PathTable[i] = float('inf')  # 無棋子的位置初始化為無限大
    # 透過 BFS 更新路徑表
    while queue:
        current = queue.popleft()
        current_distance = PathTable[current]

        for direction in DIRECTIONS:
            next_pos = current + direction
            # 確保位置有效，且不是已經被占據或已有更短的路徑
            if (0 <= next_pos < 121) and (state.mine_pieces[next_pos] != -1) and (PathTable[next_pos] > current_distance + 1):
                PathTable[next_pos] = current_distance + 1
                queue.append(next_pos)
 


# 單獨測試
if __name__ == '__main__': # 若該程式為主程式就會執行，若是被其他程式呼叫的則不會執行這段
    state = State()                             # 開始一局新遊戲
    strategies = [                              # 設定三位玩家的策略
        # lambda state: maxn_action(state, depth=3),        
        random_action,
        random_action,
        random_action
    ]  # 玩家1用MaxN，玩家2用model，玩家3用隨機
    
    player = 0  # 從0號玩家開始
    print(state)
    print()
    while not state.is_done():
        action = strategies[player](state)   #呼叫該位玩家使用的方法，回傳一個走步
        state = state.next(action)
        print(state)
        print()

        # 更新到下一位玩家
        player = (player + 1) % 3

    final_scores = state.finish()
    print("Final scores:", final_scores)
    print("Winner:", np.argmax(final_scores))