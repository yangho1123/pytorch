# Tri-Othello
import random
import math
# 13x13

#遊戲局勢
class State:

    def __init__(self, pieces=None, enemy_pieces=None,sec_enemy_pieces=None, depth=0):
        self.dxy = ((1, 0), (1, 1), (0, 1), (-1, 0), (-1, -1), (0, -1)) #(1, -1) (-1, 1)
        self.pass_end = False

        #傳入雙方的棋子做判斷
        self.pieces = pieces
        self.enemy_pieces = enemy_pieces
        self.sec_enemy_pieces = sec_enemy_pieces
        self.depth = depth

        # 棋盤中央交替放置黑白兩子(初始盤面)
        if pieces == None or enemy_pieces == None or sec_enemy_pieces == None:

            self.pieces = [-1]*169
            valid_positions = [19, 20, 21, 22, 23, 24, 31, 32, 33, 34, 35, 36, 37,
                   43, 44, 45, 46, 47, 48, 49, 50, 55, 56, 57, 58, 59,
                   60, 61, 62, 63, 67, 68, 69, 70, 71, 72, 73, 74, 75,
                   76, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 92,
                   93, 94, 95, 96, 97, 98, 99, 100, 101, 105, 106, 107,
                   108, 109, 110, 111, 112, 113, 118, 119, 120, 121, 122,
                   123, 124, 125, 131, 132, 133, 134, 135, 136, 137, 144,
                   145, 146, 147, 148, 149]
            for pos in valid_positions:
                self.pieces[pos] = 0                
            self.pieces[71] = self.pieces[97] = 1       #red

            self.enemy_pieces = [-1]*169
            for pos in valid_positions:
                self.enemy_pieces[pos] = 0 
            self.enemy_pieces[72] = self.enemy_pieces[84] = self.enemy_pieces[96] = 1  #green

            self.sec_enemy_pieces = [-1]*169
            for pos in valid_positions:
                self.sec_enemy_pieces[pos] = 0
            self.sec_enemy_pieces[83] = self.sec_enemy_pieces[85] = 1       #blue
        
    # 取得棋盤上的棋子數量    
    def piece_count(self, pieces):
        count = 0
        for i in pieces:
            if i == 1:
                count += 1

        return count
    
    #判定是否落敗(red最少子)
    def is_lose(self):
        return self.is_done() and (self.piece_count(self.pieces) < self.piece_count(self.enemy_pieces)) and (self.piece_count(self.pieces) < self.piece_count(self.sec_enemy_pieces))
    
    def is_draw(self):
        return self.is_done() and self.piece_count(self.pieces) == self.piece_count(self.enemy_pieces) == self.piece_count(self.sec_enemy_pieces)
    
    def is_done(self):
        return self.piece_count(self.pieces) + self.piece_count(self.enemy_pieces) + self.piece_count(self.sec_enemy_pieces) == 91 or self.pass_end
    
    #取得下一個盤面
    def next(self, action):
        
        state = State(self.pieces.copy(), self.enemy_pieces.copy(), self.sec_enemy_pieces.copy(), self.depth+1)  #先複製下一回合的盤面
        if action != 169:
            state.is_legal_action_xy(action%13, int(action/13), True)     #判斷下的棋子是否能夾住敵方
        w = state.pieces
        state.pieces = state.enemy_pieces   #交換敵我的棋子顏色
        
        state.enemy_pieces = w

        #判斷是否連續2次棄權
        if action == 169 and state.legal_actions() == [169]:
            state.pass_end = True

        return state
    
    #取得合法棋步的串列
    def legal_actions(self):
        actions = []
        for j in range(0, 13):
            for i in range(0, 13):
                if self.is_legal_action_xy(i, j):
                    actions.append(i+j*13)
        if len(actions) == 0:
            actions.append(169)
        return actions
    
    #判斷下的棋子是否能夾住敵方
    def is_legal_action_xy(self, x, y, flip=False):

        #判斷任意格子的任意方向是否能夾住棋子
        def is_legal_action_xy_dxy(x, y, dx, dy):
            x, y = x+dx, y+dy
            if self.enemy_pieces[x+y*13] == -1 or self.pieces[x+y*13] == -1 or self.sec_enemy_pieces[x+y*13] == -1: #若超出棋盤範圍則回傳false
                return False
            
            if y < 1 or 11 < y or x < 1 or 11 < x or (self.enemy_pieces[x+y*13] != 1 and self.sec_enemy_pieces[x+y*13] != 1): #若x+dx和y+dy不介於1~12之間，或該座標沒有敵方的棋，回傳False
                return False
            
            #該座標有敵方的棋才會進入下面
            if self.enemy_pieces[x+y*13] == 1:   #green

                for j in range(12): #對角線最長11個數 判斷指定方向下一格是否為空格或-1
                    if y < 1 or 11 < y or x < 1 or 11 < x or (self.enemy_pieces[x+y*13] == 0 and self.pieces[x+y*13] == 0 and self.sec_enemy_pieces[x+y*13] == 0) or (self.enemy_pieces[x+y*13] == -1 or self.pieces[x+y*13] == -1 or self.sec_enemy_pieces[x+y*13] == -1):   # 若x+dx和y+dy不介於1~12之間，或該座標為空格，回傳False
                        return False                
                # 再出現blue or red就可以夾
                    if self.pieces[x+y*13] == 1:    #如果該格有red(代表可以夾)
                        if flip:   #如果已經翻面
                            for i in range(12):      #確認為合法棋步後，將那個方向上被包夾的敵方棋子全部換成我方棋子
                                x, y = x-dx, y-dy   #變回我方的棋子座標
                                if self.pieces[x+y*13] == 1: #
                                    return True
                                self.pieces[x+y*13] = 1     #red
                                self.enemy_pieces[x+y*13] = 0
                                self.sec_enemy_pieces[x+y*13] = 0
                        return True
                    
                    if self.sec_enemy_pieces[x+y*13] == 1:  #如果該格有blue(代表可以夾)
                        if flip:   #如果已經翻面
                            for i in range(12):      #確認為合法棋步後，將那個方向上被包夾的敵方棋子全部換成我方棋子
                                x, y = x-dx, y-dy   #往反方向逐個更改為red
                                if self.pieces[x+y*13] == 1:  #
                                    return True
                                self.pieces[x+y*13] = 1      #red
                                self.enemy_pieces[x+y*13] = 0
                                self.sec_enemy_pieces[x+y*13] = 0
                        return True             

                    x, y = x+dx, y+dy   #往那個方向的下一格移動
                return False
            if self.sec_enemy_pieces[x+y*13] == 1:   #blue
                for j in range(12): #對角線最長11個數 判斷指定方向下一格是否為空格或-1
                    if y < 1 or 11 < y or x < 1 or 11 < x or (self.enemy_pieces[x+y*13] == 0 and self.pieces[x+y*13] == 0 and self.sec_enemy_pieces[x+y*13] == 0) or (self.enemy_pieces[x+y*13] == -1 or self.pieces[x+y*13] == -1 or self.sec_enemy_pieces[x+y*13] == -1):   # 若x+dx和y+dy不介於1~12之間，或該座標為空格，回傳False
                        return False
                    
                    # 再出現green or red就可以夾

                    if self.pieces[x+y*13] == 1:    #如果該格有red(代表可以夾)
                        if flip:   #如果已經翻面
                            for i in range(12):      #確認為合法棋步後，將那個方向上被包夾的敵方棋子全部換成我方棋子
                                x, y = x-dx, y-dy   #變回我方的棋子座標
                                if self.pieces[x+y*13] == 1: #
                                    return True
                                self.pieces[x+y*13] = 1     #red
                                self.enemy_pieces[x+y*13] = 0
                                self.sec_enemy_pieces[x+y*13] = 0
                        return True      
                                 
                    if self.enemy_pieces[x+y*13] == 1:  #如果該格有green(代表可以夾)
                        if flip:   #如果已經翻面
                            for i in range(12):      #確認為合法棋步後，將那個方向上被包夾的敵方棋子全部換成我方棋子
                                x, y = x-dx, y-dy   #往反方向逐個更改為red
                                if self.pieces[x+y*13] == 1:  #
                                    return True
                                self.pieces[x+y*13] = 1      #red
                                self.enemy_pieces[x+y*13] = 0
                                self.sec_enemy_pieces[x+y*13] = 0
                        return True 
                    
                    x, y = x+dx, y+dy   #往那個方向的下一格移動
                return False
        
        #如果要下的格子已經有棋子則回傳false
        if self.enemy_pieces[x+y*13] == 1 or self.pieces[x+y*13] == 1 or self.sec_enemy_pieces[x+y*13] == 1:
            return False
        
        #如果不是棋盤位置則回傳false
        if self.enemy_pieces[x+y*13] == -1 or self.pieces[x+y*13] == -1 or self.sec_enemy_pieces[x+y*13] == -1:
            return False
        
        if flip:
            self.pieces[x+y*13] = 1 #我方盤面陣列的該位置設為1
        
        #判斷6個方向是否能夠包夾敵方棋子
        flag = False
        for dx, dy in self.dxy:
            if is_legal_action_xy_dxy(x, y, dx, dy):
                flag = True
        return flag
    
    #判斷是否為先手
    def is_first_player(self):
        return self.depth%2 == 0
    
    #顯示遊戲結果
    def __str__(self):
        ox = ('o', 'x') if self.is_first_player() else ('x', 'o')
        str = ''
        for i in range(64):
            if self.pieces[i] == 1:
                str += ox[0]
            elif self.enemy_pieces[i] == 1:
                str += ox[1]
            else:
                str += '-'
            if i % 8 == 7:
                str += '\n'
        return str
    
def random_action(state):
    legal_actions = state.legal_actions()
    return legal_actions[random.randint(0, len(legal_actions)-1)]

def alpha_beta(state, alpha, beta, depth):     # 計算局勢價值
    if depth == 0:
        return alpha
    if state.is_lose():          # 只要判定其中一方是否落敗
        return -1
    if state.is_draw():
        return 0
    for action in state.legal_actions():        

        score = -alpha_beta(state.next(action), -beta, -alpha, depth-1)      #卡在這
        if score > alpha:
            alpha = score
        if alpha >= beta:
            return alpha
    return alpha

def alpha_beta_action(state, max_depth=6):
    best_action = 0
    alpha = -float('inf')   # 負無窮大
    output = ['','']
    for action in state.legal_actions():
        print("action=", action)
        score = -alpha_beta(state.next(action), -float('inf'), -alpha, max_depth-1)  #卡在這
        print("score=", score)
        if score > alpha:
            best_action = action
            alpha = score
        output[0] = '{}{:2d},'.format(output[0], action)
        output[1] = '{}{:2d},'.format(output[1], score)
    print("方法:Alpha-beta剪枝")
    print("合法棋步:", output[0], '\n 局勢價值:', output[1], '\n')
    return best_action

def playout(state):
    if state.is_lose():
        return -1
    if state.is_draw():
        return 0
    #傳回下一個盤面局勢價值
    return -playout(state.next(random_action(state))) #因為下一個局勢是對手的，所以加負號
    
def argmax(collection):
    return collection.index(max(collection))
    
def mcs_action(state):
    legal_actions = state.legal_actions() #取得合法棋步(一維陣列，ex:0,6,15,40,63)
    values = [0] * len(legal_actions)     # 建立list儲存Playout 10次的結果
    for i, action in enumerate(legal_actions):
        for _ in range(10):
            values[i] += -playout(state.next(action))
    return legal_actions[argmax(values)]
 
def mcts_action(state):
    class Node:
        def __init__(self, state):
            self.state = state
            self.w = 0
            self.n = 0
            self.child_nodes = None
        def evaluate(self):
            if self.state.is_done():
                value = -1 if self.state.is_lose() else 0   #根據遊戲結果取得價值
                
                self.w += value     #累計價值
                self.n += 1         #試驗次數
                return value
            if not self.child_nodes:        #抵達葉節點，因此進行playout
                value = playout(self.state)

                #更新累計價值與試驗次數
                self.w += value
                self.n += 1

                if self.n == 10:
                    self.expand()
                return value
            else:
                value = -self.next_child_node().evaluate()  #根據UCB1最大的子節點進行評估(遞迴)

                #更新累計價值與試驗次數
                self.w += value
                self.n += 1
                return value
            
        def expand(self):           #擴展子節點
            legal_actions = self.state.legal_actions()  #取得合法棋步
            self.child_nodes = []
            for action in legal_actions:
                self.child_nodes.append(Node(self.state.next(action))) #儲存子節點

        def next_child_node(self):      #取得UCB1最大的子節點
            #先檢查每個子節點的試驗次數是否為0，有的話就回傳
            for child_node in self.child_nodes:
                if child_node.n == 0:
                    return child_node
            
            t = 0
            for c in self.child_nodes:
                t += c.n
            ucb1_values = []
            for child_node in self.child_nodes:
                ucb1_values.append(-child_node.w/child_node.n+(2*math.log(t)/child_node.n)**0.5)

            return self.child_nodes[argmax(ucb1_values)]
        
    #建立當前局勢的節點
    root_node = Node(state)
    root_node.expand()

    #執行100次模擬
    for _ in range(100):
        root_node.evaluate()

    legal_actions = state.legal_actions()
    n_list = []
    for c in root_node.child_nodes:
        n_list.append(c.n)   

    return legal_actions[argmax(n_list)]
    
if __name__ == '__main__':        
    state = State()

    while True:
        #遊戲結束時
        if state.is_done():
            break
        state = state.next(random_action(state))

        print(state)
        print()

