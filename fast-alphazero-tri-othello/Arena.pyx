# cython: language_level=3

from pytorch_classification.utils import Bar, AverageMeter
import time


class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, player3, game, display=None):
        """
        Input:
            player 1,2,3: three functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1  # 紅方
        self.player2 = player2  # 綠方
        self.player3 = player3  # 藍方
        self.game = game
        self.display = display

    def playGame(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player1, self.player2, self.player3]
        board = self.game.getInitBoard()
        it = 0
        curPlayer = 0  # 從紅方開始
        
        while self.game.getGameEnded(board, 1) == 0:
            it += 1
            if verbose:
                assert(self.display)
                print("Turn ", str(it), "Player ", str(curPlayer))
                self.display(board)
            action = players[curPlayer](self.game.getCanonicalForm(board, 1), it)

            valids = self.game.getValidMoves(
                self.game.getCanonicalForm(board, 1), 1)

            if valids[action] == 0:
                print()
                print(action)
                print(valids)
                print()
                assert valids[action] > 0
            board, _ = self.game.getNextState(board, 1, action)
            curPlayer = (curPlayer + 1) % 3  # 輪到下一個玩家
        if verbose:
            assert(self.display)
            print("Game over: Turn ", str(it), "Result ",
                  str(self.game.getGameEnded(board, 1)))
            self.display(board)
        return self.game.getGameEnded(board, 1)  # 返回結果數組 [紅, 綠, 藍]

    def playGames(self, num, verbose=False):
        """
        進行多場遊戲，每個玩家輪流作為先手。
        
        參數:
            num: 要進行的遊戲總數
            verbose: 是否顯示詳細遊戲過程
            
        返回:
            redWins: 紅方（player1）獲勝次數
            greenWins: 綠方（player2）獲勝次數
            blueWins: 藍方（player3）獲勝次數
            draws: 平局次數
        """
        eps_time = AverageMeter()
        bar = Bar('Arena.playGames', max=num)
        end = time.time()
        eps = 0
        maxeps = int(num)

        # 三方遊戲的結果計數
        redWins = 0   # player1 獲勝次數
        greenWins = 0  # player2 獲勝次數
        blueWins = 0   # player3 獲勝次數
        draws = 0      # 平局次數

        # 讓每個玩家平等地開局
        games_per_player = num // 3
        remaining_games = num % 3  # 處理不能被3整除的情況
        
        # 紅方（player1）開局的場次
        red_starting_games = games_per_player + (1 if remaining_games > 0 else 0)
        for _ in range(red_starting_games):
            # 紅方（player1）開局
            result = self.playGame(verbose=verbose)
            
            # 結果是 [紅方分數, 綠方分數, 藍方分數]
            if result[0] > 0:  # 紅方獲勝
                redWins += 1
            elif result[1] > 0:  # 綠方獲勝
                greenWins += 1
            elif result[2] > 0:  # 藍方獲勝
                blueWins += 1
            else:  # 平局
                draws += 1
            
            # 更新統計和進度
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            total_games = redWins + greenWins + blueWins + draws
            if total_games > 0:
                red_rate = redWins / total_games * 100
                green_rate = greenWins / total_games * 100
                blue_rate = blueWins / total_games * 100
            else:
                red_rate = green_rate = blue_rate = 0
                
            bar.suffix = '({eps}/{maxeps}) R:{red:.1f}% G:{green:.1f}% B:{blue:.1f}% | Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(
                eps=eps, maxeps=maxeps, red=red_rate, green=green_rate, blue=blue_rate,
                et=eps_time.avg, total=bar.elapsed_td, eta=bar.eta_td)
            bar.next()

        # 輪換玩家順序，讓 player2（綠方）開局
        self.player1, self.player2, self.player3 = self.player2, self.player3, self.player1

        # 綠方開局的場次
        green_starting_games = games_per_player + (1 if remaining_games > 1 else 0)
        for _ in range(green_starting_games):
            # 現在綠方作為 player1 開局
            result = self.playGame(verbose=verbose)
            
            # 結果仍然是 [當前player1分數, 當前player2分數, 當前player3分數]
            # 但順序已變化，需要映射回原始玩家
            if result[0] > 0:  # 現在的player1（原綠方）獲勝
                greenWins += 1
            elif result[1] > 0:  # 現在的player2（原藍方）獲勝
                blueWins += 1
            elif result[2] > 0:  # 現在的player3（原紅方）獲勝
                redWins += 1
            else:
                draws += 1
                
            # 更新統計和進度
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            total_games = redWins + greenWins + blueWins + draws
            if total_games > 0:
                red_rate = redWins / total_games * 100
                green_rate = greenWins / total_games * 100
                blue_rate = blueWins / total_games * 100
            else:
                red_rate = green_rate = blue_rate = 0
                
            bar.suffix = '({eps}/{maxeps}) R:{red:.1f}% G:{green:.1f}% B:{blue:.1f}% | Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(
                eps=eps, maxeps=maxeps, red=red_rate, green=green_rate, blue=blue_rate,
                et=eps_time.avg, total=bar.elapsed_td, eta=bar.eta_td)
            bar.next()
        
        # 再次輪換玩家順序，讓 player3（藍方）開局
        self.player1, self.player2, self.player3 = self.player2, self.player3, self.player1

        # 藍方開局的場次
        blue_starting_games = games_per_player
        for _ in range(blue_starting_games):
            # 現在藍方作為 player1 開局
            result = self.playGame(verbose=verbose)
            
            # 結果映射回原始玩家
            if result[0] > 0:  # 現在的player1（原藍方）獲勝
                blueWins += 1
            elif result[1] > 0:  # 現在的player2（原紅方）獲勝
                redWins += 1
            elif result[2] > 0:  # 現在的player3（原綠方）獲勝
                greenWins += 1
            else:
                draws += 1
                
            # 更新統計和進度
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            total_games = redWins + greenWins + blueWins + draws
            if total_games > 0:
                red_rate = redWins / total_games * 100
                green_rate = greenWins / total_games * 100
                blue_rate = blueWins / total_games * 100
            else:
                red_rate = green_rate = blue_rate = 0
                
            bar.suffix = '({eps}/{maxeps}) R:{red:.1f}% G:{green:.1f}% B:{blue:.1f}% | Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(
                eps=eps, maxeps=maxeps, red=red_rate, green=green_rate, blue=blue_rate,
                et=eps_time.avg, total=bar.elapsed_td, eta=bar.eta_td)
            bar.next()

        # 恢復原始玩家順序
        self.player1, self.player2, self.player3 = self.player3, self.player1, self.player2

        bar.update()
        bar.finish()

        return redWins, greenWins, blueWins, draws
