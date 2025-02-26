import copy
import time
import random

class Go_Using_MinMax_Alpha_Beta_Pruning:
    #initializing all the variables in init function for using throughout code
    def __init__(self):
        self.player = None
        self.opponent = None
        self.board_size = 5
        self.max_depth = 5 #setting max depth as 5
        self.time_limit = 9.8 #setting max time limit as 9.8, slightly less than 10 so that we don't hit time limit error
        self.start_time = None
        self.previous_board = None
        self.board = None
        self.branching_factor = 5 #keeping our branching factor as 5 as it work well against all the AI Agents so far

    #We will read and parse game input using this function
    def read_input(self, input_file):
        with open(input_file, 'r') as f:
            lines = f.readlines()

        self.player = int(lines[0].strip()) #initializing our player
        self.opponent = 3 - self.player #initializing our opponent
        self.previous_board = [list(map(int, line.strip())) for line in lines[1:6]]
        self.board = [list(map(int, line.strip())) for line in lines[6:11]]

    #Writing select move to the output file using this function
    def write_output(self, best_move, output_file):
        if best_move == "PASS" or best_move is None: #Checking if there is no best move else we send the best move
            result = "PASS"
        else:
            result = f"{best_move[0]},{best_move[1]}"

        with open(output_file, 'w') as f:
            f.write(result)

    #We will be fetching all the valid moves using this function for the current player
    def fetch_known_valid_moves(self, board, player):
        valid_moves = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i][j] == 0 and self.inquire_valid_move(board, (i, j), player): #Will check if the cell is empty and will call inquire_valid_move function to check if the move is valid or not
                    valid_moves.append((i, j))
        return valid_moves

    #In this function we will check if the move is valid move or not ticking the rules like the KO rule and the liberty rule
    def inquire_valid_move(self, board, move, player):
        x, y = move
        test_board = copy.deepcopy(board)
        test_board[x][y] = player
        if self.has_liberty(test_board, x, y):
            return True
        else:
            self.remove_dead_stones(test_board, 3 - player)
            if not self.has_liberty(test_board, x, y):
                return False
            else:
                # We check for Ko rule here 
                if self.previous_board is not None and test_board == self.previous_board:
                    return False
                return True

    #Separately defined this function to check if placing a stone there has any liberty(s) or not for the move x,y in the board
    def has_liberty(self, board, x, y):
        visited = set()
        dfs_check_has_liberty = self.dfs_liberty(board, x, y, board[x][y], visited)
        return dfs_check_has_liberty

    # we will recursively explore liberties using depth-first search to check for adjacent open spaces
    def dfs_liberty(self, board, x, y, player, visited):
        if (x, y) in visited:
            return False
        visited.add((x, y)) #add the visited node here
        for nx, ny in self.fetch_neighbors_at_coords(x, y):
            if board[nx][ny] == 0:
                return True
            elif board[nx][ny] == player:
                if self.dfs_liberty(board, nx, ny, player, visited):
                    return True
        return False

    # Using this function, we will remove opponent's stones that have no liberties(basically, simulating capture of opponent's stones after a move)
    def remove_dead_stones(self, board, player):
        dead_stones = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i][j] == player:
                    if not self.has_liberty(board, i, j):
                        dead_stones.append((i, j))
        for x, y in dead_stones:
            board[x][y] = 0

    # We will get a list of coordinates for all adjacent cells neighboring the given cell at that coordinate x,y
    def fetch_neighbors_at_coords(self, x, y):
        neighbors = []
        if x > 0:
            neighbors.append((x - 1, y))
        if x < self.board_size - 1:
            neighbors.append((x + 1, y))
        if y > 0:
            neighbors.append((x, y - 1))
        if y < self.board_size - 1:
            neighbors.append((x, y + 1))
        return neighbors

    # We will evaluate the board at a current situation to score a position, considering our player's stones and opponent's stones and liberties based on both
    def evaluate_board(self, board):
        score = 0
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i][j] == self.player:
                    score += 1  # Here we will give score for Stone difference to our player
                    score += self.count_liberties(board, i, j) * 0.1  # Here we will give score for Liberties to our player
                elif board[i][j] == self.opponent:
                    score -= 1 # Here we will deduct score for Stone difference from our player
                    score -= self.count_liberties(board, i, j) * 0.1 # Here we will deduct score for Liberties from our player
        return score #We will then return the score

    # This function is to count the number of liberties for a stone at a given position
    def count_liberties(self, board, x, y):
        visited = set()
        liberty_count = self.dfs_count_liberties(board, x, y, board[x][y], visited)
        return liberty_count

    # We use dfs to calculate liberties by exploring adjacent cells using this function
    def dfs_count_liberties(self, board, x, y, player, visited):
        liberties = 0
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in visited:
                continue #continuing if coords are alraedy visited
            visited.add((cx, cy)) #if not then adding the coords in visited
            for nx, ny in self.fetch_neighbors_at_coords(cx, cy): #fetching neighboring cells and checking
                if board[nx][ny] == 0: 
                    liberties += 1 #if there is no piece, then we add liberty score 1 to it
                elif board[nx][ny] == player and (nx, ny) not in visited:
                    stack.append((nx, ny))
        return liberties

    #This function is important as it's heauristic is to score a move by estimating benefits like capturing stones and proximity to board center 
    def move_heuristic(self, board, move, player):
        x, y = move
        score = 0
        # Here we are prioritizing capturing opponent stones
        test_board = copy.deepcopy(board)
        self.play_move(test_board, move, player)
        captured = self.count_captured_stones(board, test_board, 3 - player)
        score += captured * 10

        # Here we are prioritizing moves near the center
        center = self.board_size // 2
        score += 5 - (abs(x - center) + abs(y - center))

        #They both are great heuristic to our Agent and finally we send the score based on them
        return score

    # We use this function to count opponent's stones captured by comparing boards before and after a move and return the count
    def count_captured_stones(self, old_board, new_board, opponent):
        count = 0
        for i in range(self.board_size):
            for j in range(self.board_size):
                if old_board[i][j] == opponent and new_board[i][j] == 0:
                    count += 1
        return count

    # This function helps us execute a move on the board and removes any opponent's stones that has no liberties on the board
    def play_move(self, board, move, player):
        x, y = move
        board[x][y] = player
        self.remove_dead_stones(board, 3 - player)

    # We use this function to find the best move using Minimax algorithm with alpha-beta pruning
    def get_best_move(self):
        valid_moves = self.fetch_known_valid_moves(self.board, self.player)
        if not valid_moves:
            return "PASS" #Passing if there is no valid moves
        try:
            _, best_move = self.minimax(self.board, self.max_depth, float('-inf'), float('inf'), True)
            if best_move is None:
                return random.choice(valid_moves)
            return best_move
        except TimeoutError:
            # Putting this exception because the time was running out and if time runs out, we select a random move from valid moves
            return random.choice(valid_moves)

    # Finally the Minimax algorithm with alpha-beta pruning to evaluate and select the optimal or best move
    def minimax(self, board, depth, alpha, beta, maximizing_player):
        current_time = time.time()
        if current_time - self.start_time > self.time_limit - 0.1:
            raise TimeoutError
        if depth == 0:
            return self.evaluate_board(board), None

        current_player = self.player if maximizing_player else self.opponent
        valid_moves = self.fetch_known_valid_moves(board, current_player) #Fetch valid moves here
        
        if not valid_moves:
            return self.evaluate_board(board), None

        # Sort moves based on heuristic and limit to branching factor
        sorted_moves = sorted(
            valid_moves,
            key=lambda move: self.move_heuristic(board, move, self.player if maximizing_player else self.opponent),
            reverse=True
        )
        limited_moves = sorted_moves[:self.branching_factor] #we will limit the moves based on the branching factor which we provided in the init function

        best_move = None
        if maximizing_player: # we will maximize our player (alpha) here
            max_eval = float('-inf')
            for move in limited_moves:
                new_board = copy.deepcopy(board)
                self.play_move(new_board, move, self.player)
                eval, _ = self.minimax(new_board, depth - 1, alpha, beta, False)
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else: # we will minimize our opponent (beta) here
            min_eval = float('inf')
            for move in limited_moves:
                new_board = copy.deepcopy(board)
                self.play_move(new_board, move, self.opponent)
                eval, _ = self.minimax(new_board, depth - 1, alpha, beta, True)
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, best_move
        
    # This function triggers move generation: start timing, read input, find best move, write output(Kind of main function but outside main)
    def produce_next_best_move(self):
        self.start_time = time.time()
        self.read_input('input.txt')
        best_move = self.get_best_move()
        self.write_output(best_move, 'output.txt')

if __name__ == '__main__':
    awesome_agent = Go_Using_MinMax_Alpha_Beta_Pruning()
    awesome_agent.produce_next_best_move()
