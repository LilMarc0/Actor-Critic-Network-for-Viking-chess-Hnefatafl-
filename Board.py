from Pieces import *
import numpy as np
import random
from multiprocessing.pool import ThreadPool
pool = ThreadPool(processes = 4)

class Board:
    def __init__(self, board = None):
        self.turn =  ATC
        self.message = ""
        if board == None:
            self.board = {}
        else:
            self.board = board
        self.setup()
        self.kx, self.ky = (5,5) # King pos
        self.dScore = self.aScore = 24
        self.boardCode = np.zeros((11,11))
        for piece in self.board:
            if self.board[piece].color == DEF:
                self.boardCode[piece] = 1
            elif self.board[piece].color == ATC:
                self.boardCode[piece] = 2
            elif self.board[piece].color == KNG:
                self.boardCode[piece] = 3
        self.boardCode = np.reshape(self.boardCode, (121))
        self.boardCode = np.append(self.boardCode, 100 if self.turn == ATC else -100)
        self.moves = 0
        self.c2n, self.n2c = self.genDict()

    def setup(self):
        self.board[(5, 1)] = Atc(ATC, uniDict[ATC], (5, 1))
        self.board[(1, 5)] = Atc(ATC, uniDict[ATC], (1, 5))
        self.board[(9, 5)] = Atc(ATC, uniDict[ATC], (9, 5))
        self.board[(5, 9)] = Atc(ATC, uniDict[ATC], (5, 9))
        for i in range(3,8):
            self.board[(i,0)] =  Atc(ATC, uniDict[ATC], (i,0))
            self.board[(0,i)] =  Atc(ATC, uniDict[ATC], (0,i))
            self.board[(10,i)] =  Atc(ATC, uniDict[ATC], (10,i))
            self.board[(i,10)] = Atc(ATC, uniDict[ATC], (i,10))
        self.board[(3,5)] = Def(DEF, uniDict[DEF], (3,5))
        self.board[(5,3)] = Def(DEF, uniDict[DEF], (5,3))
        self.board[(7,5)] = Def(DEF, uniDict[DEF], (7,5))
        self.board[(5,7)] = Def(DEF, uniDict[DEF], (5,7))

        for i in range(4,7):
            for j in range(4,7):
                self.board[(i,j)] = Def(DEF, uniDict[DEF], (i,j))
        self.board[(5,5)] = Kng(KNG, uniDict[KNG], (5,5))

    def printBoard(self):
        print("    0| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |10")
        for i in range(0,11):
            print("-"*48)
            if i < 10:
                print(str(i)+" ",end="|")
            else:
                print(i, end="|")
            for j in range(0,11):
                item = self.board.get((i,j)," ")
                print(str(item)+' |', end = " ")
            print()
        print("-"*48)
        print("------------ {} TO MOVE -----------".format(self.turn))

    def parseInput(self):
        try:
            print('From: ')
            a1,b1 = input().split()
            print('To: ')
            a2,b2 = input().split()
            return ((int(a1), int(b1)),(int(a2), int(b2)))
        except Exception as e:
            print(e)
            print("error decoding input. please try again")

            print(a1, b1, a2, b2)
            return((-1,-1),(-1,-1))

    def genDict(self):
        nr = 0
        num2coord = {}
        coord2num = {}
        for i in range(11):
            for j in range(11):
                num2coord[nr] = (i,j)
                coord2num[(i,j)] = nr
                nr += 1
        return  coord2num, num2coord

    def isPiece(self, pos):
        if self.board.get(pos,None):
            return True
        return False

    def gameOver(self, direct = False):

        if direct:
            print('-------------- am pierdut regele ---------')
            return True
        spaces = [
            self.board.get((self.kx + 1, self.ky), None) or (self.kx + 1, self.ky),
            self.board.get((self.kx, self.ky - 1), None) or (self.kx, self.ky - 1),
            self.board.get((self.kx, self.ky + 1), None) or (self.kx, self.ky + 1),
            self.board.get((self.kx - 1, self.ky), None) or (self.kx - 1, self.ky)
        ]
        if (self.kx, self.ky) in [(0,0), (10,0), (0, 10), (10, 10)]:
            print('-------------- Rege salvat ---------')
            self.printBoard()
            return True

        if (self.kx + 1, self.ky) == ATC and\
            (self.kx, self.ky - 1) == ATC and\
            (self.kx, self.ky + 1)== ATC and\
            (self.kx - 1, self.ky) == ATC:
                print("------------ am pierdut regele -----------")
                self.printBoard()
                return True
        return False

    def checkTakes(self):
        vectors = [[(1, 0), (-1, 0)], [(0, 1), (0, -1)]]
        r = 0

        for piece in self.board:
            p = self.board.get(piece)
            px, py = piece
            if p.color == KNG:
                continue
            for direction in vectors:
                dead = True
                for v in direction:
                    xe, ye = px + v[0], py + v[1]
                    enemy = self.board.get((xe, ye), None)
                    if enemy:
                        if (enemy.color != p.color) and not (
                                enemy.color == KNG and p.color == DEF or enemy.color == DEF and p.color == KNG):
                            continue
                        else:
                            dead = False
                            break
                    else:
                        dead = False
            if dead:
                print("You killed enemy's unit on " + str(piece))
                del self.board[piece]
                self.printBoard()
                r += 50
                break
        return r

    def takeTurn(self, fr = None, to = None, pr = False):
        #print(self.message)

        r = 0
        if self.isPiece(fr):
            r+= 2
        if fr:
            fr = (round(fr[0]), round(fr[1]))
            to = (round(to[0]), round(to[1]))
        else:
            fr, to = self.parseInput()
        try:
            target = self.board.get((fr[0], fr[1]), None)
        except:
            self.message = 'Nu se afla nimic pe pozitia aleasa / in afara tablei'
            r -= 25
            target = None

        if target:
            if target.color == ATC and self.turn == DEF or\
                target.color == DEF and self.turn == ATC or\
                target.color == KNG and self.turn == ATC:
                self.message = "you aren't allowed to move that piece this turn"
                return self.gameOver(), self.boardCode, -25
            if target.isValid(fr, to, target.color, self.board):
                self.message = "that is a valid move"
                r += 10
                self.board[to] = self.board[fr]
                self.board[to].pos = to

                if target.color == KNG:
                    self.kx, self.ky = target.pos
                    r += 5
                #print(self.boardCode)
                del self.board[fr]
                if self.turn == DEF:
                    self.turn = ATC
                    self.boardCode[-1] = 100
                else:
                    self.turn = DEF
                    self.boardCode[-1] = -100
                if pr:
                    self.moves += 1
                    print('MOVES {}: '.format((fr,to)), self.moves)
                    self.printBoard()

                self.boardCode = np.zeros((11, 11))
                for piece in self.board:
                    if self.board[piece].color == DEF:
                        self.boardCode[piece] = 1
                    elif self.board[piece].color == ATC:
                        self.boardCode[piece] = 2
                    elif self.board[piece].color == KNG:
                        self.boardCode[piece] = 3
                self.boardCode = np.reshape(self.boardCode, (121))
                self.boardCode = np.append(self.boardCode, 100 if self.turn == ATC else -100)
                #print(self.boardCode)

            else:
                self.message = "invalid move" + str(target.avMoves(fr[0], fr[1], self.board))
                return self.gameOver(), self.boardCode, -25
        else:
            self.message = "there is no piece in that space"
            r -= 25
        r += self.checkTakes()
        if self.gameOver():
            r += 1000
        return self.gameOver(), self.boardCode, r

    def randomMove(self):
        x = [self.board.get(x) for x in self.board if self.board.get(x).color == self.turn and len(self.board.get(x).avMoves(self.board.get(x).pos[0], self.board.get(x).pos[1], self.board)) != 0]

        if self.turn == DEF:
            xx = self.board.get((self.kx, self.ky)).avMoves(self.kx, self.ky, self.board)
            if len(xx) != 0:
                x.append(self.board.get((self.kx, self.ky)))
                #print(x)
        a1 = random.choice(x)
        a2 = random.choice(a1.avMoves(a1.pos[0], a1.pos[1], self.board))

        return [a1.pos[0], a1.pos[1], a2[0], a2[1]]

if __name__ == '__main__':


    b = Board()
    while True:
        b = Board()
        while not b.gameOver():
            a = b.parseInput()
            b.takeTurn(a[0], a[1])
            b.printBoard()