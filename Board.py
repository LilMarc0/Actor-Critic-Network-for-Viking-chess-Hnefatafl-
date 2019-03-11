from Pieces import *
import numpy as np
import random
from multiprocessing.pool import ThreadPool
import pickle


pool = ThreadPool(processes=4)
from collections import deque

move_dict = pickle.load(open('moves7x7.pickle', 'rb'))
rev_move_dict = {v:k for k,v in move_dict.items()}

class Board:
    def __init__(self, dim=7,board=None):
        self.dim = dim
        self.turn = ATC
        self.message = ""
        self.won = None
        if board == None:
            self.board = {}
        else:
            self.board = board
        self.setup()
        self.kx, self.ky = (dim//2, dim//2)  # King pos
        self.boardCode = np.zeros((dim, dim))

        self.boardMatrix = [[0 for _ in range(dim)] for __ in range(dim)]

        for piece in self.board:
            if self.board[piece].color == DEF:
                self.boardCode[piece] = 1
                self.boardMatrix[piece[0]][piece[1]] = 1
            elif self.board[piece].color == ATC:
                self.boardCode[piece] = -1
                self.boardMatrix[piece[0]][piece[1]] = -1
            elif self.board[piece].color == KNG:
                self.boardCode[piece] = 2
                self.boardMatrix[piece[0]][piece[1]] = 2
        self.boardCode = np.reshape(self.boardCode, (dim**2))
        self.boardCode = np.append(self.boardCode, 100 if self.turn == ATC else -100)

        self.moves = 0
        self.c2n, self.n2c = self.genDict()
        self.n2c[dim**2] = (dim-1, dim-1)
        self.n2c[dim**2+1] = (dim-1, dim-1)

    def setup(self):
        self.board[(3, 1)] = Atc(ATC, uniDict[ATC], (3, 1))
        self.board[(1, 3)] = Atc(ATC, uniDict[ATC], (1, 3))
        self.board[(5, 3)] = Atc(ATC, uniDict[ATC], (5, 3))
        self.board[(3, 5)] = Atc(ATC, uniDict[ATC], (3, 5))

        self.board[(3, 0)] = Atc(ATC, uniDict[ATC], (3, 0))
        self.board[(0, 3)] = Atc(ATC, uniDict[ATC], (0, 3))
        self.board[(6, 3)] = Atc(ATC, uniDict[ATC], (6, 3))
        self.board[(3, 6)] = Atc(ATC, uniDict[ATC], (3, 6))

        self.board[(2, 3)] = Def(DEF, uniDict[DEF], (2, 3))
        self.board[(3, 2)] = Def(DEF, uniDict[DEF], (3, 2))
        self.board[(3, 4)] = Def(DEF, uniDict[DEF], (3, 4))
        self.board[(4, 3)] = Def(DEF, uniDict[DEF], (4, 3))

        # for i in range(3, 6):
        #     self.board[(i, 0)] = Atc(ATC, uniDict[ATC], (i, 0))
        #     self.board[(0, i)] = Atc(ATC, uniDict[ATC], (0, i))
        #     self.board[(8, i)] = Atc(ATC, uniDict[ATC], (8, i))
        #     self.board[(i, 8)] = Atc(ATC, uniDict[ATC], (i, 8))
        #
        # for i in range(2, 7):
        #     for j in range(2, 7):
        #         if i == 4 or j == 4:
        #             self.board[(i, j)] = Def(DEF, uniDict[DEF], (i, j))
        self.board[(self.dim//2, self.dim//2)] = Kng(KNG, uniDict[KNG], (self.dim//2, self.dim//2))

    def genFlips(self, omem, mem, fx, type='matrix'):
        if type == 'matrix':

            mems = []
            omems = []
            for i in range(3):
                t_mem = deque(maxlen=5)
                t_omem = deque(maxlen=5)
                for _ in range(5):
                   t_mem.append(mem[_])
                   t_omem.append(omem[_])
                for m in mem:
                    t_m = m[:]
                    if i == 0:
                        t_m = np.flip(m)
                        t_mem.append(t_m)
                    elif i == 1:
                        t_m = np.flip(m, 1)
                        t_mem.append(t_m)
                    elif i==2:
                        t_m = np.flip(np.flip(m, 1))
                        t_mem.append(t_m)
                for m in omem:
                    t_m = m[:]
                    if i == 0:
                        t_m = np.flip(m)
                        t_omem.append(t_m)
                    elif i == 1:
                        t_m = np.flip(m, 1)
                        t_omem.append(t_m)
                    elif i==2:
                        t_m = np.flip(np.flip(m, 1))
                        t_omem.append(t_m)
                mems.append(t_mem)
                omems.append(t_omem)
            coord1, coord2 = self.n2c[fx[0]], self.n2c[fx[1]]
            moves = [
                [self.c2n[coord1[0], self.dim-1-coord1[1]], self.c2n[coord2[0], self.dim-1-coord2[1]]],
                [self.c2n[self.dim-1-coord1[0], coord1[1]], self.c2n[self.dim-1 - coord2[0], coord2[1]]],
                [self.c2n[self.dim-1-coord1[0], self.dim-1-coord1[1]], self.c2n[self.dim-1-coord2[0], self.dim-1-coord2[1]]]
            ]
            moves2 = []
            for i in range(3):
                moves2.append(rev_move_dict[tuple(moves[i])])
            moves360 = [np.zeros(588) for _ in range(3)]
            for i in range(3):
                moves360[i][moves2[i]] = 1
            return omems, mems, moves360

    def printBoard(self):
        print("    0| 1 | 2 | 3 | 4 | 5 | 6")
        for i in range(0, self.dim):
            print("-" * 30)
            if i < 10:
                print(str(i) + " ", end="|")
            else:
                print(i, end="|")
            for j in range(0, self.dim):
                item = self.board.get((i, j), " ")
                print(str(item) + ' |', end=" ")
            print()
        print("-" * 30)
        print("------------ {} TO MOVE -----------".format(self.turn))

    def parseInput(self):
        try:
            print('From: ')
            a1, b1 = input().split()
            print('To: ')
            a2, b2 = input().split()
            return ((int(a1), int(b1)), (int(a2), int(b2)))
        except Exception as e:
            print(e)
            print("error decoding input. please try again")

            print(a1, b1, a2, b2)
            return ((-1, -1), (-1, -1))

    def genDict(self):
        nr = 0
        num2coord = {}
        coord2num = {}
        for i in range(self.dim):
            for j in range(self.dim):
                num2coord[nr] = (i, j)
                coord2num[(i, j)] = nr
                nr += 1
        #num2coord[self.dim + 1] = (self.dim, self.dim)
        #num2coord[self.dim + 2] = (self.dim, self.dim)
        #num2coord[self.dim + 1] = (self.dim, self.dim)
        #num2coord[self.dim + 1] = (self.dim, self.dim)
        return coord2num, num2coord

    def isPiece(self, pos):
        if self.board.get(pos, None):
            return True
        return False

    def gameOver(self, direct=False):

        if direct:
            print('-------------- am pierdut regele ---------')
            return True
        spaces = [
            self.board.get((self.kx + 1, self.ky), None) or (self.kx + 1, self.ky),
            self.board.get((self.kx, self.ky - 1), None) or (self.kx, self.ky - 1),
            self.board.get((self.kx, self.ky + 1), None) or (self.kx, self.ky + 1),
            self.board.get((self.kx - 1, self.ky), None) or (self.kx - 1, self.ky)
        ]
        if (self.kx, self.ky) in [(0, 0), (0, self.dim-1), (self.dim-1, 0), (self.dim-1, self.dim-1)]:
            print('\033[93m-------------- Rege salvat ---------\033[0m')
            self.printBoard()
            self.won = DEF
            return True

        if self.board.get((self.kx + 1, self.ky), None) == ATC and \
                self.board.get((self.kx, self.ky - 1), None) == ATC and \
                self.board.get((self.kx, self.ky + 1), None) == ATC and \
                self.board.get((self.kx - 1, self.ky), None) == ATC:
            print("\033[93m------------ am pierdut regele -----------\033[0m")
            self.printBoard()
            self.won = ATC
            return True
        return False

    def checkTakes(self, p):
        vectors = [(0,1), (0,-1), (1,0), (-1, 0)]
        r = 0
        px, py = p.pos
        for direction in vectors:
            near = self.board.get((px + direction[0], py + direction[1]), None)
            if near:
                #print('are vecin {} la {}'.format(near.color, near.pos))
                if p.color == KNG and near.color == ATC or p.color == DEF and near.color == ATC or p.color == ATC and near.color == DEF:
                    nearAlly = self.board.get((near.pos[0] + direction[0], near.pos[1] + direction[1]), None)
                    if nearAlly:
                        #print('care are vecin {} {}'.format(nearAlly.color, nearAlly.pos))
                        if p.color == KNG and nearAlly.color == DEF or p.color == DEF and nearAlly.color == KNG or p.color == ATC and nearAlly.color == ATC or p.color == DEF and nearAlly.color == DEF:
                            print('\033[92m' + "{} killed enemy's unit on ".format(self.turn) + str(near.pos) + '------------------------------------------\033[0m')
                            r += 15
                            del self.board[near.pos]
        return r

    def takeTurn(self, fr=None, to=None, pr=False):
        # print(self.message)

        r = 0

        if fr == to:
            return self.gameOver(), self.boardMatrix, -2

        if fr:
            fr = (round(fr[0]), round(fr[1]))
            to = (round(to[0]), round(to[1]))
        else:
            fr, to = self.parseInput()
        try:
            target = self.board.get((fr[0], fr[1]), None)
        except:
            self.message = 'Nu se afla nimic pe pozitia aleasa / in afara tablei'
            r -= 2
            target = None

        if target:
            if target.color == ATC and self.turn == DEF or \
                    target.color == DEF and self.turn == ATC or \
                    target.color == KNG and self.turn == ATC:
                self.message = "you aren't allowed to move that piece this turn"
                return self.gameOver(), self.boardMatrix, -2
            if target.isValid(fr, to, target.color, self.board):
                self.message = "that is a valid move"
                self.board[to] = self.board[fr]
                self.board[to].pos = to

                # mut regele
                if target.color == KNG:
                    self.kx, self.ky = target.pos
                    if to[0] not in range(self.dim) and to[1] not in range(self.dim): # din centru
                        if target.pos[0] in range(self.dim) or target.pos[1] in range(self.dim): # pe margine
                            r += 2
                r += self.checkTakes(target)
                tpos = target.pos
                # mut langa rege
                if self.turn == ATC:
                    v = [(tpos[0] - 1, tpos[1]), (tpos[0] + 1, tpos[1]), (tpos[0], tpos[1] - 1), (tpos[0], tpos[1] + 1)]
                    for _ in v:
                        if isBound(_[0], _[1]):
                            vv = self.board.get(_, None)
                            if vv:
                                if vv.color == KNG:
                                    r += 2
                # print(self.boardCode)
                del self.board[fr]
                self.boardMatrix[fr[0]][fr[1]] = 0

                if self.turn == DEF:
                    self.turn = ATC
                    if target.look == DEF:
                        self.boardMatrix[to[0]][to[1]] = 1
                    else:
                        self.boardMatrix[to[0]][to[1]] = 2

                    self.boardCode[-1] = 100
                else:
                    self.turn = DEF
                    self.boardMatrix[to[0]][to[1]] = -1
                    self.boardCode[-1] = -100
                if pr:
                    self.moves += 1
                    print('\033[92m' + 'MOVES {}: \033[0m'.format((fr, to)), self.moves)
                    self.printBoard()
            else:
                self.message = "invalid move" + str(target.avMoves(fr[0], fr[1], self.board))
                return self.gameOver(), self.boardMatrix, -2
        else:
            self.message = "there is no piece in that space"
            r -= 10

        terminal = self.gameOver()
        return terminal, self.boardMatrix, r

    def randomMove(self):
        x = [self.board.get(x) for x in self.board if self.board.get(x).color == self.turn and len(
            self.board.get(x).avMoves(self.board.get(x).pos[0], self.board.get(x).pos[1], self.board)) != 0]
        
        king = self.board.get((self.kx, self.ky))
        km = king.avMoves(self.kx, self.ky, self.board)
        wm = [(0, 0), (self.dim-1, 0), (self.dim-1, 0), (self.dim-1, self.dim-1)]

        if king in x:
            for m in wm:
                if m in km:
                    return [self.kx,  self.ky, m[0], m[1]]
        
        a1 = random.choice(x)
        a2 = random.choice(a1.avMoves(a1.pos[0], a1.pos[1], self.board))
        if self.turn == DEF:
            xx = km
            if len(xx) != 0:
                a1 = king
                xxx = [km[i] for i in range(len(km)) if (km[i][1] == self.dim-1) or (km[i][0] == self.dim-1) or (km[i][1] == 0) or (km[i][0] == 0)]
                if len(xxx) != 0:
                    a2 = random.    choice(xxx)
                # print(x)
        a2 = random.choice(a1.avMoves(a1.pos[0], a1.pos[1], self.board))

        return [a1.pos[0], a1.pos[1], a2[0], a2[1]]


if __name__ == '__main__':
    b = Board()
    while True:
        b = Board()
        b.printBoard()
        q = deque(maxlen=5)
        fx = deque(maxlen=5)
        for i in range(5):
            q.append(b.boardMatrix)
        while not b.gameOver():
            a = b.parseInput()
            b.takeTurn(a[0], a[1])
            q.append(b.boardMatrix)
            p1, p2, fx2 = b.genFlips(q, q, (b.c2n[a[0]], b.c2n[a[1]]))
            #print(p1[-1], move_dict[np.argmax(fx2[0])], move_dict[np.argmax(fx2[1])], move_dict[np.argmax(fx2[2])])
