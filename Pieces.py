DEF = "defenders"
ATC = "attackers"
KNG = "king"
vectors = [(1,0),(0,1),(-1,0),(0,-1)]
uniDict = {DEF : "D", ATC : "A", KNG: "K"}


def isBound(x, y):
    if x >= 0 and x < 11 and y >= 0 and y < 11:
        return True
    return False

class Piece:

    def __init__(self, color, look, pos):
        self.pos = pos
        self.color = color
        self.look = look

    def __repr__(self):
        return self.look

    def __str__(self):
        return self.look

    def avMoves(self, x, y, gameboard):
        print("CLASA DE BAZA - NU ARE MUTARI")

    def isValid(self, startpos, endpos, Color, gameboard):
        if endpos in self.avMoves(startpos[0], startpos[1], gameboard, Color=Color):
            return True
        return False

    def slide(self, x, y, gameboard, Color, vectors):
        pos = []
        for xv, yv in vectors:
            xtemp, ytemp = x + xv, y + yv

            while isBound(xtemp, ytemp):
                target = gameboard.get((xtemp, ytemp), None)
                if target is None:
                    pos.append((xtemp, ytemp))
                else:
                    break
                xtemp, ytemp = xtemp + xv, ytemp + yv
        return pos


class Def(Piece):
    def avMoves(self, x, y, gameboard, Color=None):
        if Color is None:
            Color = self.color
        return self.slide(x, y, gameboard, Color, vectors)


class Atc(Piece):
    def avMoves(self, x, y, gameboard, Color=None):
        if Color is None:
            Color = self.color
        return self.slide(x, y, gameboard, Color, vectors)


class Kng(Piece):
    def avMoves(self, x, y, gameboard, Color=None):
        if Color is None:
            Color = self.color
        return self.slide(x, y, gameboard, Color, vectors)