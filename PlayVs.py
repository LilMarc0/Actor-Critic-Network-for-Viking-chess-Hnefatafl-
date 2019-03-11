import tensorflow as tf
import tflearn
import sys
import time
from actorcriticmoves import *
from Board import *
import os

from collections import deque
import pickle
from Pieces import *
from random import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)

move_dict = pickle.load(open('moves7x7.pickle', 'rb'))
actiondim = {0,1}

def play(sess, actor):
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('summary', sess.graph)  # summary_dir

    # Initialize target network weights
    actor.update_target_network()

    b = Board()
    s = b.boardMatrix
    mem = deque(maxlen=5)
    for _ in range(5):
        mem.append(s)
    while True:
        print('----NEW GAME----')
        while not b.gameOver():
            b.printBoard()
            if b.turn == DEF:
                mem2 = deque(maxlen=5)
                for _ in mem:
                    mem2.append(_)
                # Guess the move
                a = actor.predict(np.reshape(mem, (1, 5, 7, 7)))[0]
                move = move_dict[np.argmax(a)]
                moveFrom = b.n2c[move[0]]
                moveTo = b.n2c[move[1]]
                target = b.board.get(b.n2c[move[0]], None)
                if target:
                    if target.isValid(moveFrom, moveTo, target.color, b.board):
                        terminal, s2, r = b.takeTurn(moveFrom,
                                                     moveTo,
                                                     True)
                        mem2.append(s2)
                while target == None:
                    a = actor.predict(np.reshape(mem, (1, 5, 7, 7)))[0]
                    move = move_dict[np.argmax(a)]
                    moveFrom = b.n2c[move[0]]
                    moveTo = b.n2c[move[1]]
                    target = b.board.get(b.n2c[move[0]], None)

                    print(moveFrom, moveTo)
                    if target:
                        if target.isValid(moveFrom, moveTo, target.color, b.board):
                            terminal, s2, r = b.takeTurn(moveFrom,
                                                         moveTo,
                                                         True)
                            mem2.append(s2)

                        else:
                            target = None
                            continue
                b.printBoard()
            else:
                a = b.parseInput()
                print(b.takeTurn(a[0], a[1])[0])
                mem2 =
                b.printBoard()


with tf.Session() as sess:
    aactor = ActorNetwork(sess, [None, 5, 7, 7], actiondim, 1, 0.005, 0.05, 25)
    if(os.path.isfile('./actorF/actorF.ckpt.index')):
        aactor.load()
    asses = tf.Session()
    play(sess, aactor)