import tensorflow as tf
import tflearn
import sys
import time
from AdversarialNets.atk import *
from AdversarialNets.deff import *
from Board import *
import os

from collections import deque
import pickle
from Pieces import *
from random import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)

move_dict = pickle.load(open('moves7x7.pickle', 'rb'))
atkBufferPath = './AdversarialNets/Buffers/atkBuffer.pickle'
defBufferPath = './AdversarialNets/Buffers/defBuffer.pickle'

criticLR = 0.0005
actorLR = 0.005
criticTAU = 0.1
actorTAU = 0.1
GAMMA = 0.85
BATCH_SIZE = 50
minibatchSize = 50
actiondim = 588

defWins = 0
atcWins = 0


def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

def play(sess, dactor, dcritic):
    

    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('summary', sess.graph)

    # Initialize target network weights
    dactor.update_target_network()
    dcritic.update_target_network()

    # Initialize replay memory
    dbuff = ReplayBuffer(500)

    if(os.path.isfile(defBufferPath)):
        print('Loading def buffer...')
        abuff.load(defBufferPath)
    if(os.path.isfile(atkBufferPath)):
        print('Loading atk buffer...')
        dbuff.load(atkBufferPath)
    tflearn.is_training(True)
    
    total_moves = 0
    iterations = 0
    amovhist = deque(maxlen=5)
    dmovhist = deque(maxlen=5)

    for i in range(500):
        b = Board()
        s = b.boardMatrix
        ep_reward = 0
        ep_ave_max_q = 0
        j = 0
        last_time = time.time()
        moves = 0
        
        if i != 0:
            print('----- LAST BOARD --- ')
            lastboard.printBoard()
        lastboard = b
        
        mem = deque(maxlen=5)
        for _ in range(5):
            mem.append(s)
            
        while not b.gameOver():
            moves = b.moves
            mem2 = deque(maxlen=5)
            for _ in mem:
                mem2.append(_)

            j += 1
            iterations += 1

            # Guess the move
            if b.turn == ATC:
                b.printBoard()
                a = b.parseInput()
                _, s2, _ = b.takeTurn(a[0], a[1])
                mem2.append(s2)

                
                ############# DEF
            else:
                print('Ded Actor is thinking...')
                CURSOR_UP_ONE = '\x1b[1A'
                ERASE_LINE = '\x1b[2K'
                print(CURSOR_UP_ONE + ERASE_LINE)
                
                a = dactor.predict(np.reshape(mem, (1, 5, 7, 7)))[0]
                move = move_dict[np.argmax(a)]
                moveFrom = b.n2c[move[0]]
                moveTo = b.n2c[move[1]]

                # Make the move
                target = b.board.get(b.n2c[move[0]], None)
                while not target:
                    if target:
                        if target.isValid(moveFrom, moveTo, target.color, b.board):
                            a = dactor.predict(np.reshape(mem, (1, 5, 7, 7)))[0]
                            move = move_dict[np.argmax(a)]
                            moveFrom = b.n2c[move[0]]
                            moveTo = b.n2c[move[1]]
                            target = b.board.get(b.n2c[move[0]], None)
                        else:
                            target = None
                            continue
                    else:
                        a = dactor.predict(np.reshape(mem, (1, 5, 7, 7)))[0]
                        move = move_dict[np.argmax(a)]
                        moveFrom = b.n2c[move[0]]
                        moveTo = b.n2c[move[1]]
                        target = b.board.get(b.n2c[move[0]], None)
                        
                terminal, s2, r = b.takeTurn(moveFrom, moveTo, True)
                mem2.append(s2)

                for _ in mem2:
                    mem.append(_)
                
        print('\033[92m' + 'Saving buffer \033[0m')
        print('----- \033[92m EPISODE {}, TOTAL MOVES: {}, LAST EPISODE: {} TIME ELAPSED: {} ATC/DEF: {}|{}\033[0m -----'.format(i + 1, total_moves,
                                                                                                  moves, time.time() - last_time), atcWins, defWins)
    return aactor, acritic, dactor, dcritic



with tf.Session() as sess:
    Dactor = defActorNetwork(sess, [None, 5, 7, 7], actiondim, 1, actorLR, actorTAU, BATCH_SIZE)
    Dcritic = defCriticNetwork(sess, [None, 5, 7, 7], actiondim, criticLR, criticTAU, GAMMA, Dactor.get_num_trainable_vars())
    if (os.path.isfile('./AdversarialNets/Dactor/Dactor.ckpt.index')):
        Dactor.load()
    if (os.path.isfile('./AdversarialNets/Dcritic/Dcritic.ckpt.index')):
        Dcritic.load()
    asses = tf.Session()
    play(sess, Dactor, Dcritic)
