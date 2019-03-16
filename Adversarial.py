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
BATCH_SIZE = 250
minibatchSize = 250
actiondim = 588
illegalTh = 0.99



def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

def train(sess, env, aactor, acritic, dactor, dcritic, actor_noise):

    defWins = 0
    atcWins = 0
    episodeLen = 5000
    
    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('summary', sess.graph)

    # Initialize target network weights
    aactor.update_target_network()
    acritic.update_target_network()
    dactor.update_target_network()
    dcritic.update_target_network()

    # Initialize replay memory
    abuff = ReplayBuffer(2500)
    dbuff = ReplayBuffer(2500)

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

        if i > 2:
            aactor.save()
            acritic.save()
            dactor.save()
            dcritic.save()
        
        if (os.path.isfile(defBufferPath)):
            abuff.save(atkBufferPath)
        if (os.path.isfile(atkBufferPath)):
            dbuff.save(defBufferPath)
        if i != 0:
            print('----- LAST BOARD --- ')
            lastboard.printBoard()
        lastboard = b
        
        mem = deque(maxlen=5)
        for _ in range(5):
            mem.append(s)
        
        print(Aactor.network_params[1])
        while not b.gameOver():
            #print(len(abuff.buffer))
            moves = b.moves
            mem2 = deque(maxlen=5)
            for _ in mem:
                mem2.append(_)

            j += 1
            iterations += 1

            # Guess the move
            if b.turn == ATC:

                a = aactor.predict(np.reshape(mem, (1, 5, 7, 7)))[0] + actor_noise()
                move = move_dict[np.argmax(a)]
                moveFrom = b.n2c[move[0]]
                moveTo = b.n2c[move[1]]

                # Make the move
                target = b.board.get(b.n2c[move[0]], None)
                if target:
                    if target.isValid(moveFrom, moveTo, target.color, b.board):
                        terminal, s2, r = b.takeTurn(moveFrom, moveTo, True)
                        amovhist.append(move)
                        mem2.append(s2)
                    else:
                        if random() > illegalTh:
                            terminal, s2, r = b.takeTurn(moveFrom, moveTo, True)
                            amovhist.append(move)
                            mem2.append(s2)
                        else:
                            j-=1
                            continue
                else:
                    if random() > illegalTh:
                        terminal, s2, r = b.takeTurn(moveFrom, moveTo, True)
                        amovhist.append(move)
                        mem2.append(s2)
                    else:
                        j -= 1
                        continue

                if len(amovhist) >= 3:
                    if amovhist[-1] == amovhist[-2] == amovhist[-3]:
                        r-=4
                        #print('--rep punish')

                if iterations % 500 == 0 and iterations != 0:
                    print(move)

                abuff.add(np.reshape(mem, (5, 7, 7)), np.reshape(a, (588)), r, terminal, np.reshape(mem2, (5, 7, 7)))
                rot, rot2, fx = b.genFlips(mem, mem2, move)

                # rotatii
                for _ in range(3):
                    abuff.add(np.reshape(rot[_], (5, 7, 7)),
                                      np.reshape(fx[_], (588)),
                                      r,
                                      terminal,
                                      np.reshape(rot2[_], (5, 7, 7)))

                if abuff.size() > minibatchSize:
                    mem_batch, a_batch, r_batch, t_batch, mem2_batch = \
                        abuff.sample_batch(minibatchSize)

                    # Calculate targets
                    target_q = acritic.predict_target(
                        mem2_batch,
                        aactor.predict_target(mem2_batch))

                    y_i = []
                    for k in range(minibatchSize):
                        if t_batch[k]:
                            y_i.append(r_batch[k])
                        else:
                            y_i.append(r_batch[k] + acritic.gamma * target_q[k])

                    # Update the critic given the targets
                    predicted_q_value, _ = acritic.train(mem_batch, a_batch, np.reshape(y_i, (minibatchSize, 1)))  # (minibatchSize, 1)

                    ep_ave_max_q += np.amax(predicted_q_value)

                    # Update the actor policy using the sampled gradient
                    a_outs = aactor.predict(mem_batch)
                    grads = acritic.action_gradients(mem_batch, a_outs)
                    aactor.train(mem_batch, grads[0])

                    # Update target networks
                    aactor.update_target_network()
                    acritic.update_target_network()

                    for _ in mem2:
                        mem.append(_)
                    ep_reward += r
                    lastboard = b

                    # reset
                    if j >= episodeLen or terminal:
                        if b.won == b.turn:
                            r += 1000
                            atcWins += 1
                        else:
                            r -= 1000
                        total_moves += moves
                        summary_str = sess.run(summary_ops, feed_dict={
                            summary_vars[0]: ep_reward,
                            summary_vars[1]: ep_ave_max_q / float(j + 1)
                        })

                        writer.add_summary(summary_str, i)
                        writer.flush()

                        print(
                            '\033[92m' + '| Reward: {:d} | Episode: {:d} | Qmax: {:.4f} \033[0m'.format(int(ep_reward), \
                                                                                                        i, (ep_ave_max_q / float(
                                                                                                            j + 1))))
                        break

                ############# DEF
            else:

                a = dactor.predict(np.reshape(mem, (1, 5, 7, 7)))[0] + actor_noise()
                move = move_dict[np.argmax(a)]
                moveFrom = b.n2c[move[0]]
                moveTo = b.n2c[move[1]]

                # Make the move
                target = b.board.get(b.n2c[move[0]], None)
                if target:
                    if target.isValid(moveFrom, moveTo, target.color, b.board):
                        terminal, s2, r = b.takeTurn(moveFrom, moveTo, True)
                        dmovhist.append(move)
                        mem2.append(s2)
                    else:
                        if random() > illegalTh:
                            terminal, s2, r = b.takeTurn(moveFrom, moveTo, True)
                            dmovhist.append(move)
                            mem2.append(s2)
                        else:
                            j -= 1
                            continue
                else:
                    if random() > illegalTh:
                        terminal, s2, r = b.takeTurn(moveFrom, moveTo, True)
                        dmovhist.append(move)
                        mem2.append(s2)
                    else:
                        j -= 1
                        continue
                if len(dmovhist) >= 3:
                    if dmovhist[-1] == dmovhist[-2] == dmovhist[-3]:
                        r -= 4
                        # print('--rep punish')

                if iterations % 500 == 0 and iterations != 0:
                    print(move)
                # if iterations % 250 == 0 and iterations != 0:
                #    print(time() - last_time)

                dbuff.add(np.reshape(mem, (5, 7, 7)), np.reshape(a, (588)), r, terminal,
                          np.reshape(mem2, (5, 7, 7)))
                rot, rot2, fx = b.genFlips(mem, mem2, move)

                # rotatii
                for _ in range(3):
                    dbuff.add(np.reshape(rot[_], (5, 7, 7)),
                              np.reshape(fx[_], (588)),
                              r,
                              terminal,
                              np.reshape(rot2[_], (5, 7, 7)))

                if dbuff.size() > minibatchSize:
                    mem_batch, a_batch, r_batch, t_batch, mem2_batch = dbuff.sample_batch(minibatchSize)

                    # Calculate targets
                    target_q = dcritic.predict_target(
                        mem2_batch,
                        dactor.predict_target(mem2_batch))

                    y_i = []
                    for k in range(minibatchSize):
                        if t_batch[k]:
                            y_i.append(r_batch[k])
                        else:
                            y_i.append(r_batch[k] + dcritic.gamma * target_q[k])

                    # Update the critic given the targets
                    predicted_q_value, _ = dcritic.train(mem_batch, a_batch, np.reshape(y_i, (minibatchSize, 1)))  # (minibatchSize, 1)

                    ep_ave_max_q += np.amax(predicted_q_value)

                    # Update the actor policy using the sampled gradient
                    a_outs = dactor.predict(mem_batch)
                    grads = dcritic.action_gradients(mem_batch, a_outs)
                    dactor.train(mem_batch, grads[0])

                    # Update target networks
                    dactor.update_target_network()
                    dcritic.update_target_network()

                    for _ in mem2:
                        mem.append(_)
                    ep_reward += r
                    lastboard = b

                    # reset
                    if j >= episodeLen or terminal:
                        if b.won == b.turn:
                            r += 1000
                            defWins += 1
                        else:
                            r -= 1000
                        total_moves += moves
                        summary_str = sess.run(summary_ops, feed_dict={
                            summary_vars[0]: ep_reward,
                            summary_vars[1]: ep_ave_max_q / float(j + 1)
                        })

                        writer.add_summary(summary_str, i)
                        writer.flush()

                        print('\033[92m' + '| Reward: {:d} | Episode: {:d} | Qmax: {:.4f} \033[0m'.format(int(ep_reward), \
                                                                                                          i, (ep_ave_max_q / float(
                                                                                                                  j + 1))))
                        break
        print('\033[92m' + 'Saving buffer \033[0m')
        print('----- \033[92m EPISODE {}, TOTAL MOVES: {}, LAST EPISODE: {} TIME ELAPSED: {} ATC/DEF: {}|{}\033[0m -----'.format(i + 1, total_moves,
                                                                                                  moves, time.time() - last_time, atcWins, defWins))
    return aactor, acritic, dactor, dcritic


with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    aenv = Board()
    Aactor = atkActorNetwork(sess, [None, 5, 7, 7], actiondim, 1, actorLR, actorTAU, BATCH_SIZE)
    Acritic = atkCriticNetwork(sess, [None, 5, 7, 7], actiondim, criticLR, criticTAU, GAMMA, Aactor.get_num_trainable_vars())
    Dactor = defActorNetwork(sess, [None, 5, 7, 7], actiondim, 1, actorLR, actorTAU, BATCH_SIZE)
    Dcritic = defCriticNetwork(sess, [None, 5, 7, 7], actiondim, criticLR, criticTAU, GAMMA, Dactor.get_num_trainable_vars())
    aactor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(588))
    if(os.path.isfile('.\\AdversarialNets\\Aactor\\Aactor.ckpt.meta')):
        Aactor.load()
    if (os.path.isfile('.\\AdversarialNets\\Acritic\\Acritic.ckpt.meta')):
        Acritic.load()
    if (os.path.isfile('.\\AdversarialNets\\Dactor\\Dactor.ckpt.meta')):
        Dactor.load()
    if (os.path.isfile('.\\AdversarialNets\\Dcritic\\Dcritic.ckpt.meta')):
        Dcritic.load()
    aactor_noise = OrnsteinUhlenbeckActionNoise(mu = np.zeros(588))

    Aactor, Acritic, Dactor, Dcritic = train(sess, aenv, Aactor, Acritic, Dactor, Dcritic, aactor_noise)


