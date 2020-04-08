import gym
from gym.envs.toy_text import discrete
import sys
from contextlib import closing
import numpy as np
from six import StringIO


class ForestEnv(discrete.DiscreteEnv):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, nS=50, r1=40, r2=10, p=0.1):
        nA = 2
        P = {s : {a : [] for a in range(nA)} for s in range(nS)}
        
        isd = np.zeros(nS)
        isd[0] = 1.0
        
        for s in range(nS):
            for a in range(nA):
                li = P[s][a]
                if a == 0:# wait
                    if s == nS-1:
                        li.append((p, 0, 0, False))
                        li.append((1-p, s, r1, False))
                    else:
                        li.append((p, 0, 0, False))
                        li.append((1-p, s+1, 0, False))
                else:# cut
                    if s == nS-1:
                        li.append((1.0, 0, r2, False))
                    elif s == 0:
                        li.append((p, 0, 0, False))
                        li.append((1-p, 0, 0, False))
                    else:
                        li.append((p, 0, 0, False))
                        li.append((1-p, 0, (r2*0.5)*(float(s)/nS), False))

        super(ForestEnv, self).__init__(nS, nA, P, isd)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Wait","Cut"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n"+ str(self.s)+"\n")

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()