import unittest
from agent import LearningAgent
from environment import Environment

class AgentTestCase(unittest.TestCase):
    def test_get_maxQ(self):
        """ Does it get the maxQ from state"""
        env = Environment()
        agent = env.create_agent(LearningAgent)
        state = (0,0,0)
        agent.Q[state] = dict((action, score) for (action, score) in zip(agent.valid_actions, [0.0,1.0,0.3,1.3]))
        self.assertTrue(agent.get_maxQ(state) == 1.3)

    def test_get_argMaxQ(self):
        """ Does it get the maxQ from state"""
        env = Environment()
        agent = env.create_agent(LearningAgent)
        state = (0,0,0)
        valid_actions = [None, 'forward', 'left', 'right']
        agent.Q[state] = dict((action, score) for (action, score) in zip(valid_actions, [0.0,1.0,0.3,1.3]))
        self.assertTrue(agent.get_argMaxQ(state) == "right")

if __name__ == '__main__':
    unittest.main()
