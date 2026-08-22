from simulation.environment import DynamicRoutingEnv
from rl.agent import DoubleDQNAgent
from routing.baselines import shortest_path, ecmp_path


def test_state_and_masked_actions():
    env=DynamicRoutingEnv(seed=1,nodes=15); s=env.reset(0,14)
    assert s.shape[0]==env.STATE_SIZE
    assert set(env.valid_actions()).issubset(set(env.graph.neighbors(env.current)))


def test_failure_is_unusable():
    env=DynamicRoutingEnv(seed=2,nodes=15); env.reset(0,14)
    edge=next(iter(env.graph.edges())); env.set_link_failure(*edge,down=True)
    assert env.get_link(*edge).availability==0
    assert edge[1] not in env.valid_actions() or edge[0] not in env.valid_actions()


def test_dqn_forward_and_baselines():
    env=DynamicRoutingEnv(seed=3,nodes=15); env.reset(0,14)
    agent=DoubleDQNAgent(env.STATE_SIZE,env.NUM_NODES)
    a=agent.act(env._state(),valid_actions=env.valid_actions())
    assert a in env.valid_actions()
    assert shortest_path(env.graph,env.edge_cost,0,14)[0]==0
    assert ecmp_path(env.graph,env.edge_cost,0,14)[0]==0
