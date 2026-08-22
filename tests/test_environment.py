from simulation.environment import DynamicRoutingEnv


def test_reset_state_shape():
    env = DynamicRoutingEnv(seed=1)
    state = env.reset(source=0, destination=7)
    assert state.shape == (env.STATE_SIZE,)
    assert env.current == 0
    assert env.destination == 7


def test_valid_step_moves_to_neighbor():
    env = DynamicRoutingEnv(seed=1)
    env.reset(source=0, destination=7)
    next_node = next(iter(env.graph.neighbors(0)))
    _, reward, done, info = env.step(next_node)
    assert env.current == next_node
    assert info["valid"] is True
    assert isinstance(reward, float)
    assert isinstance(done, bool)


def test_invalid_step_is_penalized():
    env = DynamicRoutingEnv(seed=1)
    env.reset(source=0, destination=7)
    non_neighbor = next(n for n in range(env.NUM_NODES) if n != 0 and not env.graph.has_edge(0, n))
    _, reward, _, info = env.step(non_neighbor)
    assert reward < 0
    assert info["event"] == "invalid_action"
