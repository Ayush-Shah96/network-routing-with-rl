from __future__ import annotations
import math
from rl.agent import DoubleDQNAgent
from simulation.environment import DynamicRoutingEnv


def explain_decision(env, agent, state, current):
    q=agent.q_values(state); rows=[]
    for n in env.valid_actions():
        l=env.get_link(current,n)
        rows.append({"next_hop":n,"q_value":float(q[n]),"latency_ms":l.latency_ms,"congestion":l.congestion,"loss":l.packet_loss,"queue_ms":l.queue_ms,"cost":env.edge_cost(current,n),"safety":l.availability>0 and n not in env.visited})
    return sorted(rows,key=lambda r:r["q_value"],reverse=True)


def _safe_actions(env):
    actions=env.valid_actions()
    unvisited=[a for a in actions if a not in env.visited]
    return unvisited or actions


def safe_action(env, agent, state):
    """Constrained DQN: learned Q proposal + safety shield + path-existence check."""
    actions=_safe_actions(env)
    if not actions: return None, {"reason":"no_available_next_hop"}
    q=agent.q_values(state)
    scored=[]
    min_cost=min(env.edge_cost(env.current,a) for a in actions)
    # Safety shield allows the learned action, unless its observed link is dramatically worse than the best safe alternative.
    for a in actions:
        l=env.get_link(env.current,a)
        g=env.graph.copy()
        dead=[(u,v) for u,v in g.edges() if env.get_link(u,v) and env.get_link(u,v).availability<=0]
        g.remove_edges_from(dead)
        try: path_exists=nx_has_path(g,a,env.destination)
        except Exception: path_exists=False
        if not path_exists: continue
        excess=max(0.0,env.edge_cost(env.current,a)-min_cost)
        score=float(q[a])-0.035*excess
        scored.append((score,a))
    if not scored:
        return min(actions,key=lambda a:env.edge_cost(env.current,a)), {"reason":"fallback_min_cost"}
    score,action=max(scored)
    q_best=max(q[a] for _,a in scored)
    reason="dqn_policy"
    if env.edge_cost(env.current,action)>min_cost*2.2:
        action=min(actions,key=lambda a:env.edge_cost(env.current,a)); reason="safety_shield_cost_cap"
    elif q[action] < q_best-1e-9:
        reason="safety_shield"
    return int(action), {"reason":reason,"q_value":float(q[action]),"best_safe_cost":float(min_cost)}


def nx_has_path(graph, source, target):
    import networkx as nx
    return nx.has_path(graph,source,target)


def route_with_agent(env: DynamicRoutingEnv, agent: DoubleDQNAgent, source, destination, epsilon=0.0, safe=False):
    state=env.reset(source=source,destination=destination)
    route=[source]; total=0.; done=False; events=[]; decisions=[]
    while not done:
        table=explain_decision(env,agent,state,env.current); decisions.append(table)
        if safe:
            action,shield=safe_action(env,agent,state)
            if action is None:
                break
            shield["current"]=env.current; decisions[-1] = [dict(r,shield_reason=shield.get('reason','')) for r in table]
        else:
            action=agent.act(state,epsilon=epsilon,valid_actions=env.valid_actions())
        next_state,reward,done,info=env.step(action); total+=reward; events.append(info)
        if info.get('valid'): route.append(action)
        state=next_state
    return route,total,events,decisions
