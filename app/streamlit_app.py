from __future__ import annotations
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import networkx as nx

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from rl.agent import DoubleDQNAgent
from routing.baselines import shortest_path, random_path, ecmp_path
from routing.rl_router import route_with_agent, explain_decision
from simulation.environment import DynamicRoutingEnv
from simulation.topology import node_name, NODE_NAMES
from utils.metrics import route_metrics
from scripts.train_model import train

MODEL=ROOT/'models'/'dqn_routing.pt'
ART=ROOT/'artifacts'
ART.mkdir(exist_ok=True)

st.set_page_config(page_title='Adaptive Network Intelligence',page_icon='🧠',layout='wide')
st.title('🧠 Adaptive Network Intelligence')
st.caption('Research-grade dynamic routing lab: Double-DQN, dueling architecture, prioritized replay, failures, congestion, multi-scenario benchmarks and explainable routing decisions.')

if 'env' not in st.session_state:
    st.session_state.env=DynamicRoutingEnv(seed=42,nodes=15)
    st.session_state.env.reset(0,14)
if 'route' not in st.session_state: st.session_state.route=[0,14]
if 'decisions' not in st.session_state: st.session_state.decisions=[]
if 'last_metrics' not in st.session_state: st.session_state.last_metrics={}


def get_agent():
    if not MODEL.exists(): return None
    env=st.session_state.env
    return DoubleDQNAgent.load(MODEL,env.STATE_SIZE,env.NUM_NODES)


def network_figure(env,route=None):
    pos=nx.spring_layout(env.graph,seed=17)
    edge_x=[]; edge_y=[]; edge_text=[]
    hot_x=[]; hot_y=[]; hot_text=[]
    route_edges=set(tuple(sorted(e)) for e in zip(route or [],(route or [])[1:]))
    for u,v in env.graph.edges():
        x0,y0=pos[u]; x1,y1=pos[v]; l=env.get_link(u,v)
        target=(hot_x,hot_y,hot_text) if l.availability<=0 or l.congestion>.7 else (edge_x,edge_y,edge_text)
        target[0].extend([x0,x1,None]); target[1].extend([y0,y1,None]); target[2].append(f'{node_name(u)}-{node_name(v)} | latency {l.latency_ms:.1f}ms | congestion {l.congestion*100:.0f}% | loss {l.packet_loss*100:.2f}%')
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=edge_x,y=edge_y,mode='lines',line=dict(width=2),hoverinfo='text',text=edge_text,name='Healthy/degraded links'))
    fig.add_trace(go.Scatter(x=hot_x,y=hot_y,mode='lines',line=dict(width=4,dash='dot'),hoverinfo='text',text=hot_text,name='Congested/failed links'))
    if route and len(route)>1:
        rx=[]; ry=[]
        for u,v in zip(route,route[1:]):
            rx.extend([pos[u][0],pos[v][0],None]); ry.extend([pos[u][1],pos[v][1],None])
        fig.add_trace(go.Scatter(x=rx,y=ry,mode='lines',line=dict(width=8),name='Selected route'))
    fig.add_trace(go.Scatter(x=[pos[n][0] for n in env.graph.nodes],y=[pos[n][1] for n in env.graph.nodes],mode='markers+text',text=[node_name(n) for n in env.graph.nodes],textposition='top center',marker=dict(size=26),name='Nodes',hovertemplate='%{text}<extra></extra>'))
    fig.update_layout(height=520,showlegend=True,xaxis=dict(visible=False),yaxis=dict(visible=False),margin=dict(l=0,r=0,t=0,b=0))
    return fig


with st.sidebar:
    st.header('Experiment controls')
    topology=st.selectbox('Topology',['mesh','scale_free','random','grid'])
    seed=st.number_input('Seed',0,99999,42)
    source=st.selectbox('Source',NODE_NAMES[:15],0)
    destination=st.selectbox('Destination',NODE_NAMES[:15],14)
    algorithm=st.selectbox('Routing policy',['Safe DQN','DQN','Dijkstra','ECMP','Random'])
    scenario=st.selectbox('Scenario',['normal','rush_hour','degraded','failure','storm'])
    if st.button('Reset environment',use_container_width=True):
        st.session_state.env=DynamicRoutingEnv(seed=seed,topology=topology,nodes=15); st.session_state.env.reset(NODE_NAMES.index(source),NODE_NAMES.index(destination),scenario); st.session_state.route=[NODE_NAMES.index(source)]; st.session_state.decisions=[]; st.rerun()
    if st.button('Inject targeted congestion',use_container_width=True):
        st.session_state.env.inject_congestion(st.session_state.route if len(st.session_state.route)>1 else None); st.rerun()
    if st.button('Inject link failure',use_container_width=True):
        edges=list(st.session_state.env.graph.edges());
        if edges: st.session_state.env.set_link_failure(*edges[int(seed)%len(edges)],down=True)
        st.rerun()
    st.divider(); st.subheader('Model status')
    st.write('Double-DQN + Dueling + Prioritized Replay')
    st.success('Pre-trained model available' if MODEL.exists() else 'Train model first')
    if st.button('Train / retrain model',use_container_width=True):
        with st.spinner('Training advanced agent…'):
            rewards=train(episodes=1600,seed=int(seed),topology=topology,nodes=15)
        st.success(f'Finished. Last-100 reward: {np.mean(rewards[-100:]):.2f}')
        st.rerun()

# Keep environment aligned with selector when the user changes fields.
env=st.session_state.env
if env.NUM_NODES != 15 or env.graph.number_of_nodes()!=15: env=DynamicRoutingEnv(seed=seed,topology=topology,nodes=15); st.session_state.env=env

m=env.snapshot()
cols=st.columns(6)
cols[0].metric('Nodes',env.NUM_NODES)
cols[1].metric('Links',env.graph.number_of_edges())
cols[2].metric('Congested',sum(1 for l in env.links.values() if l.congestion>.7))
cols[3].metric('Failed',sum(1 for l in env.links.values() if l.availability<=0))
cols[4].metric('Scenario',scenario)
cols[5].metric('Model','DQN' if MODEL.exists() else 'N/A')

run=st.button('▶ Run adaptive routing',type='primary',use_container_width=True)
if run:
    s=NODE_NAMES.index(source); d=NODE_NAMES.index(destination)
    env.reset(s,d,scenario)
    if algorithm in ('DQN','Safe DQN'):
        agent=get_agent()
        if agent is None: st.error('Train or provide the model first.')
        else:
            route,reward,events,decisions=route_with_agent(env,agent,s,d,safe=(algorithm=='Safe DQN'))
            st.session_state.route=route; st.session_state.decisions=decisions; st.session_state.last_metrics=route_metrics(env,route); st.session_state.last_reward=reward
    elif algorithm=='Dijkstra': st.session_state.route=shortest_path(env.graph,env.edge_cost,s,d); st.session_state.decisions=[]; st.session_state.last_metrics=route_metrics(env,st.session_state.route)
    elif algorithm=='ECMP': st.session_state.route=ecmp_path(env.graph,env.edge_cost,s,d,seed=int(seed)); st.session_state.decisions=[]; st.session_state.last_metrics=route_metrics(env,st.session_state.route)
    else: st.session_state.route=random_path(env.graph,s,d,seed=int(seed)); st.session_state.decisions=[]; st.session_state.last_metrics=route_metrics(env,st.session_state.route)
    st.session_state.env=env

left,right=st.columns([1.7,1])
with left:
    st.subheader('Live network / topology state')
    st.plotly_chart(network_figure(env,st.session_state.route),use_container_width=True)
with right:
    st.subheader('Routing outcome')
    route=st.session_state.route; metrics=st.session_state.last_metrics
    st.write('**Policy:**',algorithm)
    st.write('**Route:**',' → '.join(node_name(n) for n in route))
    c1,c2=st.columns(2); c1.metric('Latency',f"{metrics.get('latency_ms',0):.1f} ms"); c2.metric('Loss',f"{metrics.get('packet_loss_pct',0):.2f}%")
    c3,c4=st.columns(2); c3.metric('Throughput',f"{metrics.get('throughput_mbps',0):.1f} Mbps"); c4.metric('Hops',metrics.get('hops',0))
    st.success('Delivered' if metrics.get('delivered') else 'Not delivered')

st.divider()
t1,t2,t3,t4,t5=st.tabs(['Explainable AI','Benchmark Lab','Training Lab','Network Telemetry','Research Notes'])
with t1:
    st.subheader('Why did the agent choose this next hop?')
    if st.session_state.decisions:
        rows=[]
        for i,table in enumerate(st.session_state.decisions,1):
            for r in table: rows.append({'decision_step':i,**r,'next_hop':node_name(r['next_hop'])})
        df=pd.DataFrame(rows); st.dataframe(df,use_container_width=True,hide_index=True)
        st.caption('Q-value is the learned action value. Cost fields show the network conditions that the agent observed at that decision point.')
    else: st.info('Run the DQN policy to populate decision explanations.')

with t2:
    st.subheader('Robust benchmark across dynamic scenarios')
    runs=st.slider('Runs per scenario',20,300,80,20)
    if st.button('Run full benchmark',use_container_width=True):
        from scripts.evaluate import evaluate
        with st.spinner('Evaluating all policies across normal, rush-hour, degraded, failure and storm conditions…'):
            df,summary=evaluate(runs=runs,seed=int(seed),topology=topology,nodes=15)
        st.dataframe(summary,use_container_width=True)
        st.download_button('Download detailed CSV',df.to_csv(index=False).encode(),'benchmark_results.csv','text/csv')

with t3:
    st.subheader('Training history')
    hist=ART/'training_history.json'
    if hist.exists():
        data=json.loads(hist.read_text()); rewards=pd.Series(data.get('rewards',[])); losses=pd.Series(data.get('losses',[]))
        st.line_chart(rewards.rolling(50).mean().rename('50-episode reward'))
        if len(losses): st.line_chart(losses.rolling(100).mean().rename('100-update loss'))
    else: st.info('Run training to create reproducible history.')

with t4:
    st.subheader('Current link telemetry')
    edges=pd.DataFrame(env.snapshot()['edges'])
    st.dataframe(edges.sort_values(['congestion','latency_ms'],ascending=False),use_container_width=True,hide_index=True)
    st.download_button('Download telemetry JSON',json.dumps(env.snapshot(),indent=2).encode(),'network_snapshot.json','application/json')

with t5:
    st.subheader('What this platform demonstrates')
    st.markdown('''\n- **Adaptive routing:** the policy reacts to congestion, queueing, packet loss and link failures.\n- **Modern DQN:** Double-DQN reduces over-estimation; the dueling head separates state value from action advantage; prioritized replay emphasizes surprising transitions.\n- **Safety:** invalid/failed links are masked from action selection, so the policy cannot intentionally select unavailable neighbors.\n- **Benchmarking:** compare learned routing with shortest-path, ECMP-style and random policies under multiple stress scenarios.\n- **Research-ready artifacts:** training history and benchmark CSVs are persisted under `artifacts/`.\n- **Safe demo:** the application is a simulator and does not modify the host's routing tables.\n''')
st.caption('For real network experimentation, the next integration stage would connect this decision layer to Mininet/ns-3/SDN rather than directly altering a production interface.')
