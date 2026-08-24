from __future__ import annotations
from pathlib import Path
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
#Updated new logic
ROOT=Path(__file__).resolve().parent
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from simulation.environment import DynamicRoutingEnv
from rl.agent import DoubleDQNAgent
from routing.rl_router import route_with_agent
from utils.metrics import route_metrics
from simulation.topology import node_name

MODEL=ROOT/'models'/'dqn_routing.pt'
app=FastAPI(title='Adaptive Network Intelligence API',version='2.0')

class RouteRequest(BaseModel):
    source:int=Field(0,ge=0,le=14)
    destination:int=Field(14,ge=0,le=14)
    scenario:str='normal'
    safe:bool=True
    seed:int=42

@app.get('/health')
def health():
    return {'status':'ok','model_available':MODEL.exists(),'simulation_only':True}

@app.post('/route')
def route(req:RouteRequest):
    if req.source==req.destination: raise HTTPException(400,'source and destination must differ')
    env=DynamicRoutingEnv(seed=req.seed,nodes=15); agent=DoubleDQNAgent.load(MODEL,env.STATE_SIZE,env.NUM_NODES) if MODEL.exists() else None
    if agent is None: raise HTTPException(503,'model checkpoint is not available')
    route,reward,events,decisions=route_with_agent(env,agent,req.source,req.destination,safe=req.safe)
    return {'source':node_name(req.source),'destination':node_name(req.destination),'route':[node_name(n) for n in route],'reward':reward,'metrics':route_metrics(env,route),'events':events,'decisions':decisions}
