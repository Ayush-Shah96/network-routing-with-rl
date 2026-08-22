from __future__ import annotations
import math


def route_metrics(env, route):
    latency=0.0; survival=1.0; min_bw=math.inf; congestion=[]; queue=[]; valid=True
    for u,v in zip(route,route[1:]):
        l=env.get_link(u,v)
        if l is None or l.availability<=0: valid=False; break
        latency += l.latency_ms + l.queue_ms; survival *= max(0,1-l.packet_loss); min_bw=min(min_bw,l.bandwidth_mbps*(1-l.congestion)); congestion.append(l.congestion); queue.append(l.queue_ms)
    hops=max(0,len(route)-1)
    if not valid:return {"valid":False,"delivered":False,"latency_ms":999.0,"packet_loss_pct":100.0,"throughput_mbps":0.0,"hops":hops,"avg_congestion_pct":100.0,"avg_queue_ms":99.0}
    return {"valid":True,"delivered":bool(route and route[-1]==env.destination),"latency_ms":latency,"packet_loss_pct":(1-survival)*100,"throughput_mbps":0 if hops==0 else max(0,min_bw),"hops":hops,"avg_congestion_pct":0 if not congestion else sum(congestion)/len(congestion)*100,"avg_queue_ms":0 if not queue else sum(queue)/len(queue)}
