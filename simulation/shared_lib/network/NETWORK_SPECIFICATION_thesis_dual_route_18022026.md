# Network Specification: thesis_dual_route_18022026

**Version:** 1.0
**Created:** 2026-02-18
**Status:** LOCKED - No changes without user approval

---

## Overview

This network is designed for studying Wardrop Equilibrium and Price of Anarchy with:
- Heterogeneous Value of Time (VOT)
- Mixed agent populations (Type A fixed + Type B adaptive)
- Bidirectional traffic with overtaking capability

---

## Network Structure

### Visual Layout

```
                       LONG PATH (curves UPWARD through +Y)
                              (9.375km outer arc)
                         ╭───────────────────────────────────╮
                        ╱                                     ╲
                       ╱    ────────────► FWD ───────────────►╲
   ENTRY              ╱     ◄─────────── BWD ◄───────────────  ╲              ENTRY
  (1km 2-lane)       ╱       [2 adjacent lanes: overtaking]    ╲           (1km 2-lane)
      ║             ╱                                           ╲             ║
      ▼            ╱                                             ╲            ▼
   ┌─────┐        ●───────────────────────────────────────────────●        ┌─────┐
   │LEFT │    node_left (0,0)                           (4775,0) node_right│RIGHT│
   └─────┘        ●───────────────────────────────────────────────●        └─────┘
      ▲            ╲                                             ╱            ▲
      ║             ╲       [2 adjacent lanes: overtaking]      ╱             ║
  (1km 2-lane)       ╲   ────────────► FWD ───────────────────►╱           (1km 2-lane)
   ENTRY              ╲  ◄─────────── BWD ◄───────────────────╱              ENTRY
                       ╲                                     ╱
                        ╲                                   ╱
                         ╰─────────────────────────────────╯
                              (7.5km inner arc)
                       SHORT PATH (curves DOWNWARD through -Y)
```

---

## Parameters

### Path Dimensions

| Parameter | Short Path | Long Path |
|-----------|------------|-----------|
| Entry section | 1.0 km | 1.0 km |
| Entry lanes | 2 | 2 |
| Body length | 7.5 km | 9.375 km |
| Body lanes | 2 (1 FWD + 1 BWD) | 2 (1 FWD + 1 BWD) |
| Total length | 8.5 km | 10.375 km |
| **Path ratio** | 1.0 | **1.25** |

### Physical Properties

| Property | Value |
|----------|-------|
| Speed limit | 40 m/s (144 km/h) |
| Lane width | 3.5 m |
| Geometry | Semicircular arcs |
| Entry geometry | Straight lines |
| spreadType | roadCenter |
| Sublane model | lateral-resolution 0.8 |

### Overtaking

| Property | Value |
|----------|-------|
| Mechanism | Opposite-lane overtaking |
| SUMO element | `<neigh>` connection |
| lcOpposite | 1.0 |
| lcSpeedGain | 1.5 |

---

## Node Definitions

| Node ID | X | Y | Type | Purpose |
|---------|---|---|------|---------|
| node_entry_long_fwd | -1000 | 100 | priority | Long path FWD entry |
| node_entry_short_fwd | -1000 | -100 | priority | Short path FWD entry |
| node_left | 0 | 0 | priority | Left junction (route choice) |
| node_right | 4775 | 0 | priority | Right junction (route choice) |
| node_entry_long_bwd | 5775 | 100 | priority | Long path BWD entry |
| node_entry_short_bwd | 5775 | -100 | priority | Short path BWD entry |

---

## Edge Definitions

### Entry Edges (2 lanes each)

| Edge ID | From | To | Lanes | Length | Description |
|---------|------|-----|-------|--------|-------------|
| entry_long_fwd | node_entry_long_fwd | node_left | 2 | 1.0 km | Long FWD entry queue |
| entry_short_fwd | node_entry_short_fwd | node_left | 2 | 1.0 km | Short FWD entry queue |
| entry_long_bwd | node_entry_long_bwd | node_right | 2 | 1.0 km | Long BWD entry queue |
| entry_short_bwd | node_entry_short_bwd | node_right | 2 | 1.0 km | Short BWD entry queue |

### Body Edges (1 lane each, with opposite-lane neighbor)

| Edge ID | From | To | Lanes | Length | Neighbor | Description |
|---------|------|-----|-------|--------|----------|-------------|
| long_fwd | node_left | node_right | 1 | 9.375 km | long_bwd | Long path forward |
| long_bwd | node_right | node_left | 1 | 9.375 km | long_fwd | Long path backward |
| short_fwd | node_left | node_right | 1 | 7.5 km | short_bwd | Short path forward |
| short_bwd | node_right | node_left | 1 | 7.5 km | short_fwd | Short path backward |

---

## Routing

### Forward Direction (Left to Right)
- **Origin entries:** entry_long_fwd, entry_short_fwd
- **Route choice:** At node_left - agents choose short_fwd or long_fwd
- **Destination:** node_right

### Backward Direction (Right to Left)
- **Origin entries:** entry_long_bwd, entry_short_bwd
- **Route choice:** At node_right - agents choose short_bwd or long_bwd
- **Destination:** node_left

---

## Capacity Estimates

| Metric | Value |
|--------|-------|
| Entry queue capacity | ~65 vehicles per lane |
| Total entry queue per path | ~130 vehicles |
| Body throughput (1 lane) | ~1800-2200 veh/h |
| Total network capacity | ~3600-4400 veh/h |

---

## Files

| File | Purpose |
|------|---------|
| thesis_dual_route_18022026.nod.xml | Node definitions |
| thesis_dual_route_18022026.edg.xml | Edge definitions |
| thesis_dual_route_18022026.net.xml | Compiled network |
| thesis_dual_route_18022026.sumocfg | SUMO configuration |

---

## Change Log

| Date | Version | Change |
|------|---------|--------|
| 2026-02-18 | 1.0 | Initial specification - LOCKED |

---

## Approval Status

**LOCKED** - This network specification has been approved by the user.

Any modifications require:
1. Explicit user approval
2. Documentation of the change
3. Update to this specification
4. Version increment
