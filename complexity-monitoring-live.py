# complexity_monitoring_live.py
"""
Live Complexity Monitoring - Überwacht Systemkomplexität in Echtzeit
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import logging
from prometheus_client import Gauge, Counter, Histogram
import networkx as nx

logger = logging.getLogger(__name__)

@dataclass
class InteractionEvent:
    """Repräsentiert eine Komponenten-Interaktion"""
    timestamp: datetime
    source_component: str
    target_component: str
    interaction_type: str  # 'sync', 'async', 'event'
    duration_ms: float
    success: bool
    data_size_bytes: int
    error_message: Optional[str] = None
    
@dataclass
class ComplexityMetrics:
    """Aktuelle Komplexitätsmetriken"""
    timestamp: datetime
    active_components: int
    active_interactions: int
    cyclomatic_complexity: float
    coupling_score: float
    interaction_density: float
    error_propagation_risk: float
    avg_interaction_latency: float
    
class LiveComplexityMonitor:
    """Überwacht System-Komplexität in Echtzeit"""
    
    def __init__(self, window_minutes: int = 60):
        self.window_minutes = window_minutes
        self.interaction_history = deque(maxlen=10000)
        self.component_health = defaultdict(lambda: {"healthy": True, "score": 1.0})
        self.interaction_graph = nx.DiGraph()
        
        # Prometheus metrics
        self.complexity_gauge = Gauge(
            'system_complexity_score',
            'Overall system complexity score',
            ['metric_type']
        )
        
        self.interaction_counter = Counter(
            'component_interactions_total',
            'Total component interactions',
            ['source', 'target', 'type']
        )
        
        self.interaction_latency = Histogram(
            'component_interaction_latency_ms',
            'Component interaction latency',
            ['source', 'target'],
            buckets=[10, 25, 50, 100, 250, 500, 1000, 2500, 5000]
        )
        
        self.error_propagation_gauge = Gauge(
            'error_propagation_risk',
            'Risk of error cascade'
        )
        
        # Thresholds for alerts
        self.complexity_thresholds = {
            "cyclomatic": 100,  # Max cyclomatic complexity
            "coupling": 0.4,    # Max coupling score
            "density": 0.6,     # Max interaction density
            "latency": 1000,    # Max avg latency in ms
            "error_prop": 0.3   # Max error propagation risk
        }
        
        # Complexity trend tracking
        self.complexity_history = deque(maxlen=1440)  # 24 hours of minute data
        
    async def record_interaction(self, event: InteractionEvent):
        """Zeichnet eine Komponenten-Interaktion auf"""
        self.interaction_history.append(event)
        
        # Update Prometheus metrics
        self.interaction_counter.labels(
            source=event.source_component,
            target=event.target_component,
            type=event.interaction_type
        ).inc()
        
        self.interaction_latency.labels(
            source=event.source_component,
            target=event.target_component
        ).observe(event.duration_ms)
        
        # Update interaction graph
        self.update_interaction_graph(event)
        
        # Check for complexity changes
        await self.analyze_complexity()
        
    def update_interaction_graph(self, event: InteractionEvent):
        """Aktualisiert den Interaktions-Graphen"""
        # Add or update edge
        if self.interaction_graph.has_edge(event.source_component, event.target_component):
            edge_data = self.interaction_graph[event.source_component][event.target_component]
            edge_data['count'] += 1
            edge_data['total_latency'] += event.duration_ms
            edge_data['avg_latency'] = edge_data['total_latency'] / edge_data['count']
            edge_data['error_rate'] = (edge_data.get('errors', 0) + (0 if event.success else 1)) / edge_data['count']
        else:
            self.interaction_graph.add_edge(
                event.source_component,
                event.target_component,
                count=1,
                total_latency=event.duration_ms,
                avg_latency=event.duration_ms,
                errors=0 if event.success else 1,
                error_rate=0 if event.success else 1
            )
            
        # Update component health
        if not event.success:
            self.component_health[event.source_component]['score'] *= 0.95
            self.component_health[event.target_component]['score'] *= 0.98
            
    async def analyze_complexity(self):
        """Analysiert aktuelle System-Komplexität"""
        metrics = self.calculate_complexity_metrics()
        
        # Update Prometheus metrics
        self.complexity_gauge.labels(metric_type='cyclomatic').set(metrics.cyclomatic_complexity)
        self.complexity_gauge.labels(metric_type='coupling').set(metrics.coupling_score)
        self.complexity_gauge.labels(metric_type='density').set(metrics.interaction_density)
        self.error_propagation_gauge.set(metrics.error_propagation_risk)
        
        # Store in history
        self.complexity_history.append(metrics)
        
        # Check for concerning trends
        await self.check_complexity_trends(metrics)
        
    def calculate_complexity_metrics(self) -> ComplexityMetrics:
        """Berechnet aktuelle Komplexitätsmetriken"""
        # Get recent interactions
        cutoff_time = datetime.now() - timedelta(minutes=self.window_minutes)
        recent_interactions = [i for i in self.interaction_history if i.timestamp > cutoff_time]
        
        if not recent_interactions:
            return ComplexityMetrics(
                timestamp=datetime.now(),
                active_components=0,
                active_interactions=0,
                cyclomatic_complexity=0,
                coupling_score=0,
                interaction_density=0,
                error_propagation_risk=0,
                avg_interaction_latency=0
            )
            
        # Active components
        active_components = set()
        for interaction in recent_interactions:
            active_components.add(interaction.source_component)
            active_components.add(interaction.target_component)
            
        num_components = len(active_components)
        num_edges = self.interaction_graph.number_of_edges()
        
        # Cyclomatic complexity
        cyclomatic = num_edges - num_components + 2 if num_components > 0 else 0
        
        # Coupling score
        max_possible_edges = num_components * (num_components - 1)
        coupling = num_edges / max_possible_edges if max_possible_edges > 0 else 0
        
        # Interaction density
        total_interactions = len(recent_interactions)
        density = total_interactions / (num_components * self.window_minutes) if num_components > 0 else 0
        
        # Average latency
        avg_latency = sum(i.duration_ms for i in recent_interactions) / len(recent_interactions)
        
        # Error propagation risk
        error_prop_risk = self.calculate_error_propagation_risk()
        
        return ComplexityMetrics(
            timestamp=datetime.now(),
            active_components=num_components,
            active_interactions=num_edges,
            cyclomatic_complexity=cyclomatic,
            coupling_score=coupling,
            interaction_density=density,
            error_propagation_risk=error_prop_risk,
            avg_interaction_latency=avg_latency
        )
        
    def calculate_error_propagation_risk(self) -> float:
        """Berechnet Risiko von Fehler-Kaskaden"""
        if self.interaction_graph.number_of_nodes() == 0:
            return 0.0
            
        # Find critical paths (high centrality nodes)
        centrality = nx.betweenness_centrality(self.interaction_graph)
        
        # Calculate risk based on error rates and centrality
        total_risk = 0.0
        for node, cent_score in centrality.items():
            # Get error rate for this node
            node_error_rate = 0.0
            for _, _, data in self.interaction_graph.edges(node, data=True):
                node_error_rate = max(node_error_rate, data.get('error_rate', 0))
                
            # Risk is product of centrality and error rate
            total_risk += cent_score * node_error_rate
            
        return min(total_risk, 1.0)
        
    async def check_complexity_trends(self, current_metrics: ComplexityMetrics):
        """Prüft Komplexitäts-Trends und generiert Warnungen"""
        warnings = []
        
        # Check absolute thresholds
        if current_metrics.cyclomatic_complexity > self.complexity_thresholds["cyclomatic"]:
            warnings.append({
                "level": "HIGH",
                "metric": "cyclomatic_complexity",
                "value": current_metrics.cyclomatic_complexity,
                "threshold": self.complexity_thresholds["cyclomatic"],
                "message": "System cyclomatic complexity exceeds safe threshold"
            })
            
        if current_metrics.coupling_score > self.complexity_thresholds["coupling"]:
            warnings.append({
                "level": "MEDIUM",
                "metric": "coupling_score",
                "value": current_metrics.coupling_score,
                "threshold": self.complexity_thresholds["coupling"],
                "message": "Component coupling is too high"
            })
            
        # Check trends (complexity increasing too fast)
        if len(self.complexity_history) > 60:  # Need at least 1 hour of data
            recent_complexity = [h.cyclomatic_complexity for h in list(self.complexity_history)[-60:]]
            older_complexity = [h.cyclomatic_complexity for h in list(self.complexity_history)[-120:-60]]
            
            recent_avg = sum(recent_complexity) / len(recent_complexity)
            older_avg = sum(older_complexity) / len(older_complexity)
            
            if recent_avg > older_avg * 1.2:  # 20% increase
                warnings.append({
                    "level": "MEDIUM",
                    "metric": "complexity_trend",
                    "value": recent_avg / older_avg,
                    "threshold": 1.2,
                    "message": "Complexity increasing rapidly"
                })
                
        # Log warnings
        for warning in warnings:
            logger.warning(f"Complexity warning: {warning['message']} "
                         f"({warning['metric']}={warning['value']:.2f})")
            
        return warnings
        
    def get_component_coupling_report(self) -> Dict[str, Any]:
        """Generiert Bericht über Komponenten-Kopplung"""
        report = {
            "timestamp": datetime.now(),
            "highly_coupled_components": [],
            "circular_dependencies": [],
            "bottleneck_components": []
        }
        
        # Find highly coupled components
        for node in self.interaction_graph.nodes():
            in_degree = self.interaction_graph.in_degree(node)
            out_degree = self.interaction_graph.out_degree(node)
            total_degree = in_degree + out_degree
            
            if total_degree > 10:  # Arbitrary threshold
                report["highly_coupled_components"].append({
                    "component": node,
                    "in_degree": in_degree,
                    "out_degree": out_degree,
                    "total_connections": total_degree
                })
                
        # Find circular dependencies
        try:
            cycles = list(nx.simple_cycles(self.interaction_graph))
            report["circular_dependencies"] = [
                {"cycle": cycle, "length": len(cycle)}
                for cycle in cycles
            ]
        except:
            pass  # Graph might not have cycles
            
        # Find bottlenecks (high betweenness centrality)
        centrality = nx.betweenness_centrality(self.interaction_graph)
        bottlenecks = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        report["bottleneck_components"] = [
            {"component": comp, "centrality_score": score}
            for comp, score in bottlenecks if score > 0.1
        ]
        
        return report
        
    async def suggest_decoupling_actions(self) -> List[Dict[str, Any]]:
        """Schlägt Maßnahmen zur Reduktion der Kopplung vor"""
        suggestions = []
        
        # Analyze interaction patterns
        coupling_report = self.get_component_coupling_report()
        
        # Suggest event bus for highly coupled components
        for comp_info in coupling_report["highly_coupled_components"]:
            if comp_info["total_connections"] > 15:
                suggestions.append({
                    "component": comp_info["component"],
                    "action": "introduce_event_bus",
                    "reason": f"Component has {comp_info['total_connections']} connections",
                    "expected_reduction": "50-70% direct dependencies"
                })
                
        # Suggest breaking circular dependencies
        for cycle_info in coupling_report["circular_dependencies"]:
            if cycle_info["length"] > 2:
                suggestions.append({
                    "components": cycle_info["cycle"],
                    "action": "break_circular_dependency",
                    "reason": f"Circular dependency of length {cycle_info['length']}",
                    "suggestion": "Introduce mediator or invert dependency"
                })
                
        # Suggest caching for bottlenecks
        for bottleneck in coupling_report["bottleneck_components"]:
            suggestions.append({
                "component": bottleneck["component"],
                "action": "add_caching_layer",
                "reason": f"High centrality score: {bottleneck['centrality_score']:.2f}",
                "expected_benefit": "Reduce load and coupling"
            })
            
        return suggestions
        
    def export_complexity_report(self) -> Dict[str, Any]:
        """Exportiert umfassenden Komplexitätsbericht"""
        current_metrics = self.calculate_complexity_metrics()
        
        report = {
            "report_timestamp": datetime.now().isoformat(),
            "summary": {
                "overall_health": self.calculate_overall_health(),
                "complexity_score": self.calculate_complexity_score(current_metrics),
                "trend": self.calculate_complexity_trend()
            },
            "current_metrics": {
                "active_components": current_metrics.active_components,
                "cyclomatic_complexity": current_metrics.cyclomatic_complexity,
                "coupling_score": current_metrics.coupling_score,
                "avg_latency_ms": current_metrics.avg_interaction_latency,
                "error_propagation_risk": current_metrics.error_propagation_risk
            },
            "coupling_analysis": self.get_component_coupling_report(),
            "recommendations": asyncio.run(self.suggest_decoupling_actions()),
            "component_health": dict(self.component_health)
        }
        
        return report
        
    def calculate_overall_health(self) -> float:
        """Berechnet Gesamt-Systemgesundheit"""
        if not self.complexity_history:
            return 1.0
            
        current = self.complexity_history[-1]
        
        # Normalize metrics to 0-1 scale
        health_factors = [
            1.0 - min(current.cyclomatic_complexity / self.complexity_thresholds["cyclomatic"], 1.0),
            1.0 - min(current.coupling_score / self.complexity_thresholds["coupling"], 1.0),
            1.0 - min(current.avg_interaction_latency / self.complexity_thresholds["latency"], 1.0),
            1.0 - current.error_propagation_risk
        ]
        
        return sum(health_factors) / len(health_factors)
        
    def calculate_complexity_score(self, metrics: ComplexityMetrics) -> float:
        """Berechnet einzelnen Komplexitäts-Score"""
        # Weighted combination of metrics
        weights = {
            "cyclomatic": 0.3,
            "coupling": 0.3,
            "density": 0.2,
            "error_risk": 0.2
        }
        
        normalized_scores = {
            "cyclomatic": min(metrics.cyclomatic_complexity / 100, 1.0),
            "coupling": metrics.coupling_score,
            "density": min(metrics.interaction_density, 1.0),
            "error_risk": metrics.error_propagation_risk
        }
        
        return sum(weights[k] * normalized_scores[k] for k in weights)
        
    def calculate_complexity_trend(self) -> str:
        """Berechnet Komplexitäts-Trend"""
        if len(self.complexity_history) < 120:  # Need 2 hours
            return "UNKNOWN"
            
        recent_scores = [
            self.calculate_complexity_score(m) 
            for m in list(self.complexity_history)[-60:]
        ]
        older_scores = [
            self.calculate_complexity_score(m) 
            for m in list(self.complexity_history)[-120:-60]
        ]
        
        recent_avg = sum(recent_scores) / len(recent_scores)
        older_avg = sum(older_scores) / len(older_scores)
        
        if recent_avg > older_avg * 1.1:
            return "INCREASING"
        elif recent_avg < older_avg * 0.9:
            return "DECREASING"
        else:
            return "STABLE"

# Integration with main system
class ComplexityAwareSystem:
    """Wrapper für Haupt-System mit Komplexitäts-Überwachung"""
    
    def __init__(self, main_system, complexity_monitor: LiveComplexityMonitor):
        self.main_system = main_system
        self.complexity_monitor = complexity_monitor
        
    async def execute_with_monitoring(self, source: str, target: str, 
                                    operation: callable, *args, **kwargs):
        """Führt Operation aus und überwacht Komplexität"""
        start_time = datetime.now()
        success = True
        error_msg = None
        result = None
        
        try:
            result = await operation(*args, **kwargs)
        except Exception as e:
            success = False
            error_msg = str(e)
            raise
        finally:
            # Record interaction
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            event = InteractionEvent(
                timestamp=start_time,
                source_component=source,
                target_component=target,
                interaction_type="sync",
                duration_ms=duration_ms,
                success=success,
                data_size_bytes=0,  # Would calculate from result
                error_message=error_msg
            )
            
            await self.complexity_monitor.record_interaction(event)
            
        return result

# Example usage
async def main():
    """Beispiel für Live Complexity Monitoring"""
    monitor = LiveComplexityMonitor(window_minutes=60)
    
    # Simulate some interactions
    for _ in range(100):
        await monitor.record_interaction(InteractionEvent(
            timestamp=datetime.now(),
            source_component="risk_management",
            target_component="portfolio_manager",
            interaction_type="sync",
            duration_ms=50,
            success=True,
            data_size_bytes=1024
        ))
        
        await asyncio.sleep(0.1)
        
    # Get complexity report
    report = monitor.export_complexity_report()
    print(json.dumps(report, indent=2, default=str))

if __name__ == "__main__":
    asyncio.run(main())