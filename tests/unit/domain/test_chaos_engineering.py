"""Chaos engineering tests for system resilience validation."""

import pytest
import asyncio
import random
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any, Optional


class ChaosScenario:
    """Represents a chaos engineering scenario."""

    def __init__(self, name: str, description: str, duration: float, severity: str):
        self.name = name
        self.description = description
        self.duration = duration
        self.severity = severity  # 'low', 'medium', 'high', 'critical'
        self.active = False
        self.start_time = None
        self.end_time = None

    async def activate(self):
        """Activate the chaos scenario."""
        self.active = True
        self.start_time = time.time()
        self.end_time = self.start_time + self.duration
        print(f"🔥 Chaos scenario activated: {self.name}")

    async def deactivate(self):
        """Deactivate the chaos scenario."""
        self.active = False
        print(f"✅ Chaos scenario deactivated: {self.name}")

    def is_active(self) -> bool:
        """Check if scenario is currently active."""
        if not self.active:
            return False

        current_time = time.time()
        if current_time >= self.end_time:
            self.active = False
            return False

        return True


class ChaosMonkey:
    """Chaos monkey for injecting failures into the system."""

    def __init__(self):
        self.active_scenarios: List[ChaosScenario] = []
        self.metrics = {
            'scenarios_executed': 0,
            'total_failures_injected': 0,
            'system_recovery_times': [],
            'resilience_score': 0.0
        }

    async def inject_network_partition(self, duration: float = 10.0, affected_services: List[str] = None):
        """Inject network partition scenario."""
        scenario = ChaosScenario(
            name="Network Partition",
            description=f"Network partition affecting services: {affected_services or ['all']}",
            duration=duration,
            severity="high"
        )

        await scenario.activate()
        self.active_scenarios.append(scenario)

        # Simulate network partition effects
        async def network_failure():
            if affected_services:
                for service in affected_services:
                    print(f"🚫 Network partition: {service} unreachable")
            else:
                print("🚫 Network partition: All external services unreachable")

        await network_failure()
        return scenario

    async def inject_high_latency(self, duration: float = 15.0, latency_ms: int = 5000):
        """Inject high latency scenario."""
        scenario = ChaosScenario(
            name="High Latency",
            description=f"Network latency increased to {latency_ms}ms",
            duration=duration,
            severity="medium"
        )

        await scenario.activate()
        self.active_scenarios.append(scenario)

        print(f"🐌 High latency injected: {latency_ms}ms delay")
        return scenario

    async def inject_memory_pressure(self, duration: float = 20.0, memory_limit_mb: int = 512):
        """Inject memory pressure scenario."""
        scenario = ChaosScenario(
            name="Memory Pressure",
            description=f"Available memory limited to {memory_limit_mb}MB",
            duration=duration,
            severity="high"
        )

        await scenario.activate()
        self.active_scenarios.append(scenario)

        print(f"💾 Memory pressure injected: {memory_limit_mb}MB limit")
        return scenario

    async def inject_cpu_spike(self, duration: float = 12.0, cpu_usage_percent: int = 95):
        """Inject CPU spike scenario."""
        scenario = ChaosScenario(
            name="CPU Spike",
            description=f"CPU usage spiked to {cpu_usage_percent}%",
            duration=duration,
            severity="medium"
        )

        await scenario.activate()
        self.active_scenarios.append(scenario)

        print(f"🔥 CPU spike injected: {cpu_usage_percent}% usage")
        return scenario

    async def inject_service_unavailable(self, duration: float = 18.0, services: List[str] = None):
        """Inject service unavailable scenario."""
        scenario = ChaosScenario(
            name="Service Unavailable",
            description=f"Services unavailable: {services or ['random']}",
            duration=duration,
            severity="critical"
        )

        await scenario.activate()
        self.active_scenarios.append(scenario)

        affected_services = services or ['metaculus_api', 'search_api']
        for service in affected_services:
            print(f"❌ Service unavailable: {service}")

        return scenario

    async def inject_random_failures(self, duration: float = 30.0, failure_rate: float = 0.1):
        """Inject random failures throughout the system."""
        scenario = ChaosScenario(
            name="Random Failures",
            description=f"Random failures at {failure_rate*100}% rate",
            duration=duration,
            severity="low"
        )

        await scenario.activate()
        self.active_scenarios.append(scenario)

        print(f"🎲 Random failures injected: {failure_rate*100}% failure rate")
        return scenario

    def get_active_scenarios(self) -> List[ChaosScenario]:
        """Get currently active chaos scenarios."""
        return [s for s in self.active_scenarios if s.is_active()]

    async def cleanup_scenarios(self):
        """Clean up completed scenarios."""
        active_scenarios = []
        for scenario in self.active_scenarios:
            if scenario.is_active():
                active_scenarios.append(scenario)
            else:
                await scenario.deactivate()

        self.active_scenarios = active_scenarios

    def calculate_resilience_score(self, test_results: Dict[str, Any]) -> float:
        """Calculate system resilience score based on test results."""
        factors = {
            'availability': test_results.get('availability', 0.0),
            'recovery_time': min(1.0, 30.0 / test_results.get('avg_recovery_time', 30.0)),
            'error_rate': 1.0 - test_results.get('error_rate', 0.0),
            'throughput_degradation': 1.0 - test_results.get('throughput_degradation', 0.0)
        }

        # Weighted average
        weights = {'availability': 0.3, 'recovery_time': 0.3, 'error_rate': 0.2, 'throughput_degradation': 0.2}
        resilience_score = sum(factors[k] * weights[k] for k in factors.keys())

        self.metrics['resilience_score'] = resilience_score
        return resilience_score


class ResilientSystem:
    """Mock resilient system for testing chaos scenarios."""

    def __init__(self):
        self.health_status = "healthy"
        self.circuit_breakers = {}
        self.retry_counts = {}
        self.fallback_modes = {}
        self.recovery_strategies = {}

    async def process_request(self, request_id: str, chaos_monkey: ChaosMonkey) -> Dict[str, Any]:
        """Process a request with chaos scenarios potentially active."""
        start_time = time.time()

        try:
            # Check for active chaos scenarios
            active_scenarios = chaos_monkey.get_active_scenarios()

            # Apply chaos effects
            for scenario in active_scenarios:
                await self._apply_chaos_effects(scenario)

            # Simulate request processing
            processing_time = 0.1 + random.uniform(0, 0.2)  # Base processing time

            # Apply chaos-induced delays
            if any(s.name == "High Latency" for s in active_scenarios):
                processing_time += 2.0  # Add significant delay

            if any(s.name == "CPU Spike" for s in active_scenarios):
                processing_time *= 1.5  # Slower processing

            await asyncio.sleep(processing_time)

            # Check for failures
            failure_probability = self._calculate_failure_probability(active_scenarios)

            if random.random() < failure_probability:
                # Attempt recovery
                recovery_successful = await self._attempt_recovery(active_scenarios)
                if not recovery_successful:
                    raise Exception(f"Request {request_id} failed due to chaos scenarios")

            end_time = time.time()
            return {
                'request_id': request_id,
                'status': 'success',
                'processing_time': end_time - start_time,
                'chaos_scenarios_active': len(active_scenarios),
                'recovery_attempted': failure_probability > 0.1
            }

        except Exception as e:
            end_time = time.time()
            return {
                'request_id': request_id,
                'status': 'failed',
                'error': str(e),
                'processing_time': end_time - start_time,
                'chaos_scenarios_active': len(active_scenarios)
            }

    async def _apply_chaos_effects(self, scenario: ChaosScenario):
        """Apply effects of a chaos scenario."""
        if scenario.name == "Network Partition":
            # Simulate network issues
            self.circuit_breakers['network'] = True

        elif scenario.name == "Memory Pressure":
            # Simulate memory constraints
            self.fallback_modes['low_memory'] = True

        elif scenario.name == "Service Unavailable":
            # Simulate service failures
            self.circuit_breakers['external_services'] = True

    def _calculate_failure_probability(self, active_scenarios: List[ChaosScenario]) -> float:
        """Calculate failure probability based on active scenarios."""
        base_failure_rate = 0.01  # 1% base failure rate

        for scenario in active_scenarios:
            if scenario.severity == "critical":
                base_failure_rate += 0.3
            elif scenario.severity == "high":
                base_failure_rate += 0.15
            elif scenario.severity == "medium":
                base_failure_rate += 0.08
            elif scenario.severity == "low":
                base_failure_rate += 0.03

        return min(base_failure_rate, 0.8)  # Cap at 80% failure rate

    async def _attempt_recovery(self, active_scenarios: List[ChaosScenario]) -> bool:
        """Attempt to recover from failures."""
        recovery_strategies = []

        for scenario in active_scenarios:
            if scenario.name == "Network Partition":
                recovery_strategies.append("use_cached_data")
                recovery_strategies.append("fallback_to_local_processing")

            elif scenario.name == "High Latency":
                recovery_strategies.append("increase_timeouts")
                recovery_strategies.append("use_circuit_breaker")

            elif scenario.name == "Memory Pressure":
                recovery_strategies.append("reduce_batch_size")
                recovery_strategies.append("garbage_collection")

            elif scenario.name == "CPU Spike":
                recovery_strategies.append("throttle_requests")
                recovery_strategies.append("queue_processing")

            elif scenario.name == "Service Unavailable":
                recovery_strategies.append("use_alternative_service")
                recovery_strategies.append("degrade_gracefully")

        # Simulate recovery attempt
        await asyncio.sleep(0.05)  # Recovery time

        # Recovery success rate depends on available strategies
        recovery_success_rate = min(0.9, len(set(recovery_strategies)) * 0.2)
        return random.random() < recovery_success_rate


@pytest.mark.chaos
class TestChaosEngineering:
    """Chaos engineering tests for system resilience."""

    @pytest.fixture
    def chaos_monkey(self):
        """Create chaos monkey instance."""
        return ChaosMonkey()

    @pytest.fixture
    def resilient_system(self):
        """Create resilient system instance."""
        return ResilientSystem()

    async def test_network_partition_resilience(self, chaos_monkey, resilient_system):
        """Test system resilience during network partition."""
        print("\n🧪 Testing network partition resilience...")

        # Inject network partition
        scenario = await chaos_monkey.inject_network_partition(
            duration=10.0,
            affected_services=['search_api', 'llm_api']
        )

        # Process requests during network partition
        results = []
        for i in range(20):
            result = await resilient_system.process_request(f"req_{i}", chaos_monkey)
            results.append(result)
            await asyncio.sleep(0.1)

        # Wait for scenario to complete
        await asyncio.sleep(scenario.duration + 1)
        await chaos_monkey.cleanup_scenarios()

        # Analyze results
        successful_requests = [r for r in results if r['status'] == 'success']
        failed_requests = [r for r in results if r['status'] == 'failed']

        success_rate = len(successful_requests) / len(results)
        avg_processing_time = sum(r['processing_time'] for r in successful_requests) / len(successful_requests) if successful_requests else 0

        print(f"Network partition results:")
        print(f"  Success rate: {success_rate:.1%}")
        print(f"  Failed requests: {len(failed_requests)}")
        print(f"  Avg processing time: {avg_processing_time:.3f}s")

        # Assert resilience requirements
        assert success_rate >= 0.6, f"Success rate {success_rate:.1%} below minimum threshold (60%)"
        assert avg_processing_time < 5.0, f"Processing time {avg_processing_time:.3f}s exceeds threshold (5s)"

    async def test_high_latency_resilience(self, chaos_monkey, resilient_system):
        """Test system resilience during high latency conditions."""
        print("\n🧪 Testing high latency resilience...")

        # Inject high latency
        scenario = await chaos_monkey.inject_high_latency(
            duration=8.0,
            latency_ms=3000
        )

        # Process requests during high latency
        results = []
        start_time = time.time()

        for i in range(15):
            result = await resilient_system.process_request(f"latency_req_{i}", chaos_monkey)
            results.append(result)

        total_time = time.time() - start_time

        # Wait for scenario to complete
        await asyncio.sleep(max(0, scenario.duration - total_time + 1))
        await chaos_monkey.cleanup_scenarios()

        # Analyze results
        successful_requests = [r for r in results if r['status'] == 'success']
        success_rate = len(successful_requests) / len(results)
        avg_processing_time = sum(r['processing_time'] for r in successful_requests) / len(successful_requests) if successful_requests else 0

        print(f"High latency results:")
        print(f"  Success rate: {success_rate:.1%}")
        print(f"  Avg processing time: {avg_processing_time:.3f}s")
        print(f"  Total test time: {total_time:.3f}s")

        # Assert resilience requirements
        assert success_rate >= 0.7, f"Success rate {success_rate:.1%} below threshold (70%)"
        # Processing time will be higher due to latency, but should still complete
        assert avg_processing_time < 10.0, f"Processing time {avg_processing_time:.3f}s exceeds threshold (10s)"

    async def test_memory_pressure_resilience(self, chaos_monkey, resilient_system):
        """Test system resilience under memory pressure."""
        print("\n🧪 Testing memory pressure resilience...")

        # Inject memory pressure
        scenario = await chaos_monkey.inject_memory_pressure(
            duration=12.0,
            memory_limit_mb=256
        )

        # Process requests under memory pressure
        results = []
        for i in range(25):
            result = await resilient_system.process_request(f"memory_req_{i}", chaos_monkey)
            results.append(result)
            await asyncio.sleep(0.05)

        # Wait for scenario to complete
        await asyncio.sleep(scenario.duration + 1)
        await chaos_monkey.cleanup_scenarios()

        # Analyze results
        successful_requests = [r for r in results if r['status'] == 'success']
        success_rate = len(successful_requests) / len(results)
        recovery_attempts = sum(1 for r in results if r.get('recovery_attempted', False))

        print(f"Memory pressure results:")
        print(f"  Success rate: {success_rate:.1%}")
        print(f"  Recovery attempts: {recovery_attempts}")
        print(f"  Total requests: {len(results)}")

        # Assert resilience requirements
        assert success_rate >= 0.65, f"Success rate {success_rate:.1%} below threshold (65%)"
        assert recovery_attempts > 0, "No recovery attempts detected during memory pressure"

    async def test_multiple_chaos_scenarios(self, chaos_monkey, resilient_system):
        """Test system resilience with multiple simultaneous chaos scenarios."""
        print("\n🧪 Testing multiple simultaneous chaos scenarios...")

        # Inject multiple scenarios simultaneously
        scenarios = []
        scenarios.append(await chaos_monkey.inject_high_latency(duration=15.0, latency_ms=2000))
        scenarios.append(await chaos_monkey.inject_cpu_spike(duration=12.0, cpu_usage_percent=90))
        scenarios.append(await chaos_monkey.inject_random_failures(duration=18.0, failure_rate=0.15))

        # Process requests with multiple chaos scenarios active
        results = []
        start_time = time.time()

        for i in range(30):
            result = await resilient_system.process_request(f"multi_chaos_req_{i}", chaos_monkey)
            results.append(result)
            await asyncio.sleep(0.1)

        total_time = time.time() - start_time

        # Wait for all scenarios to complete
        max_duration = max(s.duration for s in scenarios)
        await asyncio.sleep(max(0, max_duration - total_time + 1))
        await chaos_monkey.cleanup_scenarios()

        # Analyze results
        successful_requests = [r for r in results if r['status'] == 'success']
        failed_requests = [r for r in results if r['status'] == 'failed']
        success_rate = len(successful_requests) / len(results)

        # Calculate metrics
        avg_processing_time = sum(r['processing_time'] for r in successful_requests) / len(successful_requests) if successful_requests else 0
        max_processing_time = max(r['processing_time'] for r in successful_requests) if successful_requests else 0

        print(f"Multiple chaos scenarios results:")
        print(f"  Active scenarios: {len(scenarios)}")
        print(f"  Success rate: {success_rate:.1%}")
        print(f"  Failed requests: {len(failed_requests)}")
        print(f"  Avg processing time: {avg_processing_time:.3f}s")
        print(f"  Max processing time: {max_processing_time:.3f}s")

        # Assert resilience requirements (more lenient due to multiple scenarios)
        assert success_rate >= 0.5, f"Success rate {success_rate:.1%} below threshold (50%)"
        assert avg_processing_time < 8.0, f"Avg processing time {avg_processing_time:.3f}s exceeds threshold (8s)"

    async def test_service_unavailable_resilience(self, chaos_monkey, resilient_system):
        """Test system resilience when external services are unavailable."""
        print("\n🧪 Testing service unavailable resilience...")

        # Inject service unavailable scenario
        scenario = await chaos_monkey.inject_service_unavailable(
            duration=10.0,
            services=['metaculus_api', 'search_api']
        )

        # Process requests with services unavailable
        results = []
        for i in range(20):
            result = await resilient_system.process_request(f"service_unavail_req_{i}", chaos_monkey)
            results.append(result)
            await asyncio.sleep(0.1)

        # Wait for scenario to complete
        await asyncio.sleep(scenario.duration + 1)
        await chaos_monkey.cleanup_scenarios()

        # Analyze results
        successful_requests = [r for r in results if r['status'] == 'success']
        success_rate = len(successful_requests) / len(results)
        recovery_attempts = sum(1 for r in results if r.get('recovery_attempted', False))

        print(f"Service unavailable results:")
        print(f"  Success rate: {success_rate:.1%}")
        print(f"  Recovery attempts: {recovery_attempts}")
        print(f"  Affected services: metaculus_api, search_api")

        # Assert resilience requirements
        assert success_rate >= 0.4, f"Success rate {success_rate:.1%} below threshold (40%)"
        assert recovery_attempts >= 5, f"Recovery attempts {recovery_attempts} below threshold (5)"

    async def test_chaos_scenario_recovery(self, chaos_monkey, resilient_system):
        """Test system recovery after chaos scenarios end."""
        print("\n🧪 Testing system recovery after chaos scenarios...")

        # Phase 1: Normal operation
        print("Phase 1: Normal operation baseline")
        baseline_results = []
        for i in range(10):
            result = await resilient_system.process_request(f"baseline_req_{i}", chaos_monkey)
            baseline_results.append(result)
            await asyncio.sleep(0.05)

        baseline_success_rate = len([r for r in baseline_results if r['status'] == 'success']) / len(baseline_results)
        baseline_avg_time = sum(r['processing_time'] for r in baseline_results if r['status'] == 'success') / len([r for r in baseline_results if r['status'] == 'success'])

        # Phase 2: Chaos injection
        print("Phase 2: Chaos injection")
        scenario = await chaos_monkey.inject_network_partition(duration=8.0)

        chaos_results = []
        for i in range(15):
            result = await resilient_system.process_request(f"chaos_req_{i}", chaos_monkey)
            chaos_results.append(result)
            await asyncio.sleep(0.1)

        chaos_success_rate = len([r for r in chaos_results if r['status'] == 'success']) / len(chaos_results)

        # Wait for chaos to end
        await asyncio.sleep(scenario.duration + 1)
        await chaos_monkey.cleanup_scenarios()

        # Phase 3: Recovery
        print("Phase 3: Post-chaos recovery")
        recovery_results = []
        for i in range(15):
            result = await resilient_system.process_request(f"recovery_req_{i}", chaos_monkey)
            recovery_results.append(result)
            await asyncio.sleep(0.05)

        recovery_success_rate = len([r for r in recovery_results if r['status'] == 'success']) / len(recovery_results)
        recovery_avg_time = sum(r['processing_time'] for r in recovery_results if r['status'] == 'success') / len([r for r in recovery_results if r['status'] == 'success'])

        print(f"Recovery test results:")
        print(f"  Baseline success rate: {baseline_success_rate:.1%}")
        print(f"  Chaos success rate: {chaos_success_rate:.1%}")
        print(f"  Recovery success rate: {recovery_success_rate:.1%}")
        print(f"  Baseline avg time: {baseline_avg_time:.3f}s")
        print(f"  Recovery avg time: {recovery_avg_time:.3f}s")

        # Assert recovery requirements
        assert recovery_success_rate >= baseline_success_rate * 0.9, f"Recovery success rate {recovery_success_rate:.1%} not close to baseline {baseline_success_rate:.1%}"
        assert recovery_avg_time <= baseline_avg_time * 1.2, f"Recovery time {recovery_avg_time:.3f}s significantly slower than baseline {baseline_avg_time:.3f}s"
        assert chaos_success_rate < baseline_success_rate, "Chaos scenario should have reduced success rate"

    async def test_resilience_score_calculation(self, chaos_monkey, resilient_system):
        """Test resilience score calculation across multiple scenarios."""
        print("\n🧪 Testing resilience score calculation...")

        test_scenarios = [
            ('network_partition', lambda: chaos_monkey.inject_network_partition(duration=5.0)),
            ('high_latency', lambda: chaos_monkey.inject_high_latency(duration=6.0, latency_ms=2000)),
            ('memory_pressure', lambda: chaos_monkey.inject_memory_pressure(duration=7.0)),
            ('cpu_spike', lambda: chaos_monkey.inject_cpu_spike(duration=5.0))
        ]

        scenario_results = {}

        for scenario_name, scenario_func in test_scenarios:
            print(f"Testing scenario: {scenario_name}")

            # Inject scenario
            scenario = await scenario_func()

            # Test system under scenario
            results = []
            start_time = time.time()

            for i in range(10):
                result = await resilient_system.process_request(f"{scenario_name}_req_{i}", chaos_monkey)
                results.append(result)
                await asyncio.sleep(0.1)

            end_time = time.time()

            # Wait for scenario to complete
            await asyncio.sleep(scenario.duration + 1)
            await chaos_monkey.cleanup_scenarios()

            # Calculate scenario metrics
            successful_requests = [r for r in results if r['status'] == 'success']
            success_rate = len(successful_requests) / len(results)
            avg_processing_time = sum(r['processing_time'] for r in successful_requests) / len(successful_requests) if successful_requests else 0

            scenario_results[scenario_name] = {
                'availability': success_rate,
                'avg_recovery_time': avg_processing_time,
                'error_rate': 1 - success_rate,
                'throughput_degradation': max(0, (avg_processing_time - 0.2) / 2.0)  # Baseline 0.2s
            }

        # Calculate overall resilience score
        overall_metrics = {
            'availability': sum(r['availability'] for r in scenario_results.values()) / len(scenario_results),
            'avg_recovery_time': sum(r['avg_recovery_time'] for r in scenario_results.values()) / len(scenario_results),
            'error_rate': sum(r['error_rate'] for r in scenario_results.values()) / len(scenario_results),
            'throughput_degradation': sum(r['throughput_degradation'] for r in scenario_results.values()) / len(scenario_results)
        }

        resilience_score = chaos_monkey.calculate_resilience_score(overall_metrics)

        print(f"Resilience score calculation:")
        print(f"  Overall availability: {overall_metrics['availability']:.1%}")
        print(f"  Avg recovery time: {overall_metrics['avg_recovery_time']:.3f}s")
        print(f"  Overall error rate: {overall_metrics['error_rate']:.1%}")
        print(f"  Throughput degradation: {overall_metrics['throughput_degradation']:.1%}")
        print(f"  Final resilience score: {resilience_score:.3f}")

        # Assert resilience score requirements
        assert resilience_score >= 0.6, f"Resilience score {resilience_score:.3f} below threshold (0.6)"
        assert overall_metrics['availability'] >= 0.5, f"Overall availability {overall_metrics['availability']:.1%} below threshold (50%)"

    async def test_cascading_failure_prevention(self, chaos_monkey, resilient_system):
        """Test prevention of cascading failures."""
        print("\n🧪 Testing cascading failure prevention...")

        # Simulate a scenario that could cause cascading failures
        # Start with a minor issue that could escalate

        # Phase 1: Minor latency increase
        minor_scenario = await chaos_monkey.inject_high_latency(duration=5.0, latency_ms=1000)

        phase1_results = []
        for i in range(8):
            result = await resilient_system.process_request(f"cascade_phase1_req_{i}", chaos_monkey)
            phase1_results.append(result)
            await asyncio.sleep(0.1)

        await asyncio.sleep(minor_scenario.duration + 1)
        await chaos_monkey.cleanup_scenarios()

        # Phase 2: Add memory pressure (potential cascade trigger)
        memory_scenario = await chaos_monkey.inject_memory_pressure(duration=6.0, memory_limit_mb=128)

        phase2_results = []
        for i in range(10):
            result = await resilient_system.process_request(f"cascade_phase2_req_{i}", chaos_monkey)
            phase2_results.append(result)
            await asyncio.sleep(0.1)

        await asyncio.sleep(memory_scenario.duration + 1)
        await chaos_monkey.cleanup_scenarios()

        # Phase 3: Add service unavailability (full cascade test)
        service_scenario = await chaos_monkey.inject_service_unavailable(duration=4.0)

        phase3_results = []
        for i in range(6):
            result = await resilient_system.process_request(f"cascade_phase3_req_{i}", chaos_monkey)
            phase3_results.append(result)
            await asyncio.sleep(0.1)

        await asyncio.sleep(service_scenario.duration + 1)
        await chaos_monkey.cleanup_scenarios()

        # Analyze cascading failure prevention
        phase1_success = len([r for r in phase1_results if r['status'] == 'success']) / len(phase1_results)
        phase2_success = len([r for r in phase2_results if r['status'] == 'success']) / len(phase2_results)
        phase3_success = len([r for r in phase3_results if r['status'] == 'success']) / len(phase3_results)

        print(f"Cascading failure prevention results:")
        print(f"  Phase 1 (minor latency) success rate: {phase1_success:.1%}")
        print(f"  Phase 2 (+ memory pressure) success rate: {phase2_success:.1%}")
        print(f"  Phase 3 (+ service unavailable) success rate: {phase3_success:.1%}")

        # Assert cascading failure prevention
        # Success rates should degrade gradually, not catastrophically
        assert phase1_success >= 0.8, f"Phase 1 success rate {phase1_success:.1%} too low for minor issue"
        assert phase2_success >= 0.6, f"Phase 2 success rate {phase2_success:.1%} indicates cascading failure"
        assert phase3_success >= 0.3, f"Phase 3 success rate {phase3_success:.1%} indicates complete system failure"

        # Degradation should be controlled
        phase1_to_phase2_degradation = phase1_success - phase2_success
        phase2_to_phase3_degradation = phase2_success - phase3_success

        assert phase1_to_phase2_degradation <= 0.3, f"Phase 1-2 degradation {phase1_to_phase2_degradation:.1%} too severe"
        assert phase2_to_phase3_degradation <= 0.4, f"Phase 2-3 degradation {phase2_to_phase3_degradation:.1%} too severe"
