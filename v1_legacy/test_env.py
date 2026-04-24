"""
test_env.py — Validates OnCallEnv works correctly.

Run: python test_env.py
Requires: pip install -r requirements.txt
"""

import sys
import json
from environment import OnCallEnvironment
from models import Action
from graders import grade_task


def test_easy_optimal():
    """Test easy task with optimal action sequence including mark_resolved."""
    env = OnCallEnvironment()
    obs = env.reset("easy_memory_leak")
    assert obs.task_id == "easy_memory_leak"
    assert obs.step == 0
    assert len(obs.alerts) == 3
    print("  [PASS] Reset returns valid observation")

    # Step 1: Check logs of payment-service
    r = env.step(Action(command="check_logs payment-service"))
    assert not r.done
    assert "OutOfMemoryError" in r.observation.last_action_result
    print("  [PASS] check_logs shows OOM errors")

    # Step 2: Check metrics (dynamic: memory may degrade slightly from 94.7% baseline)
    r = env.step(Action(command="check_metrics payment-service"))
    assert "Memory usage:" in r.observation.last_action_result
    # Memory should be very high (>90%) for the payment service with a memory leak
    import re as _re
    mem_match = _re.search(r"Memory usage:\s+([\d.]+)%", r.observation.last_action_result)
    assert mem_match and float(mem_match.group(1)) > 90
    print("  [PASS] check_metrics shows high memory")

    # Step 3: Restart
    r = env.step(Action(command="restart_service payment-service"))
    assert not r.done  # Agent gets extra steps to mark_resolved
    assert "healthy" in r.observation.last_action_result.lower()
    print("  [PASS] Restart fixes service, episode continues for mark_resolved")

    # Step 4: Mark resolved
    r = env.step(Action(command="mark_resolved payment-service memory leak due to OOM kills"))
    assert r.done
    assert r.reward.total >= 0.9
    print(f"  [PASS] mark_resolved completes incident (score: {r.reward.total})")

    # Grader
    state = env.state()
    score = grade_task("easy_memory_leak", state)
    assert 0.0 <= score <= 1.0
    assert score >= 0.9
    print(f"  [PASS] Grader returns valid score: {score}")
    return score


def test_medium_optimal():
    """Test medium task with optimal action sequence."""
    env = OnCallEnvironment()
    env.reset("medium_cascading_failure")

    # Investigate the chain
    env.step(Action(command="check_metrics api-gateway"))
    env.step(Action(command="check_logs api-gateway"))
    env.step(Action(command="check_metrics order-service"))
    env.step(Action(command="check_logs order-service"))
    r = env.step(Action(command="check_config order-service"))
    assert "db_pool_size" in r.observation.last_action_result
    assert "5" in r.observation.last_action_result
    print("  [PASS] Config shows db_pool_size = 5")

    # Fix it
    r = env.step(Action(command="update_config order-service db_pool_size 50"))
    assert not r.done
    assert "resolved" in r.observation.last_action_result.lower()
    print("  [PASS] Config update fixes the issue")

    # Mark resolved
    r = env.step(Action(command="mark_resolved order-service db_pool_size connection pool exhausted config changed to 5"))
    assert r.done
    assert r.reward.total >= 0.9
    print(f"  [PASS] mark_resolved completes incident (score: {r.reward.total})")

    state = env.state()
    score = grade_task("medium_cascading_failure", state)
    assert score >= 0.8
    print(f"  [PASS] Grader score: {score}")
    return score


def test_hard_optimal():
    """Test hard task with optimal action sequence."""
    env = OnCallEnvironment()
    env.reset("hard_cache_degradation")

    # Broad investigation
    env.step(Action(command="check_metrics api-gateway"))
    env.step(Action(command="check_metrics order-service"))
    env.step(Action(command="check_metrics product-service"))
    env.step(Action(command="check_metrics cache-service"))
    env.step(Action(command="check_logs cache-service"))
    r = env.step(Action(command="check_deploy_history cache-service"))
    assert "MurmurHash3" in r.observation.last_action_result or "hashing" in r.observation.last_action_result.lower()
    print("  [PASS] Deploy history reveals hashing change")

    env.step(Action(command="check_metrics postgres-primary"))

    # Rollback cache
    r = env.step(Action(command="rollback_deploy cache-service"))
    assert not r.done
    print("  [PASS] Rollback fixes cache, episode continues")

    # Mark resolved
    r = env.step(Action(command="mark_resolved cache-service deployment changed key hashing algorithm causing cache miss"))
    assert r.done
    assert r.reward.total >= 0.9
    print(f"  [PASS] mark_resolved completes incident (score: {r.reward.total})")

    state = env.state()
    score = grade_task("hard_cache_degradation", state)
    assert score >= 0.8
    print(f"  [PASS] Grader score: {score}")
    return score


def test_dns_optimal():
    """Test DNS misconfiguration scenario."""
    env = OnCallEnvironment()
    obs = env.reset("medium_dns_misconfiguration")
    assert obs.task_id == "medium_dns_misconfiguration"
    print("  [PASS] Reset works")

    env.step(Action(command="check_metrics order-service"))
    env.step(Action(command="check_logs order-service"))
    r = env.step(Action(command="check_config order-service"))
    assert "inventory-service-v2.internal" in r.observation.last_action_result
    print("  [PASS] Config shows wrong hostname")

    env.step(Action(command="check_metrics inventory-service"))

    r = env.step(Action(command="update_config order-service inventory_host inventory-service.internal"))
    assert not r.done
    print("  [PASS] Config fix applied")

    r = env.step(Action(command="mark_resolved order-service dns hostname misconfiguration inventory_host pointed to wrong host"))
    assert r.done
    assert r.reward.total >= 0.9
    print(f"  [PASS] DNS scenario completed (score: {r.reward.total})")

    state = env.state()
    score = grade_task("medium_dns_misconfiguration", state)
    assert score >= 0.8
    print(f"  [PASS] Grader score: {score}")
    return score


def test_replication_lag_optimal():
    """Test DB replication lag scenario."""
    env = OnCallEnvironment()
    obs = env.reset("hard_replication_lag")
    assert obs.task_id == "hard_replication_lag"
    print("  [PASS] Reset works")

    env.step(Action(command="check_metrics user-service"))
    env.step(Action(command="check_logs user-service"))
    env.step(Action(command="check_metrics order-service"))
    env.step(Action(command="check_metrics postgres-primary"))
    env.step(Action(command="check_logs postgres-primary"))
    env.step(Action(command="check_config postgres-primary"))
    env.step(Action(command="check_metrics postgres-replica"))
    print("  [PASS] Investigation chain complete")

    r = env.step(Action(command="update_config postgres-primary batch_job_enabled false"))
    assert not r.done
    print("  [PASS] Batch job disabled")

    r = env.step(Action(command="mark_resolved postgres-primary batch job nightly_aggregation causing replication lag"))
    assert r.done
    assert r.reward.total >= 0.8
    print(f"  [PASS] Replication lag scenario completed (score: {r.reward.total})")

    state = env.state()
    score = grade_task("hard_replication_lag", state)
    assert score >= 0.7
    print(f"  [PASS] Grader score: {score}")
    return score


def test_wrong_actions():
    """Test that wrong actions get penalized."""
    env = OnCallEnvironment()
    env.reset("easy_memory_leak")

    # Restart wrong service
    r = env.step(Action(command="restart_service user-service"))
    assert not r.done
    print("  [PASS] Restarting wrong service doesn't resolve")

    # Check state has penalty
    state = env.state()
    assert state.reward_breakdown.get("penalty", 0) < 0
    print("  [PASS] Penalty applied for wrong action")


def test_max_steps():
    """Test episode ends at max steps."""
    env = OnCallEnvironment()
    env.reset("easy_memory_leak")

    # Burn through all steps with no-ops
    for i in range(10):
        r = env.step(Action(command="check_metrics api-gateway"))
    assert r.done
    print(f"  [PASS] Episode ends at max steps (score: {r.reward.total})")


def test_invalid_commands():
    """Test error handling for invalid commands."""
    env = OnCallEnvironment()
    env.reset("easy_memory_leak")

    r = env.step(Action(command="delete_everything"))
    assert r.observation.last_action_error
    print("  [PASS] Invalid command returns error")

    r = env.step(Action(command="check_metrics nonexistent-service"))
    assert r.observation.last_action_error
    print("  [PASS] Unknown service returns error")


def test_list_tasks():
    """Test task listing."""
    env = OnCallEnvironment()
    tasks = env.list_tasks()
    assert len(tasks) == 6
    difficulties = {t["difficulty"] for t in tasks}
    assert "easy" in difficulties
    assert "medium" in difficulties
    assert "hard" in difficulties
    assert "expert" in difficulties
    print(f"  [PASS] {len(tasks)} tasks with difficulty range")


def test_state_endpoint():
    """Test state returns valid data."""
    env = OnCallEnvironment()
    env.reset("easy_memory_leak")
    env.step(Action(command="check_logs payment-service"))

    state = env.state()
    assert state.task_id == "easy_memory_leak"
    assert state.step == 1
    assert len(state.actions_taken) == 1
    assert "payment-service" in state.investigation_log
    print("  [PASS] State endpoint returns correct data")


def test_score_range():
    """Verify all scores are in [0.0, 1.0]."""
    env = OnCallEnvironment()

    for task_id in ["easy_memory_leak", "medium_cascading_failure", "hard_cache_degradation",
                    "medium_dns_misconfiguration", "hard_replication_lag",
                    "expert_multi_root_cause"]:
        env.reset(task_id)
        for _ in range(5):
            r = env.step(Action(command="check_metrics api-gateway"))
        state = env.state()
        assert 0.0 <= state.score <= 1.0, f"{task_id}: score {state.score} out of range"
    print("  [PASS] All scores in [0.0, 1.0]")


def test_mark_resolved_positive():
    """Test mark_resolved with correct keywords gives full root cause credit."""
    env = OnCallEnvironment()
    env.reset("easy_memory_leak")
    env.step(Action(command="check_logs payment-service"))
    env.step(Action(command="restart_service payment-service"))
    r = env.step(Action(command="mark_resolved payment-service memory leak OOM heap"))
    assert r.done
    state = env.state()
    assert state.root_cause_identified
    print(f"  [PASS] Correct mark_resolved (score: {state.score})")


def test_mark_resolved_negative():
    """Test mark_resolved with wrong keywords doesn't give full credit."""
    env = OnCallEnvironment()
    env.reset("easy_memory_leak")
    r = env.step(Action(command="mark_resolved everything is broken somewhere"))
    assert not r.done
    state = env.state()
    assert not state.root_cause_identified
    print("  [PASS] Wrong mark_resolved rejected")


def test_mark_resolved_partial():
    """Test mark_resolved with partial keywords gives partial credit."""
    env = OnCallEnvironment()
    env.reset("easy_memory_leak")
    r = env.step(Action(command="mark_resolved memory issue detected"))
    state = env.state()
    assert state.root_cause_identified  # partial: has 1 keyword
    print("  [PASS] Partial mark_resolved gives partial credit")


def test_remediation_without_mark_resolved():
    """Test that correct remediation without mark_resolved still ends eventually."""
    env = OnCallEnvironment()
    env.reset("easy_memory_leak")
    env.step(Action(command="restart_service payment-service"))
    # 2 more steps allowed after remediation
    r = env.step(Action(command="check_metrics api-gateway"))
    assert not r.done  # step 1 after remediation
    r = env.step(Action(command="check_metrics api-gateway"))
    assert r.done  # step 2 after remediation — auto-ends
    state = env.state()
    assert state.score >= 0.3  # Gets remediation credit but no root cause or efficiency
    print(f"  [PASS] Episode ends 2 steps after remediation (score: {state.score})")


def test_expert_optimal():
    """Test expert multi-root-cause scenario with both fixes."""
    env = OnCallEnvironment()
    obs = env.reset("expert_multi_root_cause")
    assert obs.task_id == "expert_multi_root_cause"
    assert len(obs.alerts) >= 3
    print("  [PASS] Reset works")

    # Investigate both failure chains
    env.step(Action(command="check_metrics api-gateway"))
    env.step(Action(command="check_logs api-gateway"))
    env.step(Action(command="check_metrics search-service"))
    env.step(Action(command="check_logs search-service"))
    r = env.step(Action(command="check_deploy_history search-service"))
    assert "v3.1.0" in r.observation.last_action_result
    print("  [PASS] Search deploy history shows broken deployment")

    env.step(Action(command="check_metrics order-service"))
    env.step(Action(command="check_logs order-service"))
    r = env.step(Action(command="check_config order-service"))
    assert "db_pool_size" in r.observation.last_action_result
    print("  [PASS] Order config shows low pool size")

    env.step(Action(command="check_metrics elasticsearch"))

    # Fix 1: rollback search-service
    r = env.step(Action(command="rollback_deploy search-service"))
    assert not r.done
    assert "1/2" in r.observation.last_action_result
    print("  [PASS] First fix applied (1/2)")

    # Fix 2: update order-service config
    r = env.step(Action(command="update_config order-service db_pool_size 50"))
    assert not r.done
    assert "resolved" in r.observation.last_action_result.lower() or "2/2" in r.observation.last_action_result
    print("  [PASS] Second fix applied (2/2)")

    # Mark resolved
    r = env.step(Action(command="mark_resolved search-service bad deployment v3.1.0 elasticsearch query AND order-service db_pool_size config drift both issues"))
    assert r.done
    assert r.reward.total >= 0.8
    print(f"  [PASS] Expert scenario completed (score: {r.reward.total})")

    state = env.state()
    score = grade_task("expert_multi_root_cause", state)
    assert score >= 0.7
    print(f"  [PASS] Grader score: {score}")
    return score


def test_grader_independence():
    """Test that graders compute scores independently from environment reward."""
    env = OnCallEnvironment()
    env.reset("easy_memory_leak")
    env.step(Action(command="check_logs payment-service"))
    env.step(Action(command="check_metrics payment-service"))
    env.step(Action(command="restart_service payment-service"))
    env.step(Action(command="mark_resolved payment-service memory leak OOM"))

    state = env.state()
    env_score = state.score
    grader_score = grade_task("easy_memory_leak", state)

    # Both should be high (may differ slightly since they compute independently)
    assert grader_score >= 0.8
    assert env_score >= 0.8
    print(f"  [PASS] Grader ({grader_score}) and env ({env_score}) both score high")


if __name__ == "__main__":
    tests = [
        ("Easy optimal run", test_easy_optimal),
        ("Medium optimal run", test_medium_optimal),
        ("Hard optimal run", test_hard_optimal),
        ("DNS misconfiguration optimal", test_dns_optimal),
        ("DB replication lag optimal", test_replication_lag_optimal),
        ("Expert multi-root-cause optimal", test_expert_optimal),
        ("Wrong actions penalty", test_wrong_actions),
        ("Max steps termination", test_max_steps),
        ("Invalid commands", test_invalid_commands),
        ("Task listing", test_list_tasks),
        ("State endpoint", test_state_endpoint),
        ("Score range validation", test_score_range),
        ("mark_resolved positive", test_mark_resolved_positive),
        ("mark_resolved negative", test_mark_resolved_negative),
        ("mark_resolved partial", test_mark_resolved_partial),
        ("Remediation without mark_resolved", test_remediation_without_mark_resolved),
        ("Grader independence", test_grader_independence),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\n{'─'*50}")
        print(f"TEST: {name}")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'═'*50}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
    print("All tests passed!")
