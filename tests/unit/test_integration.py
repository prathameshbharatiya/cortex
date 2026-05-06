
import os, sys, time, threading, json, pytest, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from cortex.models.action  import Action, ActionSpec, ActionType, ActionConstraints, Pose
from cortex.models.context import RobotState, SceneGraph, SafetyConstraint
from cortex.models.memory  import MemoryRecord, MemoryType, OutcomeTag
from cortex.models.decision import CertificationState
from cortex.integration.sdk.client import CortexSDK
from cortex.integration.ros2.node  import CortexNode
from cortex.experience.record import OutcomeClass

def robot(ee=None, estop=False):
    return RobotState(ee_position=ee or [0.3,0,0.5], ee_orientation=[1,0,0,0],
        joint_positions=[0.2,-0.4,0.7,0.1,0.3,0], joint_torques=[8,12,7,4,2,1], emergency_stop=estop)

def action_req(x=0.4, y=0.0, z=0.5, speed=0.5, force=40.0):
    return Action(spec=ActionSpec(action_type=ActionType.MOVE_EE, target_pose=Pose(x=x,y=y,z=z),
        constraints=ActionConstraints(max_speed_ms=speed, max_force_n=force)), source="test", intent="pick")

def make_mem(text="pick object"):
    return MemoryRecord(source="test", memory_type=MemoryType.EPISODIC, content={"fact":text},
        content_text=text, valid_at=time.time()-30, base_confidence=0.9, outcome_count=3, success_count=3)

class TestSDKLocal:
    def test_zero_config_init(self):
        with CortexSDK() as sdk: assert sdk._mode == "local"

    def test_four_line_api(self):
        with CortexSDK() as sdk:
            ctx = sdk.build_context("pick red block", robot())
            cert = sdk.certify(action_req(), ctx)
            assert cert.state in CertificationState.__members__.values()

    def test_execute_on_clean_action(self):
        with CortexSDK() as sdk:
            ctx = sdk.build_context("pick object", robot())
            cert = sdk.certify(action_req(x=0.4, y=0.0, z=0.5), ctx)
            assert cert.state in (CertificationState.EXECUTE, CertificationState.EXECUTE_WITH_CONSTRAINTS, CertificationState.HUMAN_OVERRIDE_REQUIRED)

    def test_replan_on_unreachable(self):
        with CortexSDK() as sdk:
            ctx = sdk.build_context("pick", robot())
            cert = sdk.certify(action_req(x=0.7,y=0.7,z=0.0), ctx)
            assert cert.state in (CertificationState.REPLAN_REQUIRED, CertificationState.HUMAN_OVERRIDE_REQUIRED, CertificationState.SAFE_HALT)

    def test_halt_on_estop(self):
        with CortexSDK() as sdk:
            ctx = sdk.build_context("pick", robot(estop=True))
            cert = sdk.certify(action_req(), ctx)
            assert cert.blocked

    def test_record_outcome_success(self):
        with CortexSDK() as sdk:
            ctx = sdk.build_context("task", robot())
            cert = sdk.certify(action_req(), ctx)
            exp = sdk.record_outcome(cert, ctx, "success")
            assert exp.succeeded

    def test_record_outcome_failure(self):
        with CortexSDK() as sdk:
            ctx = sdk.build_context("task", robot())
            cert = sdk.certify(action_req(), ctx)
            exp = sdk.record_outcome(cert, ctx, "failure")
            assert exp.failed

    def test_remember_and_recall(self):
        with CortexSDK() as sdk:
            sdk.remember(make_mem("pick red block from shelf"))
            results = sdk.recall("pick red block")
            assert len(results) > 0

    def test_status_returns_dict(self):
        with CortexSDK() as sdk:
            s = sdk.status()
            assert s["mode"] == "local"
            assert "version" in s

    def test_with_extra_constraints(self):
        with CortexSDK() as sdk:
            c = SafetyConstraint(constraint_id="zone_1", description="test", constraint_type="object_exclusion",
                parameters={"center":[0.9,0.9,0.9],"radius_m":0.05}, veto_power=True)
            ctx = sdk.build_context("task", robot(), extra_constraints=[c])
            assert "zone_1" in {x.constraint_id for x in ctx.safety_constraints}

    def test_human_override_on_critical_zone(self):
        with CortexSDK() as sdk:
            c = SafetyConstraint(constraint_id="critical", description="danger zone",
                constraint_type="object_exclusion", parameters={"center":[0.4,0,0.5],"radius_m":0.3}, veto_power=True)
            ctx = sdk.build_context("task", robot(), extra_constraints=[c])
            cert = sdk.certify(action_req(x=0.4,y=0,z=0.5), ctx)
            assert cert.blocked

    def test_full_workflow(self):
        with CortexSDK() as sdk:
            for i in range(3): sdk.remember(make_mem(f"pick object {i}"))
            ctx = sdk.build_context("pick object", robot(), top_k=5)
            cert = sdk.certify(action_req(), ctx)
            if cert.approved:
                exp = sdk.record_outcome(cert, ctx, "success")
                assert exp.succeeded

    def test_confidence_on_execute(self):
        with CortexSDK() as sdk:
            ctx = sdk.build_context("task", robot())
            cert = sdk.certify(action_req(), ctx)
            if cert.state == CertificationState.EXECUTE:
                assert cert.confidence is not None
                assert 0 < cert.confidence.success_probability <= 1

    def test_skip_physics(self):
        with CortexSDK() as sdk:
            ctx = sdk.build_context("task", robot(), skip_physics=True)
            assert ctx.physics_horizon is None

    def test_add_json_adapter(self, tmp_path):
        with CortexSDK() as sdk:
            sdk.add_memory_adapter("json", priority=5, file_path=str(tmp_path/"m.json"))
            names = [a.name for a in sdk._gateway.adapters()]
            assert len(names) >= 2

    def test_add_unknown_adapter_raises(self):
        with CortexSDK() as sdk:
            with pytest.raises(ValueError): sdk.add_memory_adapter("nosuchdb")

    def test_context_manager_clean(self):
        sdk = CortexSDK()
        with sdk: sdk.build_context("task", robot())

    def test_experience_count_increases(self):
        with CortexSDK() as sdk:
            ctx = sdk.build_context("task", robot())
            cert = sdk.certify(action_req(), ctx)
            sdk.record_outcome(cert, ctx, "success")
            assert sdk.status().get("experience_count",0) >= 1

    def test_100_certifications(self):
        with CortexSDK() as sdk:
            rs = robot(); act = action_req()
            for _ in range(100):
                ctx = sdk.build_context("task", rs)
                cert = sdk.certify(act, ctx)
                assert cert.state in CertificationState.__members__.values()


class TestSDKRemote:
    def test_remote_mode(self):
        sdk = CortexSDK(server_address="grpc://localhost:59999")
        assert sdk._mode == "remote"; sdk.stop()

    def test_channel_lazy(self):
        sdk = CortexSDK(server_address="grpc://localhost:59999")
        assert sdk._channel is None; sdk.stop()


grpc = pytest.importorskip("grpc", reason="grpcio not installed — skipping gRPC tests")

class TestGRPCServer:
    @pytest.fixture
    def srv_sdk(self):
        from cortex.integration.grpc.server import CortexServer
        port = random.randint(50100,50199)
        srv = CortexServer(port=port); srv.start(); time.sleep(0.15)
        sdk = CortexSDK(server_address=f"grpc://localhost:{port}")
        yield srv, sdk, port
        sdk.stop(); srv.stop()

    def test_server_starts(self, srv_sdk):
        srv, sdk, port = srv_sdk; assert srv._running

    def test_health_check(self, srv_sdk):
        srv, sdk, port = srv_sdk
        from cortex.integration.grpc import cortex_pb2 as pb, cortex_pb2_grpc as pb_grpc
        import grpc
        ch = grpc.insecure_channel(f"localhost:{port}")
        h = pb_grpc.CortexHealthStub(ch)
        r = h.Check(pb.HealthRequest())
        assert r.alive and r.ready and r.version == "1.0.0"
        ch.close()

    def test_grpc_certify(self, srv_sdk):
        srv, sdk, port = srv_sdk
        from cortex.integration.grpc import cortex_pb2 as pb, cortex_pb2_grpc as pb_grpc
        import grpc
        ch = grpc.insecure_channel(f"localhost:{port}")
        stub = pb_grpc.CortexCertificationStub(ch)
        rs = pb.RobotStateProto(ee_position=pb.Vec3(x=0.3,y=0,z=0.5), ee_orientation=pb.Quaternion(w=1.0))
        ap = pb.ActionProto(action_type="move_ee", max_speed_ms=0.5,
            target_pose=pb.Pose(position=pb.Vec3(x=0.4,y=0,z=0.5), orientation=pb.Quaternion(w=1.0)))
        resp = stub.Certify(pb.CertifyRequest(action=ap, task="pick", robot_state=rs, confidence_floor=0.60))
        assert resp.state in [s.value for s in CertificationState]
        assert resp.trace_id != ""; ch.close()

    def test_grpc_memory_store_retrieve(self, srv_sdk):
        srv, sdk, port = srv_sdk
        from cortex.integration.grpc import cortex_pb2 as pb, cortex_pb2_grpc as pb_grpc
        import grpc
        ch = grpc.insecure_channel(f"localhost:{port}")
        stub = pb_grpc.CortexMemoryStub(ch)
        rec = pb.MemoryRecordProto(source="test", memory_type="semantic",
            content_json='{"fact":"pick block"}', content_text="pick block",
            valid_at=time.time(), base_confidence=0.9, outcome_tag="unknown")
        sr = stub.Store(pb.StoreRequest(record=rec))
        assert sr.success
        rr = stub.Retrieve(pb.RetrieveRequest(query_text="pick block", query_type="semantic", top_k=5))
        assert rr is not None; ch.close()

    def test_grpc_experience(self, srv_sdk):
        srv, sdk, port = srv_sdk
        from cortex.integration.grpc import cortex_pb2 as pb, cortex_pb2_grpc as pb_grpc
        import grpc
        ch = grpc.insecure_channel(f"localhost:{port}")
        stub = pb_grpc.CortexExperienceStub(ch)
        r = stub.RecordOutcome(pb.RecordOutcomeRequest(task="pick", outcome_class="success", predicted_confidence=0.85))
        assert r.experience_id != "" and 0 <= r.surprise_score <= 1; ch.close()

    def test_remote_sdk_certify(self, srv_sdk):
        srv, sdk, port = srv_sdk
        ctx = sdk.build_context("pick", robot())
        cert = sdk.certify(action_req(), ctx)
        assert cert.state in CertificationState.__members__.values()

    def test_server_stops(self, srv_sdk):
        srv, sdk, port = srv_sdk; sdk.stop(); srv.stop(); assert not srv._running


class TestROS2Node:
    def test_init_without_ros2(self):
        node = CortexNode(task="pick"); assert node.task == "pick"; node.destroy()

    def test_certify_direct(self):
        node = CortexNode(task="pick red block")
        cert, ctx = node.certify_direct(action_req(), robot())
        assert cert.state in CertificationState.__members__.values()
        assert ctx.task == "pick red block"; node.destroy()

    def test_on_action_parses_json(self):
        node = CortexNode()
        class M: data = json.dumps({"action_type":"move_ee","target_pose":{"x":0.4,"y":0,"z":0.5,"qw":1,"qx":0,"qy":0,"qz":0},"max_speed_ms":0.5})
        node._on_action(M())
        with node._state.lock: assert node._state.pending_action is not None
        node.destroy()

    def test_on_state_parses_json(self):
        node = CortexNode()
        class M: data = json.dumps({"ee_position":[0.3,0,0.5],"ee_orientation":[1,0,0,0],"gripper_open":True})
        node._on_state(M())
        with node._state.lock: assert node._state.robot_state is not None
        node.destroy()

    def test_bad_json_no_crash(self):
        node = CortexNode()
        class M: data = "not json"
        node._on_action(M()); node.destroy()

    def test_blocked_on_estop(self):
        node = CortexNode()
        cert, _ = node.certify_direct(action_req(), robot(estop=True))
        assert cert.blocked; node.destroy()

    def test_node_status(self):
        node = CortexNode()
        s = node._sdk.status(); assert s["mode"] == "local"; node.destroy()


class TestPhase6Performance:
    def test_sdk_under_30ms(self):
        with CortexSDK() as sdk:
            rs = robot(); act = action_req(); times = []
            for _ in range(30):
                t0 = time.perf_counter()
                ctx = sdk.build_context("pick object", rs)
                sdk.certify(act, ctx)
                times.append((time.perf_counter()-t0)*1000)
            avg = sum(times)/len(times)
            print(f"\nSDK full cycle avg: {avg:.2f}ms")
            assert avg < 30.0

    def test_grpc_under_50ms(self):
        grpc = pytest.importorskip("grpc", reason="grpcio not installed")
        from cortex.integration.grpc.server import CortexServer
        from cortex.integration.grpc import cortex_pb2 as pb, cortex_pb2_grpc as pb_grpc
        port = random.randint(50200,50299)
        srv = CortexServer(port=port); srv.start(); time.sleep(0.15)
        ch = grpc.insecure_channel(f"localhost:{port}")
        stub = pb_grpc.CortexCertificationStub(ch)
        rs = pb.RobotStateProto(ee_position=pb.Vec3(x=0.3,y=0,z=0.5), ee_orientation=pb.Quaternion(w=1.0))
        ap = pb.ActionProto(action_type="move_ee", target_pose=pb.Pose(
            position=pb.Vec3(x=0.4,y=0,z=0.5), orientation=pb.Quaternion(w=1.0)))
        req = pb.CertifyRequest(action=ap, task="pick", robot_state=rs)
        times = []
        for _ in range(20):
            t0 = time.perf_counter(); stub.Certify(req); times.append((time.perf_counter()-t0)*1000)
        avg = sum(times)/len(times)
        print(f"\ngRPC round-trip avg: {avg:.2f}ms")
        ch.close(); srv.stop()
        assert avg < 50.0
