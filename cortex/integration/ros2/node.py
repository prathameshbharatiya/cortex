"""
Cortex ROS 2 Node — integrates Cortex into any ROS 2 robotics stack.
"""
from __future__ import annotations
import json, threading, time
from typing import Any

from cortex.models.action  import Action, ActionSpec, ActionType, ActionConstraints, Pose
from cortex.models.context import RobotState, SceneGraph
from cortex.models.memory  import MemoryRecord, MemoryType
from cortex.models.decision import CertificationState
from cortex.integration.sdk.client import CortexSDK

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String, Float64
    from std_srvs.srv import Trigger
    _ROS2 = True
except ImportError:
    _ROS2 = False
    class Node:
        def __init__(self, *a, **kw): pass

class _State:
    def __init__(self):
        self.pending_action = None
        self.robot_state    = None
        self.last_cert_state = None
        self.last_confidence = 0.0
        self.last_decision_id = ""
        self.lock = threading.RLock()

class CortexNode(Node if _ROS2 else object):
    """
    Cortex ROS 2 node.

    Publishes: /cortex/certification_state, /cortex/confidence, /cortex/health
    Subscribes: /cortex/action_request, /cortex/robot_state, /cortex/outcome
    Services:   /cortex/certify, /cortex/get_context, /cortex/status

    Without ROS 2, use certify_direct() for testing.
    """

    def __init__(self, task="robot task", sdk=None, deployment_id="", platform_id="", publish_rate_hz=10.0):
        self.task = task
        self._state = _State()
        self._sdk = sdk or CortexSDK(deployment_id=deployment_id, platform_id=platform_id)
        if _ROS2:
            super().__init__("cortex_node")
            self._pub_state  = self.create_publisher(String,  "/cortex/certification_state", 10)
            self._pub_conf   = self.create_publisher(Float64, "/cortex/confidence",          10)
            self._pub_health = self.create_publisher(String,  "/cortex/health",              10)
            self.create_subscription(String, "/cortex/action_request",  self._on_action,  10)
            self.create_subscription(String, "/cortex/robot_state",     self._on_state,   10)
            self.create_subscription(String, "/cortex/outcome",         self._on_outcome, 10)
            self.create_service(Trigger, "/cortex/certify",     self._svc_certify)
            self.create_service(Trigger, "/cortex/get_context", self._svc_context)
            self.create_service(Trigger, "/cortex/status",      self._svc_status)
            self.create_timer(1.0/publish_rate_hz, self._tick)

    def _on_action(self, msg):
        try:
            d = json.loads(msg.data)
            tp = d.get("target_pose", {})
            pose = Pose(x=tp.get("x",0), y=tp.get("y",0), z=tp.get("z",0),
                        qw=tp.get("qw",1), qx=tp.get("qx",0), qy=tp.get("qy",0), qz=tp.get("qz",0)) if tp else None
            try: at = ActionType(d.get("action_type","move_ee"))
            except: at = ActionType.MOVE_EE
            a = Action(spec=ActionSpec(action_type=at, target_pose=pose,
                constraints=ActionConstraints(max_speed_ms=d.get("max_speed_ms"), max_force_n=d.get("max_force_n"))),
                source=d.get("source","ros2"), intent=d.get("intent",""), confidence=d.get("confidence",1.0))
            with self._state.lock: self._state.pending_action = a
        except Exception as e: self._warn(f"bad action: {e}")

    def _on_state(self, msg):
        try:
            d = json.loads(msg.data)
            rs = RobotState(
                ee_position=d.get("ee_position",[0,0,0]),
                ee_orientation=d.get("ee_orientation",[1,0,0,0]),
                joint_positions=d.get("joint_positions",[]),
                joint_torques=d.get("joint_torques",[]),
                gripper_open=d.get("gripper_open",True),
                emergency_stop=d.get("emergency_stop",False),
            )
            with self._state.lock: self._state.robot_state = rs
        except Exception as e: self._warn(f"bad state: {e}")

    def _on_outcome(self, msg):
        try:
            d = json.loads(msg.data)
            oc = d.get("outcome_class","success")
            with self._state.lock:
                rs = self._state.robot_state or self._default_rs()
                last = self._state.last_cert_state
            ctx = self._sdk.build_context(self.task, rs, skip_physics=True)
            from cortex.models.decision import CertificationDecision, DecisionTrace
            trace = DecisionTrace(ctx_id=ctx.ctx_id, action_id="ros2",
                                  certification_state=last or CertificationState.EXECUTE)
            dec = CertificationDecision(state=last or CertificationState.EXECUTE,
                                        action=None, trace=trace, reason="ros2")
            self._sdk.record_outcome(dec, ctx, oc, d.get("description",""))
        except Exception as e: self._warn(f"outcome error: {e}")

    def _svc_certify(self, req, resp):
        with self._state.lock:
            action = self._state.pending_action
            rs     = self._state.robot_state or self._default_rs()
        if action is None:
            resp.success = False; resp.message = "No pending action"; return resp
        try:
            ctx  = self._sdk.build_context(self.task, rs)
            cert = self._sdk.certify(action, ctx)
            with self._state.lock:
                self._state.last_cert_state  = cert.state
                self._state.last_confidence  = cert.confidence.success_probability if cert.confidence else 0.0
                self._state.last_decision_id = cert.decision_id
            resp.success = cert.approved
            resp.message = json.dumps({"state": cert.state.value, "decision_id": cert.decision_id,
                                       "confidence": self._state.last_confidence, "reason": cert.reason})
        except Exception as e:
            resp.success = False; resp.message = f"Error: {e}"
        return resp

    def _svc_context(self, req, resp):
        with self._state.lock: rs = self._state.robot_state or self._default_rs()
        try:
            ctx = self._sdk.build_context(self.task, rs)
            resp.success = True
            resp.message = json.dumps({"ctx_id": ctx.ctx_id, "task": ctx.task,
                "memory_records": len(ctx.memory_records), "constraints": len(ctx.safety_constraints),
                "memory_confidence": ctx.memory_confidence, "physics_feasible": ctx.physics_feasible})
        except Exception as e:
            resp.success = False; resp.message = f"Error: {e}"
        return resp

    def _svc_status(self, req, resp):
        resp.success = True; resp.message = json.dumps(self._sdk.status()); return resp

    def _tick(self):
        if not _ROS2: return
        with self._state.lock: state, conf = self._state.last_cert_state, self._state.last_confidence
        m = String(); m.data = state.value if state else "UNKNOWN"; self._pub_state.publish(m)
        f = Float64(); f.data = conf; self._pub_conf.publish(f)

    def certify_direct(self, action, robot_state=None):
        """Direct certification without ROS 2 (testing + local use)."""
        rs = robot_state or self._state.robot_state or self._default_rs()
        ctx  = self._sdk.build_context(self.task, rs)
        cert = self._sdk.certify(action, ctx)
        with self._state.lock:
            self._state.last_cert_state  = cert.state
            self._state.last_confidence  = cert.confidence.success_probability if cert.confidence else 0.0
        return cert, ctx

    def spin(self):
        if _ROS2: rclpy.spin(self)
        else: raise RuntimeError("rclpy not installed")

    def destroy(self):
        self._sdk.stop()
        if _ROS2: super().destroy_node()

    @staticmethod
    def _default_rs():
        return RobotState(ee_position=[0.3,0,0.5], ee_orientation=[1,0,0,0])

    def _warn(self, msg):
        if _ROS2: self.get_logger().warn(f"[Cortex] {msg}")
        else: print(f"[CORTEX WARN] {msg}")

def main(args=None):
    if not _ROS2: print("rclpy not available."); return
    rclpy.init(args=args)
    node = CortexNode()
    try: node.spin()
    except KeyboardInterrupt: pass
    finally: node.destroy(); rclpy.shutdown()

if __name__ == "__main__":
    main()
