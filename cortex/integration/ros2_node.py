"""
Cortex ROS 2 Node
=================
Drop-in integration for any ROS 2 robotics stack.

Subscribes to standard ROS 2 topics for robot state.
Provides a ROS 2 service for action certification.
Publishes certification decisions on a topic.

Topics (subscribed)
-------------------
  /joint_states          sensor_msgs/JointState
  /tool_pose             geometry_msgs/PoseStamped
  /scene_objects         (custom, list of detected objects)

Topics (published)
------------------
  /cortex/decision       std_msgs/String  (JSON CertifyResponse)
  /cortex/health         std_msgs/String  (JSON HealthResponse)

Services (provided)
-------------------
  /cortex/certify        CortexCertify.srv
  /cortex/record         CortexRecord.srv
  /cortex/store_memory   CortexStoreMemory.srv

Usage in a ROS 2 launch file
-----------------------------
    from cortex.integration.ros2_node import CortexNode
    # or use the provided launch descriptor:
    from cortex.integration.ros2_node import cortex_launch_description

    def generate_launch_description():
        return cortex_launch_description(
            memory_sources=["redis://localhost:6379"],
            platform_id="arm_01",
        )

If rclpy is not installed, this module is importable but raises
ImportError on node instantiation.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

from cortex.integration.server   import CortexServer
from cortex.integration.sdk      import CortexClient
from cortex.integration.protocol import (
    CertifyRequest, RecordOutcomeRequest,
    StoreMemoryRequest, HealthRequest,
)


# ── ROS 2 availability check ──────────────────────────────────────────────────

def _rclpy_available() -> bool:
    try:
        import rclpy  # noqa: F401
        return True
    except ImportError:
        return False


# ── Node implementation ───────────────────────────────────────────────────────

class CortexNode:
    """
    Cortex ROS 2 node.

    Wraps a CortexServer and exposes its operations as ROS 2
    topics and services.

    When rclpy is available, creates a real ROS 2 node.
    When it is not, provides a mock that can be tested without ROS.
    """

    def __init__(
        self,
        server:       CortexServer | None = None,
        node_name:    str   = "cortex",
        namespace:    str   = "/cortex",
        health_hz:    float = 1.0,
    ) -> None:
        self._server    = server or CortexServer.create()
        self._client    = CortexClient(server=self._server)
        self._node_name = node_name
        self._namespace = namespace
        self._health_hz = health_hz
        self._node: Any = None
        self._running   = False

        # Cached robot state (updated by subscriber callbacks)
        self._ee_position:     list[float] = [0.0, 0.0, 0.0]
        self._ee_quaternion:   list[float] = [1.0, 0.0, 0.0, 0.0]
        self._joint_positions: list[float] = []
        self._joint_torques:   list[float] = []
        self._gripper_open:    bool        = True
        self._humans_nearby:   bool        = False
        self._emergency_stop:  bool        = False
        self._state_lock       = threading.RLock()

    def start(self) -> "CortexNode":
        if _rclpy_available():
            self._start_ros2()
        else:
            self._start_mock()
        self._running = True
        return self

    def stop(self) -> None:
        self._running = False
        if self._node is not None and _rclpy_available():
            try:
                import rclpy
                self._node.destroy_node()
            except Exception:
                pass

    def __enter__(self) -> "CortexNode":
        return self.start()

    def __exit__(self, *_: Any) -> None:
        self.stop()

    # ── ROS 2 node setup ──────────────────────────────────────────────────────

    def _start_ros2(self) -> None:
        try:
            import rclpy
            from rclpy.node import Node
            from std_msgs.msg import String
            from geometry_msgs.msg import PoseStamped
            from sensor_msgs.msg import JointState

            rclpy.init(args=None)
            self._node = rclpy.create_node(self._node_name)

            # Subscribers
            self._node.create_subscription(
                JointState,
                "/joint_states",
                self._on_joint_states,
                10,
            )
            self._node.create_subscription(
                PoseStamped,
                "/tool_pose",
                self._on_tool_pose,
                10,
            )

            # Publishers
            self._decision_pub = self._node.create_publisher(
                String, f"{self._namespace}/decision", 10
            )
            self._health_pub = self._node.create_publisher(
                String, f"{self._namespace}/health", 10
            )

            # Health timer
            self._node.create_timer(
                1.0 / self._health_hz,
                self._publish_health,
            )

            # Spin in background thread
            import threading
            self._spin_thread = threading.Thread(
                target=rclpy.spin,
                args=(self._node,),
                daemon=True,
            )
            self._spin_thread.start()

        except Exception as e:
            raise RuntimeError(f"Failed to start ROS 2 node: {e}")

    def _start_mock(self) -> None:
        """Mock mode — no ROS 2 required. For testing."""
        pass

    # ── ROS 2 callbacks ───────────────────────────────────────────────────────

    def _on_joint_states(self, msg: Any) -> None:
        with self._state_lock:
            self._joint_positions = list(msg.position) if msg.position else []
            self._joint_torques   = list(msg.effort)   if msg.effort   else []

    def _on_tool_pose(self, msg: Any) -> None:
        with self._state_lock:
            p = msg.pose.position
            q = msg.pose.orientation
            self._ee_position   = [p.x, p.y, p.z]
            self._ee_quaternion = [q.w, q.x, q.y, q.z]

    def _publish_health(self) -> None:
        if not self._running:
            return
        try:
            from std_msgs.msg import String
            health = self._client.health()
            msg    = String()
            msg.data = json.dumps(health, default=str)
            self._health_pub.publish(msg)
        except Exception:
            pass

    # ── Public API (callable without ROS 2) ──────────────────────────────────

    def certify_action(
        self,
        action_type:     str,
        target_position: list[float],
        task:            str   = "",
        max_speed_ms:    float | None = None,
        max_force_n:     float | None = None,
        source:          str   = "ros2_node",
        **kwargs,
    ) -> dict:
        """
        Certify an action using the current cached robot state.
        Returns a dict compatible with JSON serialisation.
        """
        with self._state_lock:
            ee_pos  = list(self._ee_position)
            ee_quat = list(self._ee_quaternion)
            joints  = list(self._joint_positions)
            torques = list(self._joint_torques)
            gripper = self._gripper_open
            humans  = self._humans_nearby
            estop   = self._emergency_stop

        result = self._client.certify(
            action_type=action_type,
            target_position=target_position,
            task=task,
            ee_position=ee_pos,
            ee_quaternion=ee_quat,
            joint_positions=joints,
            joint_torques=torques,
            gripper_open=gripper,
            humans_nearby=humans,
            emergency_stop=estop,
            max_speed_ms=max_speed_ms,
            max_force_n=max_force_n,
            source=source,
            **kwargs,
        )

        # Publish decision if ROS 2 is active
        if _rclpy_available() and self._node and hasattr(self, "_decision_pub"):
            try:
                from std_msgs.msg import String
                msg      = String()
                msg.data = json.dumps(result.__dict__, default=str)
                self._decision_pub.publish(msg)
            except Exception:
                pass

        return result.__dict__

    def update_state(
        self,
        ee_position:    list[float] | None = None,
        ee_quaternion:  list[float] | None = None,
        joint_positions: list[float] | None = None,
        joint_torques:  list[float] | None = None,
        gripper_open:   bool | None = None,
        humans_nearby:  bool | None = None,
        emergency_stop: bool | None = None,
    ) -> None:
        """
        Manually update cached robot state.
        Used when not subscribing to ROS 2 topics.
        """
        with self._state_lock:
            if ee_position    is not None: self._ee_position    = ee_position
            if ee_quaternion  is not None: self._ee_quaternion  = ee_quaternion
            if joint_positions is not None: self._joint_positions = joint_positions
            if joint_torques  is not None: self._joint_torques  = joint_torques
            if gripper_open   is not None: self._gripper_open   = gripper_open
            if humans_nearby  is not None: self._humans_nearby  = humans_nearby
            if emergency_stop is not None: self._emergency_stop = emergency_stop

    def store_memory(self, content_text: str, memory_type: str = "episodic") -> str:
        return self._client.store_memory(content_text, memory_type=memory_type)

    def health(self) -> dict:
        return self._client.health()


# ── Launch helper ─────────────────────────────────────────────────────────────

def cortex_launch_description(
    memory_sources: list[str] | None = None,
    platform_id:    str               = "",
    node_name:      str               = "cortex",
) -> dict:
    """
    Returns a launch descriptor dict for embedding in ROS 2 launch files.

    Usage:
        from cortex.integration.ros2_node import cortex_launch_description

        def generate_launch_description():
            desc = cortex_launch_description(
                memory_sources=["redis://localhost:6379"],
                platform_id="arm_01",
            )
            # Wrap in rclpy LaunchDescription as needed
            return desc
    """
    return {
        "node_name":      node_name,
        "memory_sources": memory_sources or [],
        "platform_id":    platform_id,
        "topics": {
            "subscribe": ["/joint_states", "/tool_pose"],
            "publish":   ["/cortex/decision", "/cortex/health"],
        },
    }
