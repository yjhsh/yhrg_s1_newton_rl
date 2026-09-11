"""
Simulator Proxy Interface and Implementations for View Layer.

This module provides simulator communication using the Proxy Pattern:
- ISimulatorProxy: Abstract interface for simulator communication
- NewtonSimulatorProxy: Newton-specific implementation
- NewtonArticulation / NewtonRigidBody: batched facades used by the
  controller and by the RL environment
- SimulatorProxyFactory: Factory for creating proxy instances

The Proxy pattern enables:
- Decoupled communication between controller and simulator
- Easy replacement of simulator backends
- Centralized simulator-specific logic
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import torch

import newton
import warp as wp

from model.registry import Registry
from model.loaders import NewtonRobotLoader, quat_xyzw

logger = logging.getLogger(__name__)


class ISimulatorProxy(ABC):
    """
    Abstract interface for simulator communication.

    This interface defines the contract for communicating with different
    physics simulators, enabling decoupled and interchangeable backends.

    The Proxy pattern allows the Controller layer to interact with the
    simulator without knowing the specific implementation details.
    """

    @abstractmethod
    def initialize(self, backend: str = "gpu") -> bool:
        """
        Initialize the simulator.

        Args:
            backend: Computation backend ("gpu" or "cpu").

        Returns:
            True if initialization successful.
        """
        pass

    @abstractmethod
    def create_scene(self, **options) -> Any:
        """
        Create a simulation scene.

        Args:
            **options: Scene configuration options.

        Returns:
            Scene entity.
        """
        pass

    @abstractmethod
    def load_urdf(self, scene: Any, urdf_path: str, **options) -> Any:
        """
        Load a URDF model into the scene.

        Args:
            scene: The scene entity.
            urdf_path: Path to the URDF file.
            **options: Loading options (position, orientation, fixed, etc.).

        Returns:
            Loaded entity.
        """
        pass

    @abstractmethod
    def add_plane(self, scene: Any) -> Any:
        """
        Add a ground plane to the scene.

        Args:
            scene: The scene entity.

        Returns:
            Plane entity.
        """
        pass

    @abstractmethod
    def build_scene(self, scene: Any) -> None:
        """
        Build the scene for simulation.

        Args:
            scene: The scene entity.
        """
        pass

    @abstractmethod
    def step(self, scene: Any) -> None:
        """
        Step the simulation forward.

        Args:
            scene: The scene entity.
        """
        pass

    @abstractmethod
    def get_joint_positions(self, robot: Any, joint_indices: List[int]) -> np.ndarray:
        """
        Get current joint positions.

        Args:
            robot: The robot entity.
            joint_indices: List of joint DOF indices.

        Returns:
            Array of joint positions.
        """
        pass

    @abstractmethod
    def get_joint_velocities(self, robot: Any, joint_indices: List[int]) -> np.ndarray:
        """
        Get current joint velocities.

        Args:
            robot: The robot entity.
            joint_indices: List of joint DOF indices.

        Returns:
            Array of joint velocities.
        """
        pass

    @abstractmethod
    def get_joint_efforts(self, robot: Any, joint_indices: List[int]) -> np.ndarray:
        """
        Get current joint efforts.

        Args:
            robot: The robot entity.
            joint_indices: List of joint DOF indices.

        Returns:
            Array of joint efforts.
        """
        pass

    @abstractmethod
    def set_joint_positions(self, robot: Any, positions: np.ndarray, joint_indices: List[int]) -> None:
        """
        Set joint positions directly.

        Args:
            robot: The robot entity.
            positions: Target positions.
            joint_indices: List of joint DOF indices.
        """
        pass

    @abstractmethod
    def control_joint_positions(self, robot: Any, positions: np.ndarray, joint_indices: List[int]) -> None:
        """
        Control joints to target positions.

        Args:
            robot: The robot entity.
            positions: Target positions.
            joint_indices: List of joint DOF indices.
        """
        pass

    @abstractmethod
    def control_joint_velocities(self, robot: Any, velocities: np.ndarray, joint_indices: List[int]) -> None:
        """
        Control joints with target velocities.

        Args:
            robot: The robot entity.
            velocities: Target velocities.
            joint_indices: List of joint DOF indices.
        """
        pass

    @abstractmethod
    def control_joint_efforts(self, robot: Any, efforts: np.ndarray, joint_indices: List[int]) -> None:
        """
        Control joints with target efforts.

        Args:
            robot: The robot entity.
            efforts: Target efforts.
            joint_indices: List of joint DOF indices.
        """
        pass

    @abstractmethod
    def set_joint_kp(self, robot: Any, kp: np.ndarray, joint_indices: List[int]) -> None:
        """
        Set position control gains.

        Args:
            robot: The robot entity.
            kp: Position gains.
            joint_indices: List of joint DOF indices.
        """
        pass

    @abstractmethod
    def set_joint_kv(self, robot: Any, kv: np.ndarray, joint_indices: List[int]) -> None:
        """
        Set velocity control gains.

        Args:
            robot: The robot entity.
            kv: Velocity gains.
            joint_indices: List of joint DOF indices.
        """
        pass

    @abstractmethod
    def set_joint_force_range(self, robot: Any, lower: np.ndarray, upper: np.ndarray, joint_indices: List[int]) -> None:
        """
        Set joint force limits.

        Args:
            robot: The robot entity.
            lower: Lower force limits.
            upper: Upper force limits.
            joint_indices: List of joint DOF indices.
        """
        pass

    @abstractmethod
    def get_joint(self, robot: Any, joint_name: str) -> Any:
        """
        Get a joint by name.

        Args:
            robot: The robot entity.
            joint_name: Name of the joint.

        Returns:
            Joint entity.
        """
        pass

    @abstractmethod
    def get_control_force(self, robot: Any, joint_indices: List[int]) -> np.ndarray:
        """
        Get current control forces.

        Args:
            robot: The robot entity.
            joint_indices: List of joint DOF indices.

        Returns:
            Array of control forces.
        """
        pass

    @abstractmethod
    def get_simulator_type(self) -> str:
        """
        Return the simulator type identifier.

        Returns:
            String identifier (e.g., "newton", "pybullet").
        """
        pass

    @abstractmethod
    def is_initialized(self) -> bool:
        """
        Check if the simulator is initialized.

        Returns:
            True if initialized.
        """
        pass

    @abstractmethod
    def get_simulator_module(self) -> Any:
        """
        Get the underlying simulator module.

        Returns:
            The simulator module for advanced usage.
        """
        pass


@dataclass
class NewtonJoint:
    """
    Handle for a single joint of an imported Newton articulation.

    Mirrors the attribute the MVC layer expects from a simulator joint entity.

    Attributes:
        name: Joint name as defined in the URDF.
        joint_start: First joint index of the owning import range.
        dofs_idx_local: DOF indices, local to the owning articulation.
    """
    name: str
    joint_start: int
    dofs_idx_local: List[int] = field(default_factory=list)


@dataclass
class NewtonRobot:
    """
    Handle for an articulation imported through :meth:`NewtonSimulatorProxy.load_urdf`.

    Attributes:
        name: Human readable entity name.
        scene: Owning :class:`NewtonScene`.
        joint_start: First joint index created by the import.
        joint_end: One past the last joint index created by the import.
        qd_start: Global DOF offset of the articulation (valid after build).
    """
    name: str
    scene: Any
    joint_start: int
    joint_end: int
    qd_start: int = 0


@dataclass
class NewtonScene:
    """
    Newton simulation scene.

    The scene lives in two phases:
    1. *Builder phase* - a ``newton.ModelBuilder`` collects bodies, shapes and
       joints. Gains, effort limits and joint indices are resolved here.
    2. *Built phase* - ``build_scene()`` finalizes the builder into a
       ``newton.Model`` and creates the states, control struct, collision
       pipeline, solver and (optionally) the viewer.
    """
    builder: Any
    dt: float = 0.01
    show_viewer: bool = True
    camera: Dict[str, Any] = field(default_factory=dict)
    shape_cfg: Any = None

    model: Any = None
    state_0: Any = None
    state_1: Any = None
    control: Any = None
    solver: Any = None
    pipeline: Any = None
    contacts: Any = None
    viewer: Any = None
    robots: List[NewtonRobot] = field(default_factory=list)
    time: float = 0.0

    @property
    def built(self) -> bool:
        """Whether :meth:`build_scene` has already finalized the model."""
        return self.model is not None

    @property
    def state(self):
        """The state holding the most recent simulation results."""
        return self.state_0


class NewtonSimulatorProxy(ISimulatorProxy):
    """
    Newton-specific simulator proxy implementation.

    This class implements the ISimulatorProxy interface for the Newton
    physics simulator, providing a clean abstraction layer for all
    Newton-specific operations.

    The Proxy pattern enables:
    - Decoupling from Newton/Warp-specific APIs
    - Centralized Newton-specific logic
    - Easy testing with mock implementations

    Notes on the joint-space semantics:
    - All DOF indices handed to / returned by this proxy are *local* to the
      articulation, matching the contract of the previous backend.
    - ``get_joint_efforts()`` reads ``State.joint_tau``, which the reduced
      coordinate solver fills with the applied joint forces.
    - ``get_control_force()`` is not separable in Newton; it returns the same
      applied joint forces and logs a warning on first use.
    """

    def __init__(self):
        """Initialize the Newton simulator proxy."""
        self._newton = None
        self._wp = None
        self._initialized = False
        self._backend = "gpu"
        self._warned_control_force = False
        logger.info("NewtonSimulatorProxy created")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self, backend: str = "gpu") -> bool:
        """
        Initialize Warp / Newton.

        Args:
            backend: Computation backend ("gpu" or "cpu").

        Returns:
            True if initialization successful.
        """
        if self._initialized:
            logger.warning("Newton already initialized")
            return True

        try:
            self._backend = backend.lower()
            device = "cuda:0" if self._backend == "gpu" else "cpu"

            logger.info(f"Initializing Warp with {self._backend.upper()} backend...")
            wp.init()
            wp.set_device(device)

            self._newton = newton
            self._wp = wp
            self._initialized = True
            logger.info(f"Newton simulator initialized on {wp.get_device()}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Newton: {e}")
            return False

    def create_scene(self, **options) -> NewtonScene:
        """
        Create a Newton scene in *builder* phase.

        Args:
            **options: Scene options including:
                - dt: Simulation time step
                - show_viewer: Whether to attach a viewer after building
                - camera_pos / camera_lookat / camera_fov: Viewer camera hints

        Returns:
            NewtonScene entity.
        """
        if not self._initialized:
            raise RuntimeError("Simulator not initialized")

        shape_cfg = newton.ModelBuilder.ShapeConfig()
        # Contact stiffness/friction stay at the Newton defaults; overriding ke
        # with 1e5 made the light cube (0.128 kg) bounce out of the table.
        shape_cfg.mu = 0.8

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        # Numerical settings determined by tests/sweep_stability.py:
        # - ``armature`` keeps the ~1e-4 kg*m^2 wrist/finger inertias well
        #   conditioned for the reduced-coordinate solver.
        # - ``limit_ke`` is the dominant stability term. At the control
        #   timestep of 0.01 s / 2 substeps (dt = 0.005 s) the arm diverges
        #   within 12 steps for limit_ke = 1e4 but stays bounded for 1e3.
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            limit_ke=1.0e3,
            limit_kd=1.0e1,
            damping=0.1,
            armature=0.01,
        )

        scene = NewtonScene(
            builder=builder,
            dt=options.get("dt", 0.01),
            show_viewer=options.get("show_viewer", True),
            camera={
                "pos": options.get("camera_pos", (0.0, -3.5, 2.5)),
                "lookat": options.get("camera_lookat", (0.0, 0.0, 0.5)),
                "fov": options.get("camera_fov", 30.0),
            },
            shape_cfg=shape_cfg,
        )

        logger.info("Newton scene created")
        return scene

    def load_urdf(self, scene: NewtonScene, urdf_path: str, **options) -> NewtonRobot:
        """
        Load a URDF model into the scene.

        Args:
            scene: The Newton scene (must still be in builder phase).
            urdf_path: Path to URDF file.
            **options: Loading options (pos, quat, fixed, name).

        Returns:
            NewtonRobot handle.
        """
        if scene.built:
            raise RuntimeError("Cannot add entities after build_scene()")

        name = options.get("name") or urdf_path.rsplit("/", 1)[-1]

        joint_start, joint_end = NewtonRobotLoader.build_urdf(
            scene.builder,
            urdf_path,
            base_position=options.get("pos", (0.0, 0.0, 0.0)),
            base_orientation=options.get("quat", (1.0, 0.0, 0.0, 0.0)),
            fixed_base=options.get("fixed", False),
        )

        robot = NewtonRobot(
            name=name,
            scene=scene,
            joint_start=joint_start,
            joint_end=joint_end,
        )
        scene.robots.append(robot)

        logger.debug(f"URDF loaded: {urdf_path} (joints {joint_start}:{joint_end})")
        return robot

    def add_plane(self, scene: NewtonScene) -> int:
        """
        Add a ground plane to the scene.

        Args:
            scene: The Newton scene (must still be in builder phase).

        Returns:
            Index of the created plane shape.
        """
        scene.builder.add_ground_plane(cfg=scene.shape_cfg)
        shape_index = scene.builder.shape_count - 1
        logger.debug("Ground plane added")
        return shape_index

    def build_scene(self, scene: NewtonScene) -> None:
        """
        Finalize the builder and create states, solver, pipeline and viewer.

        Args:
            scene: The Newton scene.
        """
        if scene.built:
            logger.warning("Scene already built")
            return

        scene.model = scene.builder.finalize()
        scene.state_0 = scene.model.state()
        scene.state_1 = scene.model.state()
        scene.control = scene.model.control()
        scene.pipeline = newton.CollisionPipeline(scene.model)
        scene.contacts = scene.pipeline.contacts()
        scene.solver = newton.solvers.SolverFeatherstone(scene.model)

        # Refresh the global DOF offsets now that the model exists.
        qd_start = scene.model.joint_qd_start.numpy()
        for robot in scene.robots:
            robot.qd_start = int(qd_start[robot.joint_start])

        newton.eval_fk(scene.model, scene.model.joint_q, scene.model.joint_qd, scene.state_0)
        newton.eval_fk(scene.model, scene.model.joint_q, scene.model.joint_qd, scene.state_1)

        if scene.show_viewer:
            scene.viewer = newton.viewer.ViewerGL()
        else:
            scene.viewer = newton.viewer.ViewerNull()

        if hasattr(scene.viewer, "set_model"):
            scene.viewer.set_model(scene.model)
        if hasattr(scene.viewer, "set_camera"):
            scene.viewer.set_camera(
                pos=wp.vec3(*scene.camera["pos"]),
                pitch=-20.0,
                yaw=135.0,
            )

        logger.info("Newton scene built")

    def step(self, scene: NewtonScene) -> None:
        """
        Step the simulation forward by one time step.

        Args:
            scene: The Newton scene.
        """
        if not scene.built:
            raise RuntimeError("Scene must be built before stepping")

        state_in, state_out = scene.state_0, scene.state_1

        state_in.clear_forces()
        scene.pipeline.collide(state_in, scene.contacts)
        scene.solver.step(state_in, state_out, scene.control, scene.contacts, scene.dt)

        # state_out now holds the newest results; recycle the old input.
        scene.state_0, scene.state_1 = state_out, state_in
        scene.time += scene.dt

        if scene.viewer is not None:
            scene.viewer.begin_frame(scene.time)
            scene.viewer.log_state(scene.state_0)
            scene.viewer.end_frame()

    # ------------------------------------------------------------------
    # Joint access
    # ------------------------------------------------------------------

    def get_joint(self, robot: NewtonRobot, joint_name: str) -> NewtonJoint:
        """
        Get a joint handle by name.

        The Newton builder already knows ``joint_label`` / ``joint_qd_start``
        before finalization, so the DOF index can be resolved immediately.

        Args:
            robot: The NewtonRobot handle.
            joint_name: Name of the joint.

        Returns:
            NewtonJoint handle exposing ``dofs_idx_local``.
        """
        indices = NewtonRobotLoader.resolve_dof_indices(
            robot.scene.builder,
            (joint_name,),
            robot.joint_start,
            robot.joint_end,
        )
        if not indices:
            raise ValueError(f"Joint '{joint_name}' not found on entity '{robot.name}'")

        return NewtonJoint(
            name=joint_name,
            joint_start=robot.joint_start,
            dofs_idx_local=indices,
        )

    def get_joint_positions(self, robot: NewtonRobot, joint_indices: List[int]) -> np.ndarray:
        """
        Get current joint positions.

        Args:
            robot: The NewtonRobot handle.
            joint_indices: List of joint DOF indices (local to the robot).

        Returns:
            Array of joint positions.
        """
        return self._read(robot, "joint_q", joint_indices)

    def get_joint_velocities(self, robot: NewtonRobot, joint_indices: List[int]) -> np.ndarray:
        """
        Get current joint velocities.

        Args:
            robot: The NewtonRobot handle.
            joint_indices: List of joint DOF indices (local to the robot).

        Returns:
            Array of joint velocities.
        """
        return self._read(robot, "joint_qd", joint_indices)

    def get_joint_efforts(self, robot: NewtonRobot, joint_indices: List[int]) -> np.ndarray:
        """
        Get current joint efforts (applied joint forces).

        Args:
            robot: The NewtonRobot handle.
            joint_indices: List of joint DOF indices (local to the robot).

        Returns:
            Array of joint efforts.
        """
        return self._read(robot, "joint_tau", joint_indices)

    def get_control_force(self, robot: NewtonRobot, joint_indices: List[int]) -> np.ndarray:
        """
        Get the forces produced by the position controller.

        Newton does not split the applied joint force into "control" and
        "external" contributions, so this returns the applied joint forces.

        Args:
            robot: The NewtonRobot handle.
            joint_indices: List of joint DOF indices (local to the robot).

        Returns:
            Array of applied joint forces.
        """
        if not self._warned_control_force:
            logger.warning(
                "get_control_force() is not separable in Newton; "
                "returning the applied joint forces (State.joint_tau)."
            )
            self._warned_control_force = True
        return self._read(robot, "joint_tau", joint_indices)

    def set_joint_positions(self, robot: NewtonRobot, positions: np.ndarray, joint_indices: List[int]) -> None:
        """
        Set joint positions directly and refresh forward kinematics.

        Args:
            robot: The NewtonRobot handle.
            positions: Target positions.
            joint_indices: List of joint DOF indices (local to the robot).
        """
        scene = robot.scene
        if not scene.built:
            raise RuntimeError("Scene must be built before setting joint positions")

        state = scene.state_0
        self._write(state.joint_q, self._global_dofs(robot, joint_indices), positions)
        self._write(state.joint_qd, self._global_dofs(robot, joint_indices), 0.0)
        newton.eval_fk(scene.model, state.joint_q, state.joint_qd, state)

    def control_joint_positions(self, robot: NewtonRobot, positions: np.ndarray, joint_indices: List[int]) -> None:
        """
        Command joints to target positions.

        Args:
            robot: The NewtonRobot handle.
            positions: Target positions.
            joint_indices: List of joint DOF indices (local to the robot).
        """
        self._write_control(
            robot,
            "joint_target_q",
            joint_indices,
            positions,
            "joint_target_q (position targets) is not available on this model",
        )

    def control_joint_velocities(self, robot: NewtonRobot, velocities: np.ndarray, joint_indices: List[int]) -> None:
        """
        Command joints with target velocities.

        Args:
            robot: The NewtonRobot handle.
            velocities: Target velocities.
            joint_indices: List of joint DOF indices (local to the robot).
        """
        self._write_control(
            robot,
            "joint_target_qd",
            joint_indices,
            velocities,
            "joint_target_qd (velocity targets) is not available on this model",
        )

    def control_joint_efforts(self, robot: NewtonRobot, efforts: np.ndarray, joint_indices: List[int]) -> None:
        """
        Command joints with target efforts.

        Args:
            robot: The NewtonRobot handle.
            efforts: Target efforts.
            joint_indices: List of joint DOF indices (local to the robot).
        """
        self._write_control(
            robot,
            "joint_act",
            joint_indices,
            efforts,
            "joint_act (effort targets) is not available on this model",
        )

    def set_joint_kp(self, robot: NewtonRobot, kp: np.ndarray, joint_indices: List[int]) -> None:
        """
        Set position control gains.

        Before ``build_scene()`` the gains are baked into the model; afterwards
        they are updated through the control struct.

        Args:
            robot: The NewtonRobot handle.
            kp: Position gains.
            joint_indices: List of joint DOF indices (local to the robot).
        """
        self._set_gain(robot, "joint_target_ke", joint_indices, kp)

    def set_joint_kv(self, robot: NewtonRobot, kv: np.ndarray, joint_indices: List[int]) -> None:
        """
        Set velocity control gains.

        Args:
            robot: The NewtonRobot handle.
            kv: Velocity gains.
            joint_indices: List of joint DOF indices (local to the robot).
        """
        self._set_gain(robot, "joint_target_kd", joint_indices, kv)

    def set_joint_force_range(self, robot: NewtonRobot, lower: np.ndarray, upper: np.ndarray, joint_indices: List[int]) -> None:
        """
        Set joint force limits.

        Newton stores a single symmetric effort limit per DOF, so the tightest
        of the two Genesis-style bounds is used.

        Args:
            robot: The NewtonRobot handle.
            lower: Lower force limits.
            upper: Upper force limits.
            joint_indices: List of joint DOF indices (local to the robot).
        """
        scene = robot.scene
        if scene.built:
            logger.warning("Effort limits are baked into the model; ignoring post-build update")
            return

        lower_arr = np.atleast_1d(np.asarray(lower, dtype=float))
        upper_arr = np.atleast_1d(np.asarray(upper, dtype=float))
        limits = np.minimum(np.abs(lower_arr), np.abs(upper_arr))

        for dof, value in zip(self._global_dofs(robot, joint_indices), limits):
            scene.builder.joint_effort_limit[dof] = float(value)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_simulator_type(self) -> str:
        return "newton"

    def is_initialized(self) -> bool:
        return self._initialized

    def get_simulator_module(self) -> Any:
        return self._newton

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _global_dofs(robot: NewtonRobot, joint_indices: List[int]) -> List[int]:
        """Convert articulation-local DOF indices into global DOF indices."""
        return [robot.qd_start + int(i) for i in joint_indices]

    def _read(self, robot: NewtonRobot, field_name: str, joint_indices: List[int]) -> np.ndarray:
        """Read a joint-space array from the current state."""
        scene = robot.scene
        if not scene.built:
            raise RuntimeError("Scene must be built before reading joint state")

        array = getattr(scene.state_0, field_name, None)
        if array is None:
            # Featherstone allocates its auxiliary arrays (joint_tau, ...)
            # on the solver whenever the state does not require gradients.
            array = getattr(scene.solver, field_name, None)
        if array is None:
            raise RuntimeError(f"Neither state nor solver exposes '{field_name}'")

        dofs = self._global_dofs(robot, joint_indices)
        return np.asarray(array.numpy())[dofs].copy()

    @staticmethod
    def _write(array, dofs: List[int], values) -> None:
        """Write values into a device array at the given indices."""
        host = np.asarray(array.numpy()).copy()
        host[dofs] = np.asarray(values, dtype=host.dtype).reshape(-1)[: len(dofs)]
        array.assign(host)

    def _write_control(self, robot: NewtonRobot, field_name: str, joint_indices: List[int], values, missing_msg: str) -> None:
        """Write joint targets into the control struct."""
        scene = robot.scene
        if not scene.built:
            raise RuntimeError("Scene must be built before sending joint commands")

        array = getattr(scene.control, field_name, None)
        if array is None:
            logger.warning(missing_msg)
            return

        self._write(array, self._global_dofs(robot, joint_indices), values)

    def _set_gain(self, robot: NewtonRobot, field_name: str, joint_indices: List[int], values) -> None:
        """Set a PD gain, either on the builder (pre-build) or the control struct."""
        scene = robot.scene
        values_arr = np.atleast_1d(np.asarray(values, dtype=float))
        dofs = self._global_dofs(robot, joint_indices)

        if not scene.built:
            builder_array = getattr(scene.builder, field_name, None)
            if builder_array is None:
                raise RuntimeError(f"Builder does not expose '{field_name}'")
            for dof, value in zip(dofs, values_arr.reshape(-1)):
                builder_array[dof] = float(value)
            return

        array = getattr(scene.control, field_name, None)
        if array is None:
            logger.warning(f"{field_name} is not available on the control struct")
            return
        self._write(array, dofs, values_arr)


# Module-level registry instance for simulator proxies
_proxy_registry = Registry[ISimulatorProxy]("simulator_proxies")
_proxy_registry.register("newton", NewtonSimulatorProxy)


class SimulatorProxyFactory:
    """
    Factory for creating simulator proxy instances, backed by model.registry.Registry.
    """

    @classmethod
    def register(cls, name: str, proxy_class: Type[ISimulatorProxy]) -> None:
        _proxy_registry.register(name, proxy_class)

    @classmethod
    def create(cls, name: str = "newton") -> ISimulatorProxy:
        return _proxy_registry.create(name)

    @classmethod
    def get_available_simulators(cls) -> List[str]:
        return _proxy_registry.get_available()


def quat_rotate_torch(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """
    Rotate vectors by quaternions.

    Args:
        quat: Quaternions in (x, y, z, w) order, shape (..., 4).
        vec: Vectors, shape (..., 3).

    Returns:
        Rotated vectors, shape (..., 3).
    """
    qx, qy, qz, qw = quat.unbind(-1)
    vx, vy, vz = vec.unbind(-1)

    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)

    return torch.stack(
        (
            vx + qw * tx + (qy * tz - qz * ty),
            vy + qw * ty + (qz * tx - qx * tz),
            vz + qw * tz + (qx * ty - qy * tx),
        ),
        dim=-1,
    )


class NewtonArticulation:
    """
    Batched facade over one Newton articulation replicated across all worlds.

    The interactive path uses :class:`NewtonSimulatorProxy` (single robot),
    while the batched RL path uses this class. It exposes the same small
    surface the controller expects, so ``controller.task_space_controller``
    stays simulator agnostic.

    Jacobian convention
    -------------------
    ``ArticulationView.eval_jacobian()`` returns, per link, six rows ordered as
    ``[linear(3), angular(3)]``. The linear block is the velocity of the link
    **COM**, whereas ``body_q`` stores the link **origin** frame. This facade
    converts the linear block to the link-origin frame with

    ``J_origin[0:3] = J_com[0:3] + (R @ body_com) x J_ang``

    which matches the link-frame Jacobian the controller was written against.
    """

    def __init__(self, model, pattern: str):
        """
        Args:
            model: A finalized ``newton.Model``.
            pattern: Articulation name pattern, e.g. ``"fangzhenjixiebi*"``.
        """
        self.model = model
        self.view = newton.selection.ArticulationView(model, pattern, verbose=False)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.count = int(self.view.count)
        self.dof_count = int(self.view.joint_dof_count)
        self.link_count = int(self.view.link_count)
        self.link_names = list(self.view.link_names)
        self.joint_names = list(self.view.joint_dof_names)

        # Bodies that back each link (world 0 layout; replicated per world).
        shape_body = np.asarray(model.shape_body.numpy())
        self.link_body_local = [int(shape_body[shapes[0]]) for shapes in self.view.link_shapes]
        self.bodies_per_world = int(model.body_count) // max(1, self.count)

        body_com = np.asarray(model.body_com.numpy())
        self.body_com_local = torch.as_tensor(
            body_com[self.link_body_local], dtype=torch.float32, device=self.device
        )

        self._jac_buffer = None

        # ------------------------------------------------------------------
        # Global DOF mapping
        # ------------------------------------------------------------------
        # ``replicate()`` lays the worlds out back to back, so the global DOF
        # index of articulation-local DOF ``i`` in world ``w`` is
        # ``w * dofs_per_world + dof_start_in_world + i``.
        self.dofs_per_world = int(model.joint_dof_count) // self.count
        joints_per_world = int(model.joint_count) // self.count
        if not self.view.joint_dofs_contiguous:
            raise RuntimeError("Articulation DOFs are not contiguous")

        first_label = self.view.joint_labels[0]
        labels0 = list(model.joint_label)[:joints_per_world]
        short0 = [label.rsplit("/", 1)[-1] for label in labels0]
        short_first = first_label.rsplit("/", 1)[-1]
        if first_label in labels0:
            first_joint_in_world = labels0.index(first_label)
        elif short_first in short0:
            first_joint_in_world = short0.index(short_first)
        else:
            raise RuntimeError(f"Articulation joint '{first_label}' not found in the model")

        qd_start = np.asarray(model.joint_qd_start.numpy())
        self.dof_start_in_world = int(qd_start[first_joint_in_world])

        worlds = torch.arange(self.count, device=self.device).unsqueeze(1)
        local = torch.arange(self.dof_count, device=self.device).unsqueeze(0)
        self._dof_global = (worlds * self.dofs_per_world + self.dof_start_in_world + local)
        self._dof_global_flat = self._dof_global.reshape(-1)

        # ------------------------------------------------------------------
        # Coordinate mapping (joint_target_q is coordinate-layout!)
        # ------------------------------------------------------------------
        # With ``use_coord_layout_targets = True`` the target arrays are indexed
        # by joint *coordinates*, whose per-world stride differs from the DOF
        # stride as soon as the scene contains free bodies (7 coords / 6 dofs).
        self.coords_per_world = int(model.joint_coord_count) // self.count
        q_start = np.asarray(model.joint_q_start.numpy())
        self.coord_start_in_world = int(q_start[first_joint_in_world])
        local_c = torch.arange(self.dof_count, device=self.device).unsqueeze(0)
        self._coord_global = (
            worlds * self.coords_per_world + self.coord_start_in_world + local_c
        )
        self._coord_global_flat = self._coord_global.reshape(-1)

        # ------------------------------------------------------------------
        # Jacobian buffer
        # ------------------------------------------------------------------
        # eval_jacobian() returns one block per *articulation* (not per world):
        # the model holds one articulation per replicated entity, so the arm of
        # world w lives at articulation index artic_indices[w].
        mask = np.asarray(self.view.get_model_articulation_mask().numpy()).nonzero()[0]
        self._artic_indices = torch.as_tensor(mask, dtype=torch.long, device=self.device)
        self._jac_shape = (
            int(model.articulation_count),
            int(model.max_joints_per_articulation) * 6,
            int(model.max_dofs_per_articulation),
        )
        self._jac_buffer = wp.zeros(self._jac_shape, dtype=wp.float32, device=model.device)

        # PD gains live on the *model* in Newton (the control struct only
        # carries the targets), so both arrays are needed.
        self._model = model

    def _world_rows(self, envs_idx) -> torch.Tensor:
        """World indices for an optional bool/index mask (default: all)."""
        if envs_idx is None:
            return torch.arange(self.count, device=self.device)
        if envs_idx.dtype == torch.bool:
            return envs_idx.nonzero(as_tuple=True)[0]
        return envs_idx.to(torch.long)

        # Bound by the owning simulation loop; the state object is swapped
        # after every step, so the loop rebinds it before reading or writing.
        self.state = None
        self.control = None

    def bind(self, state, control=None) -> None:
        """
        Bind the current simulation state (and optionally the control struct).

        Args:
            state: The state holding the most recent simulation results.
            control: The control struct; only needed once.
        """
        self.state = state
        if control is not None:
            self.control = control

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve(self, state):
        state = self.state if state is None else state
        if state is None:
            raise RuntimeError("No simulation state bound to this articulation")
        return state

    def link_index(self, link) -> int:
        """
        Resolve a link name (or pass through an index).

        Newton keeps fully qualified labels such as "fangzhenjixiebi/6_Link",
        so both the full label and its last path component are accepted.
        """
        if isinstance(link, int):
            return link
        if link in self.link_names:
            return self.link_names.index(link)
        short = [name.rsplit("/", 1)[-1] for name in self.link_names]
        if link in short:
            return short.index(link)
        raise ValueError(f"link '{link}' not found in {self.link_names}")

    def _mask(self, envs_idx):
        """Build an articulation mask for the given environment indices."""
        if envs_idx is None:
            return None
        mask = torch.zeros(self.count, dtype=torch.bool, device=self.device)
        mask[envs_idx] = True
        return wp.from_torch(mask.contiguous())

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    def get_qpos(self, state=None) -> torch.Tensor:
        """
        Joint positions, shape (worlds, dofs).

        Read straight from ``state.joint_q`` through the global *coordinate*
        mapping. ``state.joint_q`` uses the coordinate layout (7 coords for a
        free body vs 6 dofs), whose per-world stride differs from the DOF
        stride used by ``joint_qd`` / ``joint_target_q``; mixing them shifts
        every world after the first. ``ArticulationView.get_dof_positions()``
        only fills the first articulation of a replicated model, so it must
        not be used here.
        """
        state = self._resolve(state)
        return wp.to_torch(state.joint_q)[self._coord_global]

    def get_dofs_velocity(self, state=None) -> torch.Tensor:
        """Joint velocities, shape (worlds, dofs)."""
        state = self._resolve(state)
        return wp.to_torch(state.joint_qd)[self._dof_global]

    def get_link_transforms(self, state=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        World transforms of every link.

        Returns:
            ``(positions, quaternions)`` with shapes ``(worlds, links, 3)`` and
            ``(worlds, links, 4)``; quaternions use the Warp (x, y, z, w) order.
        """
        state = self._resolve(state)
        transforms = wp.to_torch(self.view.get_link_transforms(state)).reshape(
            self.count, self.link_count, 7
        )
        return transforms[:, :, 0:3], transforms[:, :, 3:7]

    def get_jacobian(self, link, state=None) -> torch.Tensor:
        """
        Link-frame Jacobian for one link.

        Args:
            link: Link name or index inside the articulation.
            state: Optional state override; defaults to the bound state.

        Returns:
            Tensor of shape ``(worlds, 6, dofs)`` ordered as
            ``[linear(3), angular(3)]`` and referenced to the link origin.
        """
        state = self._resolve(state)
        idx = self.link_index(link)

        jac = self.view.eval_jacobian(state, self._jac_buffer)
        j_t = wp.to_torch(jac).reshape(self._jac_shape)

        # Select the arm articulation of every world, then its EE rows.
        j_ee = j_t[self._artic_indices][:, idx * 6 : (idx + 1) * 6, : self.dof_count]
        j_ang = j_ee[:, 3:6, :]

        # Convert the linear block from the link COM to the link origin.
        _, quats = self.get_link_transforms(state)
        r = quat_rotate_torch(
            quats[:, idx, :],
            self.body_com_local[idx].unsqueeze(0).expand(self.count, 3),
        )
        j_lin = j_ee[:, 0:3, :] + torch.cross(
            r.unsqueeze(-1).expand(-1, 3, self.dof_count), j_ang, dim=1
        )

        return torch.cat([j_lin, j_ang], dim=1)

    # ------------------------------------------------------------------
    # State writing
    # ------------------------------------------------------------------

    def set_qpos(self, values: torch.Tensor, envs_idx=None, state=None) -> None:
        """
        Set joint positions (and zero velocities) for the given worlds.

        Args:
            values: Joint positions, shape ``(num_selected, dofs)`` when a mask
                is given, otherwise ``(count, dofs)``.
            envs_idx: Optional bool mask or index tensor.
            state: Optional state override.
        """
        state = self._resolve(state)
        rows = self._world_rows(envs_idx)
        n_sel = int(rows.numel())

        vals = values.to(torch.float32)
        if vals.ndim == 1:
            # One row broadcast to every selected world.
            vals = vals.unsqueeze(0).expand(n_sel, -1)
        elif vals.shape[0] != n_sel and vals.shape[0] == self.count:
            # Full-batch input with a mask: keep the selected rows only.
            vals = vals[rows]

        # ``state.joint_q`` is coordinate-layout while ``state.joint_qd`` is
        # DOF-layout (a free body has 7 coords but 6 dofs), so each array needs
        # its own mapping. Using the DOF mapping for ``joint_q`` shifts every
        # world after the first by (coords_per_world - dofs_per_world) slots.
        q_idx = self._coord_global[rows].reshape(-1)
        qd_idx = self._dof_global[rows].reshape(-1)
        flat = vals.reshape(-1)

        q = wp.to_torch(state.joint_q)
        q[q_idx] = flat

        qd = wp.to_torch(state.joint_qd)
        qd[qd_idx] = 0.0

        self.view.eval_fk(state)

    def _set_dof_buffer(self, array, values, dofs_idx) -> None:
        """
        Write per-DOF values into a model/control array at the DOF mapping.

        ``values`` may be 1-D (one value per DOF, applied to every world) or
        2-D (one row per world), mirroring the previous backend's API.
        """
        host = wp.to_torch(array)
        vals = values.to(host.dtype)
        if vals.ndim == 1:
            for k, local_dof in enumerate(dofs_idx):
                host[self._dof_global[:, int(local_dof)]] = vals[k]
        else:
            for k, local_dof in enumerate(dofs_idx):
                host[self._dof_global[:, int(local_dof)]] = vals[:, k]

    def set_dofs_kp(self, values: torch.Tensor, dofs_idx) -> None:
        """Set position gains for the given DOFs (stored on the model)."""
        array = getattr(self._model, "joint_target_ke", None)
        if array is None:
            logger.warning("joint_target_ke is not available on the model")
            return
        self._set_dof_buffer(array, values, dofs_idx)

    def set_dofs_kv(self, values: torch.Tensor, dofs_idx) -> None:
        """Set velocity gains for the given DOFs (stored on the model)."""
        array = getattr(self._model, "joint_target_kd", None)
        if array is None:
            logger.warning("joint_target_kd is not available on the model")
            return
        self._set_dof_buffer(array, values, dofs_idx)

    def control_dofs_position(self, values: torch.Tensor) -> None:
        """
        Command joint position targets for all DOFs.

        The control target arrays are DOF-indexed (``joint_dof_count`` entries),
        even though ``state.joint_q`` uses the coordinate layout.
        """
        if self.control is None:
            raise RuntimeError("No control struct bound to this articulation")

        host = wp.to_torch(self.control.joint_target_q)
        host[self._dof_global_flat] = values.reshape(-1).to(host.dtype)

    def clamp_dofs_position(
        self,
        lower,
        upper,
        dofs_idx,
        state=None,
    ) -> None:
        """
        Hard mechanical stop: project selected DOFs into ``[lower, upper]``.

        The gripper's prismatic travel is only [0, 0.05] m, but the solver's
        joint-limit constraint does not hold the fingers: they slide steadily
        past the limit and drift far away from the gripper. Clamping the
        position and zeroing the velocity of any DOF that hits a stop models
        the real hard stop of the mechanism.

        Args:
            lower: Per-DOF lower bound (scalar or sequence, one per DOF).
            upper: Per-DOF upper bound.
            dofs_idx: Local DOF indices inside the articulation.
            state: Optional state override.
        """
        state = self._resolve(state)
        q = wp.to_torch(state.joint_q)
        qd = wp.to_torch(state.joint_qd)

        for k, local_dof in enumerate(dofs_idx):
            q_idx = self._coord_global[:, int(local_dof)]
            qd_idx = self._dof_global[:, int(local_dof)]

            lo = lower[k] if hasattr(lower, "__getitem__") else float(lower)
            hi = upper[k] if hasattr(upper, "__getitem__") else float(upper)

            cur = q[q_idx]
            new = cur.clamp(float(lo), float(hi))
            hit_stop = (cur < float(lo)) | (cur > float(hi))
            q[q_idx] = new
            qd[qd_idx] = torch.where(hit_stop, torch.zeros_like(new), qd[qd_idx])


class NewtonRigidBody:
    """
    Batched facade over a free rigid body replicated across all worlds.

    Provides the ``get_pos / get_quat / get_vel / set_pos / set_quat`` surface
    the RL environment uses for the pushed cube.
    """

    def __init__(self, model, pattern: str):
        """
        Args:
            model: A finalized ``newton.Model``.
            pattern: Articulation name pattern, e.g. ``"cube*"``.
        """
        self.model = model
        self.view = newton.selection.ArticulationView(model, pattern, verbose=False)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.count = int(self.view.count)
        self.state = None

        # ------------------------------------------------------------------
        # Free-joint coordinate mapping
        # ------------------------------------------------------------------
        # A free body is backed by a free joint with 7 coordinates
        # (x, y, z, qx, qy, qz, qw). Positions/orientations must be written
        # there (and followed by eval_fk); writing ``body_q`` directly is
        # overwritten by the next forward-kinematics pass.
        self.coords_per_world = int(model.joint_coord_count) // self.count
        if not self.view.joint_coords_contiguous:
            raise RuntimeError("Articulation coordinates are not contiguous")

        first_label = self.view.joint_labels[0]
        joints_per_world = int(model.joint_count) // self.count
        labels0 = list(model.joint_label)[:joints_per_world]
        short0 = [label.rsplit("/", 1)[-1] for label in labels0]
        short_first = first_label.rsplit("/", 1)[-1]
        if first_label in labels0:
            first_joint_in_world = labels0.index(first_label)
        elif short_first in short0:
            first_joint_in_world = short0.index(short_first)
        else:
            raise RuntimeError(f"Articulation joint '{first_label}' not found in the model")

        q_start = np.asarray(model.joint_q_start.numpy())
        self.coord_start_in_world = int(q_start[first_joint_in_world])

        worlds = torch.arange(self.count, device=self.device).unsqueeze(1)
        local = torch.arange(7, device=self.device).unsqueeze(0)
        self._coord_global = (
            worlds * self.coords_per_world + self.coord_start_in_world + local
        )

        # Global body index (used for the body velocity readout).
        shape_body = np.asarray(model.shape_body.numpy())
        body_local = int(shape_body[self.view.link_shapes[0][0]])
        bodies_per_world = int(model.body_count) // max(1, self.count)
        self._body_global = (
            torch.arange(self.count, device=self.device) * bodies_per_world + body_local
        )

    def bind(self, state) -> None:
        """
        Bind the current simulation state.

        Args:
            state: The state holding the most recent simulation results.
        """
        self.state = state

    def _resolve(self, state):
        state = self.state if state is None else state
        if state is None:
            raise RuntimeError("No simulation state bound to this body")
        return state

    def _world_rows(self, envs_idx) -> torch.Tensor:
        if envs_idx is None:
            return torch.arange(self.count, device=self.device)
        if envs_idx.dtype == torch.bool:
            return envs_idx.nonzero(as_tuple=True)[0]
        return envs_idx.to(torch.long)

    def _free_coords(self, state) -> torch.Tensor:
        """Free-joint coordinates, shape (worlds, 7) as (x, y, z, qx, qy, qz, qw)."""
        return wp.to_torch(state.joint_q)[self._coord_global]

    def get_pos(self, state=None) -> torch.Tensor:
        """World positions, shape (worlds, 3)."""
        return self._free_coords(self._resolve(state))[:, 0:3]

    def get_quat(self, state=None) -> torch.Tensor:
        """
        World orientations in (x, y, z, w) order, shape (worlds, 4).

        Note: Genesis uses (w, x, y, z); callers must convert if they need the
        URDF/Genesis ordering.
        """
        return self._free_coords(self._resolve(state))[:, 3:7]

    def get_vel(self, state=None) -> torch.Tensor:
        """Linear velocities, shape (worlds, 3)."""
        state = self._resolve(state)
        rows = self._world_rows(None)
        body_qd = wp.to_torch(state.body_qd)[self._body_global[rows]]
        # Warp spatial vectors are [angular(3), linear(3)].
        return body_qd[:, 3:6]

    def _set_free_coords(self, state, envs_idx, values, start: int, width: int) -> None:
        """Write part of the 7-component free-joint coordinate block."""
        rows = self._world_rows(envs_idx)

        vals = values
        if vals.shape[0] != rows.numel() and vals.shape[0] == self.count:
            # Full-batch input with a mask: keep the selected rows only.
            vals = vals[rows]

        coords = wp.to_torch(state.joint_q)
        idx = self._coord_global[rows]
        coords[idx[:, start : start + width]] = vals.to(coords.dtype).reshape(-1, width)
        self.view.eval_fk(state)

    def set_pos(self, positions: torch.Tensor, envs_idx=None, state=None) -> None:
        """Set world positions for the given worlds."""
        state = self._resolve(state)
        self._set_free_coords(state, envs_idx, positions, 0, 3)

    def set_quat(self, quats: torch.Tensor, envs_idx=None, state=None) -> None:
        """
        Set world orientations for the given worlds.

        Args:
            quats: Orientations in the Warp (x, y, z, w) order.
            envs_idx: Optional world indices.
            state: Optional state override.
        """
        state = self._resolve(state)
        self._set_free_coords(state, envs_idx, quats, 3, 4)
