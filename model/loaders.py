"""
Loader Interfaces and Implementations for Model Layer.

This module provides loader components using design patterns:
- Abstract Factory Pattern: ISceneLoaderFactory, IRobotLoaderFactory, IObjectLoaderFactory
- Simple Factory Pattern: LoaderFactoryRegistry for concrete loader creation

The loaders are responsible for:
- Loading scenes, robots, and objects into the simulation
- Managing asset configurations
- Providing standardized loading interfaces

The concrete ``Newton*Loader`` implementations talk to the simulator through
the ``ISimulatorProxy`` interface (see ``view/proxy.py``), so this module never
imports a simulator directly.

Shared URDF import
------------------
``NewtonRobotLoader.build_urdf()`` is the single place where the YHRG S1 URDF
is imported into a Newton ``ModelBuilder``. Both the interactive MVC path and
the batched RL environment (``rl_push_env.py``) call it, so the import options
(gains, self-collision, fixed-joint collapsing) stay consistent.
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import warp as wp

from config.scene_config import (
    SceneConfig,
    RobotConfig,
    DesktopConfig,
    TargetObjectConfig,
)
from model.state import RobotState, SceneState
from model.registry import Registry

logger = logging.getLogger(__name__)


def quat_xyzw(quat_wxyz: Tuple[float, float, float, float]) -> wp.quat:
    """
    Convert a (w, x, y, z) quaternion (URDF / Genesis convention) into the
    (x, y, z, w) quaternion used by Warp / Newton.

    Args:
        quat_wxyz: Quaternion in (w, x, y, z) order.

    Returns:
        ``wp.quat`` in (x, y, z, w) order.
    """
    w, x, y, z = quat_wxyz
    return wp.quat(float(x), float(y), float(z), float(w))


@dataclass
class LoadResult:
    """
    Result of a loading operation.

    Attributes:
        success: Whether the loading was successful.
        entity: The loaded entity (if successful).
        error_message: Error message (if failed).
        metadata: Additional metadata about the loaded entity.
    """
    success: bool
    entity: Any = None
    error_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, entity: Any, **metadata) -> "LoadResult":
        """Create a successful result."""
        return cls(success=True, entity=entity, metadata=metadata)

    @classmethod
    def error(cls, message: str, **metadata) -> "LoadResult":
        """Create an error result."""
        return cls(success=False, error_message=message, metadata=metadata)


class ISceneLoaderFactory(ABC):
    """
    Abstract factory interface for scene loaders.

    Defines the contract for creating scene loaders that can load
    complete simulation scenes with different simulator backends.
    """

    @abstractmethod
    def create_loader(self, config: SceneConfig) -> "ISceneLoader":
        """
        Create a scene loader instance.

        Args:
            config: Scene configuration.

        Returns:
            ISceneLoader instance.
        """
        pass

    @abstractmethod
    def get_loader_type(self) -> str:
        """
        Return the type identifier for this factory.

        Returns:
            String identifier (e.g., "newton", "pybullet").
        """
        pass


class IRobotLoaderFactory(ABC):
    """
    Abstract factory interface for robot loaders.

    Defines the contract for creating robot loaders that can load
    robot models from URDF or other model formats.
    """

    @abstractmethod
    def create_loader(self, config: RobotConfig) -> "IRobotLoader":
        """
        Create a robot loader instance.

        Args:
            config: Robot configuration.

        Returns:
            IRobotLoader instance.
        """
        pass

    @abstractmethod
    def get_loader_type(self) -> str:
        """
        Return the type identifier for this factory.

        Returns:
            String identifier (e.g., "urdf", "mjcf").
        """
        pass


class IObjectLoaderFactory(ABC):
    """
    Abstract factory interface for object loaders.

    Defines the contract for creating object loaders that can load
    target objects, desktops, and other scene objects.
    """

    @abstractmethod
    def create_loader(self, config: Any) -> "IObjectLoader":
        """
        Create an object loader instance.

        Args:
            config: Object configuration.

        Returns:
            IObjectLoader instance.
        """
        pass

    @abstractmethod
    def get_loader_type(self) -> str:
        """
        Return the type identifier for this factory.

        Returns:
            String identifier (e.g., "urdf", "mesh").
        """
        pass


class ISceneLoader(ABC):
    """
    Abstract interface for scene loading operations.

    Defines the contract for loading complete simulation scenes.
    """

    @abstractmethod
    def load(self, context: Dict[str, Any]) -> LoadResult:
        """
        Load the scene.

        Args:
            context: Context containing the simulator proxy.

        Returns:
            LoadResult with scene entity.
        """
        pass


class IRobotLoader(ABC):
    """
    Abstract interface for robot loading operations.

    Defines the contract for loading robot models.
    """

    @abstractmethod
    def load(self, context: Dict[str, Any]) -> LoadResult:
        """
        Load the robot.

        Args:
            context: Context containing scene reference.

        Returns:
            LoadResult with robot entity.
        """
        pass

    @abstractmethod
    def get_robot_state(self) -> RobotState:
        """
        Get the robot state after loading.

        Returns:
            RobotState instance.
        """
        pass


class IObjectLoader(ABC):
    """
    Abstract interface for object loading operations.

    Defines the contract for loading scene objects.
    """

    @abstractmethod
    def load(self, context: Dict[str, Any]) -> LoadResult:
        """
        Load the object.

        Args:
            context: Context containing scene reference.

        Returns:
            LoadResult with object entity.
        """
        pass


class NewtonSceneLoader(ISceneLoader):
    """
    Scene loader implementation for the Newton simulator.

    Creates the scene through the simulator proxy and, when enabled, adds the
    ground plane. The scene stays in "builder" mode until the view layer calls
    ``proxy.build_scene()``.
    """

    def __init__(self, config: SceneConfig):
        """
        Initialize the Newton scene loader.

        Args:
            config: Scene configuration.
        """
        self._config = config
        self._scene = None
        self._proxy = None
        logger.info("NewtonSceneLoader initialized")

    def load(self, context: Dict[str, Any]) -> LoadResult:
        """
        Load the Newton scene.

        Args:
            context: Context containing 'simulator_proxy' (ISimulatorProxy).

        Returns:
            LoadResult with scene entity.
        """
        self._proxy = context.get("simulator_proxy")
        if self._proxy is None:
            return LoadResult.error("Simulator proxy not available in context")

        try:
            ground_config = self._config.ground

            self._scene = self._proxy.create_scene(
                show_viewer=context.get("show_viewer", True),
                dt=0.01,
            )

            if ground_config.enabled:
                plane = self._proxy.add_plane(self._scene)
                context["plane"] = plane

            context["scene"] = self._scene
            context["scene_config"] = self._config

            logger.info("Newton scene loaded successfully")
            return LoadResult.ok(
                self._scene,
                ground_enabled=ground_config.enabled,
                camera_position=self._config.camera.position,
            )

        except Exception as e:
            logger.error(f"Failed to load Newton scene: {e}")
            return LoadResult.error(str(e))

    def get_scene(self):
        """Get the loaded scene."""
        return self._scene


class NewtonRobotLoader(IRobotLoader):
    """
    Robot loader implementation for the Newton simulator.

    Loads robot models from URDF files and initializes joint states.
    Joint DOF indices are resolved straight from the Newton builder, which
    already knows ``joint_label`` / ``joint_qd_start`` before ``finalize()``.
    """

    def __init__(self, config: RobotConfig, robot_state: Optional[RobotState] = None):
        """
        Initialize the Newton robot loader.

        Args:
            config: Robot configuration.
            robot_state: Optional pre-configured robot state.
        """
        self._config = config
        self._robot_state = robot_state or RobotState(
            name=config.name,
            urdf_path=config.urdf_path,
            fixed_base=config.fixed_base,
        )
        self._robot = None
        self._motor_dof_idx = []
        logger.info(f"NewtonRobotLoader initialized for: {config.urdf_path}")

    @staticmethod
    def build_urdf(
        builder,
        urdf_path: str,
        base_position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        base_orientation: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        fixed_base: bool = True,
        enable_self_collisions: bool = True,
        collapse_fixed_joints: bool = True,
        collide_base_link: bool = False,
    ) -> Tuple[int, int]:
        """
        Import the YHRG S1 URDF into a Newton ModelBuilder.

        This is the single source of truth for the URDF import and is shared by
        the interactive MVC path and the batched RL environment.

        Args:
            builder: ``newton.ModelBuilder`` to import into.
            urdf_path: Path to the URDF file.
            base_position: Base position in world coordinates.
            base_orientation: Base orientation as (w, x, y, z).
            fixed_base: Whether the base is fixed to the world.
            enable_self_collisions: Must stay ``True``. Newton's blanket
                self-collision filter (active when ``False``) filters *every*
                colliding shape pair of the robot and cannot be undone, which
                would override the selective per-link rules below. Actual
                self-collision behaviour is governed by those rules.
            collapse_fixed_joints: Whether to merge fixed joints (the massless
                ``base_link`` is merged into the world body when enabled).

        Returns:
            ``(joint_start, joint_end)``: the half-open range of joint indices
            that were created by this import.
        """
        joint_start = builder.joint_count
        shape_start = builder.shape_count
        # NOTE: ``enable_self_collisions`` must stay True here. When False,
        # Newton blanket-filters *every* colliding shape pair of the robot
        # (import_urdf builds C(n,2) pairs), which silently overrides the
        # selective per-link policy implemented below (filter pairs can only
        # be added, never removed). The rules below own the filtering.
        builder.add_urdf(
            urdf_path,
            xform=wp.transform(wp.vec3(*base_position), quat_xyzw(base_orientation)),
            floating=not fixed_base,
            enable_self_collisions=enable_self_collisions,
            collapse_fixed_joints=collapse_fixed_joints,
        )
        joint_end = builder.joint_count
        shape_end = builder.shape_count

        # Hold the imported configuration until the first command arrives:
        # Newton's default joint target is zero, which would make the arm swing
        # towards the zero pose as soon as the simulation starts.
        qd_start = builder.joint_qd_start[joint_start]
        n_dofs = builder.joint_qd_start[joint_end - 1] + 1 - qd_start
        for i in range(n_dofs):
            builder.joint_target_q[qd_start + i] = builder.joint_q[qd_start + i]

        if not collide_base_link:
            # ``base_link`` is massless and fixed, so ``collapse_fixed_joints``
            # merged it into the world body. Its collision mesh then becomes
            # *static* geometry that overlaps whatever the arm stands on (the
            # table top in the push task), producing static/static contact
            # pairs that the solver does not handle meaningfully.
            # NOTE: the robot occupies ``shape_start..shape_end`` — one visual
            # plus one collision shape per *link* (including base_link), i.e.
            # 2 * n_links shapes, NOT ``2 * n_joints``. The old approximation
            # under-counted by the base_link shapes and left its static
            # collision hull active.
            import newton as _newton

            for i in range(shape_start, shape_end):
                body = builder.shape_body[i]
                if body < 0:
                    builder.shape_flags[i] &= ~_newton.ShapeFlags.COLLIDE_SHAPES

        # ── Collision geometry: convex hulls instead of full SDF meshes ──
        # ``add_urdf`` registers every URDF collision mesh as ``GeoType.MESH``,
        # which is handled by the SDF "triangle pair" path. That path only
        # produces contacts while the SDF lies inside its narrow band, which
        # ``import_urdf`` hard-codes to +-0.1 m. Once the gripper is driven
        # deeper than that into a link, no contacts are generated at all and
        # the solver can never separate the bodies — the residual
        # gripper/arm-body interpenetration seen in some environments.
        # Convex hulls are handled by the GJK/MPR kernel instead, which has no
        # band limit (deep overlap is resolved), is much cheaper, and does not
        # consume any triangle-pair buffer. Only *collision* shapes are
        # converted; visual meshes keep their full geometry for rendering.
        import newton as _newton

        for i in range(shape_start, shape_end):
            if (
                builder.shape_type[i] == int(_newton.GeoType.MESH)
                and builder.shape_flags[i] & _newton.ShapeFlags.COLLIDE_SHAPES
            ):
                builder.shape_type[i] = int(_newton.GeoType.CONVEX_MESH)

        # ── Selective self-collision ──
        # The URDF collision shapes are full triangle MESH (SDF) geometries, so
        # *every* intra-robot MESH-MESH pair goes through the narrow-phase
        # "triangle pair" path; at 2048 worlds an unfiltered robot overflowed
        # its buffer (5.35M > 1M) and silently dropped contacts. Visual-only
        # meshes share ``GeoType.MESH`` and must always be filtered.
        #
        # Rules (anything not listed as "allowed" is filtered):
        #   - visual-only shapes: always filtered (overflow source);
        #   - adjacent bodies (parent-child, incl. the collapsed ``base_link``
        #     attached to the world): always filtered — their meshes overlap at
        #     the shared joint by construction;
        #   - fingers vs each other / vs their parent wrist: filtered (would
        #     fight the kp=400 close servo at q=0);
        #   - arm body links (1..5) among themselves: filtered — the folded
        #     reset pose interpenetrates them and contacts would eject links;
        #   - ALLOWED: the end-effector assembly (wrist 6_Link + fingers
        #     7/8_Link) vs the arm body (1..5_Link), non-adjacent — without
        #     this the gripper passes straight through the arm links.
        import newton as _newton

        _robot_shapes = list(range(shape_start, shape_end))
        _colliding = {
            i for i in _robot_shapes if builder.shape_flags[i] & _newton.ShapeFlags.COLLIDE_SHAPES
        }

        _adjacent: set[tuple[int, int]] = set()
        for j in range(joint_start, joint_end):
            _p = builder.joint_parent[j]
            _c = builder.joint_child[j]
            _adjacent.add((_p, _c))
            _adjacent.add((_c, _p))

        def _link_name(shape_idx: int) -> str | None:
            body = builder.shape_body[shape_idx]
            if body < 0:
                return None  # world-attached (collapsed base_link)
            return builder.body_label[body].split("/")[-1]

        _EE_LINKS = {"6_Link", "7_Link", "8_Link"}
        _ARM_LINKS = {"1_Link", "2_Link", "3_Link", "4_Link", "5_Link"}

        for _a, _i in enumerate(_robot_shapes):
            for _j in _robot_shapes[_a + 1 :]:
                if _i not in _colliding or _j not in _colliding:
                    builder.add_shape_collision_filter_pair(_i, _j)  # visual mesh involved
                    continue
                _li, _lj = _link_name(_i), _link_name(_j)
                if _li is None or _lj is None:
                    builder.add_shape_collision_filter_pair(_i, _j)  # world/base-attached
                    continue
                if (builder.shape_body[_i], builder.shape_body[_j]) in _adjacent:
                    builder.add_shape_collision_filter_pair(_i, _j)  # parent-child overlap
                    continue
                _ee_arm = (_li in _EE_LINKS and _lj in _ARM_LINKS) or (
                    _lj in _EE_LINKS and _li in _ARM_LINKS
                )
                if not _ee_arm:
                    builder.add_shape_collision_filter_pair(_i, _j)  # arm-internal etc.
                # else: keep collision — gripper/wrist must not pass through the arm.

        # ── Stiff mechanical stops for the gripper prismatic joints ──
        # Travel is [0, 0.05] m, but the global ``limit_ke=1e3`` is a *soft*
        # stop: after MuJoCo's force-space scaling the effective stiffness for
        # these light DOFs is only ~150 N/m, while the kp=400 servo is capped
        # by the URDF effort limit (3 N). Any real table/cube contact force
        # then pushes the fingers far past their travel (they visibly drifted
        # away from the gripper). Raise the stop stiffness only for these two
        # low-inertia, low-speed DOFs (≈15 kN/m effective — near-rigid), which
        # does not touch the global limit stiffness that the arm needs for
        # stability at dt=0.005 s.
        # ``resolve_dof_indices`` returns *local* DOF indices (relative to the
        # import's first DOF); the builder's per-DOF lists are global.
        _base_qd = builder.joint_qd_start[joint_start]
        gripper_dofs = [
            _base_qd + _d
            for _d in NewtonRobotLoader.resolve_dof_indices(
                builder, ("7joint", "8joint"), joint_start, joint_end
            )
        ]
        for dof in gripper_dofs:
            builder.joint_limit_ke[dof] = 1.0e5
            builder.joint_limit_kd[dof] = 1.0e3

        return joint_start, joint_end

    @staticmethod
    def resolve_dof_indices(
        builder,
        joint_names: Tuple[str, ...],
        joint_start: int,
        joint_end: int,
    ) -> List[int]:
        """
        Resolve DOF indices (local to the imported articulation) by joint name.

        Args:
            builder: ``newton.ModelBuilder`` that holds the imported URDF.
            joint_names: Joint names to resolve, in the desired order.
            joint_start: First joint index of the import (from ``build_urdf``).
            joint_end: One past the last joint index of the import.

        Returns:
            List of DOF indices, one per resolved joint name.
        """
        labels = list(builder.joint_label[joint_start:joint_end])
        # Newton keeps fully qualified labels such as "fangzhenjixiebi/1joint",
        # so match on both the full label and its last path component.
        short_labels = [label.rsplit("/", 1)[-1] for label in labels]
        base_qd = builder.joint_qd_start[joint_start]

        indices: List[int] = []
        for name in joint_names:
            if name in short_labels:
                offset = short_labels.index(name)
            elif name in labels:
                offset = labels.index(name)
            else:
                logger.warning(f"Could not find joint '{name}' in the imported URDF")
                continue
            joint_index = joint_start + offset
            indices.append(builder.joint_qd_start[joint_index] - base_qd)

        return indices

    def load(self, context: Dict[str, Any]) -> LoadResult:
        """
        Load the robot into the scene.

        Args:
            context: Context containing 'scene' and 'simulator_proxy'.

        Returns:
            LoadResult with robot entity.
        """
        scene = context.get("scene")
        proxy = context.get("simulator_proxy")

        if scene is None:
            return LoadResult.error("Scene not available in context")
        if proxy is None:
            return LoadResult.error("Simulator proxy not available in context")

        urdf_path = self._config.urdf_path
        if not os.path.exists(urdf_path):
            return LoadResult.error(f"URDF file not found: {urdf_path}")

        try:
            self._robot = proxy.load_urdf(
                scene,
                urdf_path,
                pos=self._config.base_position,
                quat=self._config.base_orientation,
                fixed=self._config.fixed_base,
            )

            robot_config = context.get("robot_config", {})
            joint_names = robot_config.get("joint_names", ())

            self._motor_dof_idx = []
            for name in joint_names:
                try:
                    joint = proxy.get_joint(self._robot, name)
                    self._motor_dof_idx.append(joint.dofs_idx_local[0])
                except Exception as e:
                    logger.warning(f"Could not find joint '{name}': {e}")

            self._robot_state.initialize_joints(joint_names)

            joint_limits = robot_config.get("joint_limits", {})
            self._robot_state.joint_limits_lower = np.array(
                joint_limits.get("lower", np.zeros(len(joint_names)))
            )
            self._robot_state.joint_limits_upper = np.array(
                joint_limits.get("upper", np.zeros(len(joint_names)))
            )

            control_gains = robot_config.get("control_gains", {})
            self._robot_state.control_gains_kp = np.array(
                control_gains.get("kp", np.ones(len(joint_names)))
            )
            self._robot_state.control_gains_kv = np.array(
                control_gains.get("kv", np.ones(len(joint_names)))
            )

            force_limits = robot_config.get("force_limits", {})
            self._robot_state.force_limits_lower = np.array(
                force_limits.get("lower", np.full(len(joint_names), -10.0))
            )
            self._robot_state.force_limits_upper = np.array(
                force_limits.get("upper", np.full(len(joint_names), 10.0))
            )

            context["robot"] = self._robot
            context["motor_dof_idx"] = self._motor_dof_idx
            context["robot_state"] = self._robot_state

            logger.info(f"Robot loaded successfully with {len(self._motor_dof_idx)} joints")
            return LoadResult.ok(
                self._robot,
                num_joints=len(self._motor_dof_idx),
                joint_names=list(joint_names),
            )

        except Exception as e:
            logger.error(f"Failed to load robot: {e}")
            return LoadResult.error(str(e))

    def get_robot_state(self) -> RobotState:
        """Get the robot state."""
        return self._robot_state

    def get_motor_dof_indices(self) -> List[int]:
        """Get the motor DOF indices."""
        return self._motor_dof_idx


class NewtonObjectLoader(IObjectLoader):
    """
    Object loader implementation for the Newton simulator.

    Loads scene objects (desktops, targets, ...) from URDF files.
    """

    def __init__(self, config: Any, object_type: str = "object"):
        """
        Initialize the Newton object loader.

        Args:
            config: Object configuration (DesktopConfig or TargetObjectConfig).
            object_type: Type identifier for the object.
        """
        self._config = config
        self._object_type = object_type
        self._entity = None
        logger.info(f"NewtonObjectLoader initialized for {object_type}")

    def load(self, context: Dict[str, Any]) -> LoadResult:
        """
        Load the object into the scene.

        Args:
            context: Context containing 'scene' and 'simulator_proxy'.

        Returns:
            LoadResult with object entity.
        """
        scene = context.get("scene")
        proxy = context.get("simulator_proxy")

        if scene is None:
            return LoadResult.error("Scene not available in context")
        if proxy is None:
            return LoadResult.error("Simulator proxy not available in context")

        urdf_path = getattr(self._config, "urdf_path", None)
        if urdf_path is None:
            return LoadResult.error("Configuration does not have urdf_path")

        if not os.path.exists(urdf_path):
            logger.warning(f"Object URDF not found: {urdf_path}")
            return LoadResult.error(f"URDF file not found: {urdf_path}")

        try:
            position = getattr(self._config, "position", (0, 0, 0))
            orientation = getattr(self._config, "orientation", (1, 0, 0, 0))
            fixed = getattr(self._config, "fixed", False)

            self._entity = proxy.load_urdf(
                scene,
                urdf_path,
                pos=position,
                quat=orientation,
                fixed=fixed,
            )

            object_name = getattr(self._config, "name", self._object_type)
            context[f"{object_name}_entity"] = self._entity

            logger.info(f"Object '{object_name}' loaded at position: {position}")
            return LoadResult.ok(
                self._entity,
                name=object_name,
                position=position,
            )

        except Exception as e:
            logger.error(f"Failed to load object: {e}")
            return LoadResult.error(str(e))


class NewtonSceneLoaderFactory(ISceneLoaderFactory):
    """
    Factory for creating Newton scene loaders.

    Implements the Abstract Factory pattern for scene loader creation.
    """

    def create_loader(self, config: SceneConfig) -> ISceneLoader:
        """
        Create a Newton scene loader.

        Args:
            config: Scene configuration.

        Returns:
            NewtonSceneLoader instance.
        """
        return NewtonSceneLoader(config)

    def get_loader_type(self) -> str:
        return "newton"


class NewtonRobotLoaderFactory(IRobotLoaderFactory):
    """
    Factory for creating Newton robot loaders.

    Implements the Abstract Factory pattern for robot loader creation.
    """

    def create_loader(self, config: RobotConfig) -> IRobotLoader:
        """
        Create a Newton robot loader.

        Args:
            config: Robot configuration.

        Returns:
            NewtonRobotLoader instance.
        """
        return NewtonRobotLoader(config)

    def get_loader_type(self) -> str:
        return "newton_urdf"


class NewtonObjectLoaderFactory(IObjectLoaderFactory):
    """
    Factory for creating Newton object loaders.

    Implements the Abstract Factory pattern for object loader creation.
    """

    def create_loader(self, config: Any) -> IObjectLoader:
        """
        Create a Newton object loader.

        Args:
            config: Object configuration.

        Returns:
            NewtonObjectLoader instance.
        """
        object_type = getattr(config, "name", "object")
        return NewtonObjectLoader(config, object_type)

    def get_loader_type(self) -> str:
        return "newton_urdf"


# Module-level registries for loader factories
_scene_registry = Registry[ISceneLoaderFactory]("scene_loaders")
_robot_registry = Registry[IRobotLoaderFactory]("robot_loaders")
_object_registry = Registry[IObjectLoaderFactory]("object_loaders")

_scene_registry.register("newton", NewtonSceneLoaderFactory)
_robot_registry.register("newton_urdf", NewtonRobotLoaderFactory)
_object_registry.register("newton_urdf", NewtonObjectLoaderFactory)


class LoaderFactoryRegistry:
    """
    Registry for loader factories, backed by model.registry.Registry.

    Provides a centralized registry for creating loaders of different types,
    enabling easy extension to new simulator backends.
    """

    @classmethod
    def register_scene_factory(cls, name: str, factory_class: Type[ISceneLoaderFactory]) -> None:
        _scene_registry.register(name, factory_class)

    @classmethod
    def register_robot_factory(cls, name: str, factory_class: Type[IRobotLoaderFactory]) -> None:
        _robot_registry.register(name, factory_class)

    @classmethod
    def register_object_factory(cls, name: str, factory_class: Type[IObjectLoaderFactory]) -> None:
        _object_registry.register(name, factory_class)

    @classmethod
    def create_scene_loader(cls, name: str, config: SceneConfig) -> ISceneLoader:
        factory = _scene_registry.create(name)
        return factory.create_loader(config)

    @classmethod
    def create_robot_loader(cls, name: str, config: RobotConfig) -> IRobotLoader:
        factory = _robot_registry.create(name)
        return factory.create_loader(config)

    @classmethod
    def create_object_loader(cls, name: str, config: Any) -> IObjectLoader:
        factory = _object_registry.create(name)
        return factory.create_loader(config)
