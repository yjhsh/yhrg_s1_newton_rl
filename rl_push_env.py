"""
Push Environment for YHRG S1 Robotic Arm using Newton Simulator.

Non-prehensile manipulation task: push a cube on table surface to a goal pose.
The gripper stays closed at all times. Uses PyTorch tensors exclusively.

Simulator backend: Newton (reduced-coordinate ``SolverFeatherstone``), accessed
through the batched facades in ``view.proxy``:
  - ``NewtonArticulation``  — the manipulator (joint state, Jacobians, targets)
  - ``NewtonRigidBody``     — the cube and the goal marker

World layout note
-----------------
The scene is built once and replicated with ``ModelBuilder.replicate(spacing=0)``.
Newton's broad phase filters contacts per world, so the worlds never interact
even though they overlap exactly. The practical benefit is that world
coordinates are already environment-local, which matches the Genesis semantics
``get_pos()`` used to provide. The trade-off is that ``show_viewer`` renders all
environments superimposed; use ``--num_envs 1`` for visualization.

Scene layout (from model.robot_config.SceneProfile):
  - Table: Box at (0,0,0.2), size (0.4,0.5,0.4), top surface at Z=0.4
  - Robot: base at (-0.4, 0.0, 0.4), 180° yaw around Z
  - Object: cube on table surface, XY in [-0.16,0.16]×[-0.20,0.20]
  - Goal: XY in [-0.16,0.16]×[-0.20,0.20]

Control architecture (18D action space):
  - delta_pose [0:6]: task-space error → DLS IK → joint-space delta
  - kp_raw [6:12]: adaptive proportional gains mapped from [-1,1] to physical range
  - kd_raw [12:18]: adaptive derivative gains mapped from [-1,1] to physical range

Observation vector (32D):
  [0:3]   EE position (local XYZ)
  [3:7]   EE quaternion (local wxyz)
  [7:10]  Cube position (local XYZ)
  [10:12] Goal position (local XY)
  [12:15] EE→Cube vector (local XYZ)
  [15:17] Cube→Goal vector (local XY)
  [17:23] Arm joint positions (6)
  [23:29] Arm joint velocities (6)
  [29:32] Point cloud centroid (local XYZ)

Tuning rationale (Tuned 2026-06 based on diagnostic analysis):
  - episode_length_s=7.0: gives agent enough steps to complete the task
  - PD gains (recalibrated 2026-09, see model.robot_config.CONTROL_KP):
      kp=[100,200,200,100,50,50], kd=[20,20,20,10,10,5]
      The former "low-gain compliant" values (kp=[10,20,20,10,5,5]) could not
      hold the arm at the configured pose — a held target drifted ~1.5 rad,
      which is why the arm looked frozen during training. Holding accuracy
      measured at kp x1: 0.127 rad, x10: ~0.010 rad, x20: 0.008 rad.
  - force_range: [48,48,48,18,18,10] Nm — headroom for the transient torques
    the adaptive gains produce; the old limits were clipping them.

External dependencies (used as-is, no modifications):
  - newton / warp: Simulator implementation (production-grade)
  - torch: PyTorch tensor operations
  - tensordict: TensorDict for rsl_rl 4.x observation interface

Quaternion convention
---------------------
Genesis reports quaternions as (w, x, y, z); Newton/Warp use (x, y, z, w).
Every quaternion crossing the simulator boundary is converted here so that the
observation layout, ``_get_cube_yaw()`` and the reset code keep using the
(w, x, y, z) ordering the rest of the project expects.
"""

import math

import numpy as np
import torch
from tensordict import TensorDict

import newton
import warp as wp

from model.loaders import NewtonRobotLoader, quat_xyzw
from view.proxy import NewtonArticulation, NewtonRigidBody, quat_rotate_torch

from model.robot_config import (
    URDF_PATH,
    JOINT_NAMES,
    ARM_DOF,
    GRIPPER_DOF,
    TOTAL_DOF,
    EE_LINK_NAME,
    GRIPPER_LEFT_LINK,
    GRIPPER_RIGHT_LINK,
    CONTROL_KP,
    CONTROL_KV,
    FORCE_LIMITS_LOWER,
    FORCE_LIMITS_UPPER,
    RL_PUSH_DEFAULT_POSITION,
    get_rl_push_profile,
    get_rl_push_scene_profile,
    get_rl_push_object_randomizer,
    get_rl_push_success_config,
    ObjectRandomizer,
)
from model.pointcloud import (
    PointCloudProcessor,
    PointCloudConfig,
    get_rl_push_pointcloud_config,
    transform_pcd_by_pose,
    farthest_point_sample,
)
from controller.ik_solver import DLSSolver
from controller.task_space_controller import TaskSpaceController


# Density used to derive the cube mass from its size [kg/m^3]. Genesis derived
# the mass from the box morph defaults; the Newton backend needs it explicitly.
OBJECT_DENSITY = 1000.0


class PushEnv:
    """
    RL environment for non-prehensile pushing with YHRG S1 arm.

    The robot must push a cube on the ground surface to a randomly sampled
    goal position. The gripper remains closed throughout.

    Reward, termination, randomization, and success logic migrated from
    IsaacGym AbbPushBox (s2_cyclic_geometry_obs.py).

    Compatible with rsl_rl 4.x OnPolicyRunner / VecEnv interface.
    """

    # ── YHRG S1 robot constants (from model.robot_config) ──
    URDF_PATH = URDF_PATH
    JOINT_NAMES = JOINT_NAMES
    ARM_DOF = ARM_DOF
    GRIPPER_DOF = GRIPPER_DOF
    TOTAL_DOF = TOTAL_DOF
    EE_LINK_NAME = EE_LINK_NAME
    GRIPPER_LEFT_LINK = GRIPPER_LEFT_LINK
    GRIPPER_RIGHT_LINK = GRIPPER_RIGHT_LINK

    # ── Success & termination thresholds (from SuccessConfig) ──
    # SUCCESS_THRESHOLD is now read from self.success_config.dist_threshold

    # ── Environment grid spacing (visualization) ──
    ENV_SPACING = (2.0, 2.0)  # 2m between adjacent environments

    def __init__(
        self,
        num_envs: int = 16,
        show_viewer: bool = False,
        ctrl_dt: float = 0.01,
        episode_length_s: float = 5.0,
        action_scale: float = 0.30,
        object_randomizer: ObjectRandomizer = None,
        use_cuda_graph: bool = True,
        gain_sync_interval: int = 16,
    ) -> None:
        self.num_envs = num_envs
        self.num_obs = 32
        self.num_actions = 18
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.ctrl_dt = ctrl_dt
        self.action_scale = action_scale
        self.max_episode_length = math.ceil(episode_length_s / ctrl_dt)

        # ── Adaptive-gain sync decimation (see :meth:`_notify_gains_changed`) ──
        # Push the policy's kp/kd to the MuJoCo backend every N control steps
        # (N=16 at 100 Hz control ≈ 6 Hz) instead of every step.
        self.gain_sync_interval = max(1, int(gain_sync_interval))
        self._steps_since_gain_sync = 0

        # ── CUDA-graph capture of the physics step (see :meth:`_step_physics`) ──
        # Enabled by default on CUDA; silently falls back to kernel launches if
        # capture fails or the device is not CUDA.
        self.use_cuda_graph = bool(use_cuda_graph) and self.device.type == "cuda"
        self._wp_device = wp.get_device(str(self.device)) if self.use_cuda_graph else None
        self._physics_graph = None

        # Control period 0.01 s with 2 physics substeps -> dt = 0.005 s.
        # tests/sweep_stability.py shows this is the largest step that keeps the
        # reduced-coordinate solver stable with the tuned limit stiffness.
        self.sim_substeps = 2
        self.sim_dt = self.ctrl_dt / self.sim_substeps

        # A CUDA-graph replay cannot re-run the Python-side state swap, so the
        # two state buffers must keep fixed roles: with an even number of
        # substeps the newest results always land back in ``state_0``. An odd
        # count would leave them in ``state_1`` after replay, so fall back to
        # plain kernel launches in that case.
        if self.use_cuda_graph and self.sim_substeps % 2 != 0:
            self.use_cuda_graph = False

        # rsl_rl 4.x expects env.cfg
        self.cfg = {
            "num_envs": num_envs,
            "num_obs": self.num_obs,
            "num_actions": self.num_actions,
            "episode_length_s": episode_length_s,
            "ctrl_dt": ctrl_dt,
        }

        # ── Scene profile (from Model layer) ──
        self.scene_profile = get_rl_push_scene_profile()
        sp = self.scene_profile

        # ── Success config (from Model layer) ──
        self.success_config = get_rl_push_success_config()

        # ── build a single world ──
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        # Numerical settings from tests/sweep_stability.py: the reduced
        # coordinate solver diverges at dt = 0.005 s unless the limit stiffness
        # stays around 1e3, and the tiny wrist/finger inertias need an armature.
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            limit_ke=1.0e3,
            limit_kd=1.0e1,
            damping=0.1,
            armature=0.01,
        )

        shape_cfg = newton.ModelBuilder.ShapeConfig()
        # Contact stiffness/friction stay at the Newton defaults; a stiffer ke
        # makes the light cube (0.128 kg) bounce out of the table on the first
        # contact (see tests/sweep_stability.py notes).
        shape_cfg.mu = 0.8
        builder.default_shape_cfg = shape_cfg
        self.shape_cfg = shape_cfg

        # ground plane
        builder.add_ground_plane()

        # table (from SceneProfile) — static geometry attached to the world
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(*sp.table_pos), wp.quat_identity()),
            hx=sp.table_dims[0] / 2.0,
            hy=sp.table_dims[1] / 2.0,
            hz=sp.table_dims[2] / 2.0,
            cfg=shape_cfg,
            color=(0.6, 0.4, 0.2),
            label="table",
        )
        table_shape_idx = builder.shape_count - 1
        robot_shape_begin = builder.shape_count  # first robot shape (table already added)

        # robot (from SceneProfile: positioned on table top, 180° yaw).
        # Delegated to the shared import so the interactive MVC path and this
        # environment always use identical URDF options.
        joint_start, joint_end = NewtonRobotLoader.build_urdf(
            builder,
            self.URDF_PATH,
            base_position=sp.robot_base_pos,
            base_orientation=sp.robot_base_quat,  # (w, x, y, z)
            fixed_base=True,
        )

        robot_shape_end = builder.shape_count

        # ── Force limits (from model.robot_config) ──
        # Newton stores a single symmetric effort limit per DOF, so the tighter
        # of the two Genesis-style bounds is used (they are symmetric here).
        # They have to be written before finalize(), i.e. here on the builder.
        profile = get_rl_push_profile()
        base_qd = builder.joint_qd_start[joint_start]
        for local_dof, lower, upper in zip(
            NewtonRobotLoader.resolve_dof_indices(
                builder, self.JOINT_NAMES, joint_start, joint_end
            ),
            profile.force_lower,
            profile.force_upper,
        ):
            builder.joint_effort_limit[base_qd + local_dof] = float(
                min(abs(float(lower)), abs(float(upper)))
            )

        # push target cube (dynamic, spawned on table surface)
        # ── Object randomizer ──
        self.obj_randomizer = object_randomizer or get_rl_push_object_randomizer()
        sampled_sizes = self.obj_randomizer.sample_sizes(num_envs)

        # Z height: use max possible size to ensure no penetration
        obj_z = self.obj_randomizer.get_max_obj_z(sp.table_top_z, sp.z_eps)

        if self.obj_randomizer.size_enabled:
            raise NotImplementedError(
                "per-environment object size randomization is not supported by "
                "the Newton backend (worlds are replicated from one builder)"
            )

        cube_half = [s / 2.0 for s in sp.cube_size]
        cube_mass = OBJECT_DENSITY * float(
            sp.cube_size[0] * sp.cube_size[1] * sp.cube_size[2]
        )
        # Solid cube: I = (1/6) m s^2 = (2/3) m h^2 about any principal axis.
        cube_inertia = (2.0 / 3.0) * cube_mass * cube_half[0] ** 2
        inertia = wp.mat33(
            (cube_inertia, 0.0, 0.0),
            (0.0, cube_inertia, 0.0),
            (0.0, 0.0, cube_inertia),
        )

        cube_body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, obj_z), wp.quat_identity()),
            mass=cube_mass,
            inertia=inertia,
            label="cube",
        )
        builder.add_shape_box(
            body=cube_body,
            hx=cube_half[0],
            hy=cube_half[1],
            hz=cube_half[2],
            cfg=shape_cfg,
            color=(0.2, 0.8, 0.2),
            label="cube",
        )

        # Goal marker (visualization only): thin red square on the table top.
        # ``as_site=True`` makes it a non-colliding reference shape, matching
        # the Genesis ``collision=False`` behaviour.
        goal_body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, sp.goal_marker_z), wp.quat_identity()),
            # Newton treats mass-0 kinematic bodies as static geometry, but
            # MuJoCo (an alternative Newton backend) rejects them outright, so
            # a tiny mass keeps the marker valid for every backend.
            mass=1.0e-6,
            is_kinematic=True,
            label="goal",
        )
        builder.add_shape_box(
            body=goal_body,
            hx=sp.goal_marker_size / 2.0,
            hy=sp.goal_marker_size / 2.0,
            hz=sp.goal_marker_thickness / 2.0,
            as_site=True,
            color=sp.goal_marker_color,
            label="goal",
        )
        # Belt and braces: the marker is purely visual. If the backend ignored
        # the site flag, the massless kinematic body would take part in contact
        # resolution and blow up the cube resting on it.
        #
        # Verified 2026-09: this is confirmed to work. ``SolverMuJoCo`` runs
        # with ``_use_mujoco_contacts=True``, so it generates its own contacts
        # from the MuJoCo geoms and ignores the Newton contacts produced by
        # ``CollisionPipeline`` altogether. Newton maps a shape without
        # COLLIDE_SHAPES to ``contype=0`` *and* ``conaffinity=0``, so this
        # marker never becomes a MuJoCo geom and cannot block the cube.
        # (Judging collision filtering from ``contacts.rigid_contact_*`` is
        # misleading here: those arrays are the *ignored* Newton candidates and
        # their ``rigid_contact_force`` stays zero.)
        builder.shape_flags[builder.shape_count - 1] &= ~newton.ShapeFlags.COLLIDE_SHAPES

        # Store per-env sizes and randomization state
        self._sampled_sizes = sampled_sizes
        self._sampled_masses = None
        self._sampled_frictions = None

        # ── Arm ↔ table collision filter ──
        # The folded reset pose puts the *base-side* arm links level with the
        # table top. Newton approximates collision meshes with convex hulls, so
        # those links interpenetrate the table and the contact solver ejects
        # them. Only those links are filtered: the wrist (6_Link) and the
        # gripper fingers (7_Link / 8_Link) keep their table collision so the
        # fingers properly rest on / push against the surface instead of
        # passing straight through it. The arm ↔ cube contact (the actual
        # task) is unaffected either way — the cube was never filtered.
        #
        # Verified 2026-09: this filter does reach the MuJoCo backend. Newton
        # compiles the explicit filter pairs into per-geom bitmasks, which gives
        # the table ``conaffinity=61`` while links 1-3 keep ``contype=2`` — the
        # two are disjoint, so those links never touch the table. Links 4-8 keep
        # bit patterns that still overlap the table, exactly the intended split.
        _TABLE_FILTERED_LINKS = ("1_Link", "2_Link", "3_Link")
        for i in range(robot_shape_begin, robot_shape_end):
            body = builder.shape_body[i]
            if body < 0:
                # World-attached shape (the collapsed ``base_link``): its
                # collision flags are already cleared by ``build_urdf``;
                # keep the filter pair as belt-and-braces.
                builder.add_shape_collision_filter_pair(table_shape_idx, i)
                continue
            # Body labels are hierarchical ("{articulation}/{link}"); match on
            # the trailing link name.
            link_name = builder.body_label[body].split("/")[-1]
            if link_name in _TABLE_FILTERED_LINKS:
                builder.add_shape_collision_filter_pair(table_shape_idx, i)

        # ── Replicate the world and finalize ──
        # spacing=(0,0,0): Newton's broad phase filters contacts per world, so
        # the worlds never interact and world coords are already env-local.
        worlds = newton.ModelBuilder(up_axis=newton.Axis.Z)
        worlds.replicate(builder, world_count=num_envs, spacing=(0.0, 0.0, 0.0))
        self.model = worlds.finalize()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        # ``broad_phase='nxn'`` is the only implementation that filters
        # candidate pairs per world, which is what makes the zero-spaced
        # replication in this environment safe (the worlds overlap exactly).
        # ``max_triangle_pairs`` is the narrow-phase buffer for MESH-MESH (SDF)
        # "triangle pair" contacts. Self-collision is selectively filtered in
        # ``NewtonRobotLoader.build_urdf`` (visual meshes and adjacent/arm-body
        # pairs are removed; the gripper-vs-arm pairs the task needs remain),
        # so the count stays far below this env-scaled headroom. Without it the
        # buffer silently dropped contacts after warning (5.35M > 1M at 2048
        # envs). Cost is only ~12 B/entry plus the reducer table.
        self.pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="nxn",
            max_triangle_pairs=max(2_000_000, 4000 * num_envs),
        )
        self.contacts = self.pipeline.contacts()
        # MuJoCo is Newton's most robust backend for fixed-base manipulators:
        # it resolves the arm/table contacts and the light push cube without
        # the numerical divergence observed with SolverFeatherstone here.
        # ``njmax``/``nconmax`` cap the per-world constraint/contact buffers.
        # Their default is estimated from the *initial* (contact-light) state,
        # which is too small once a few of the many replicated worlds pile up
        # contacts during training -> "nefc overflow - please increase njmax".
        self.solver = newton.solvers.SolverMuJoCo(self.model, njmax=512, nconmax=256)

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)

        # ── Batched facades (the controller talks to these) ──
        self.robot = NewtonArticulation(self.model, "fangzhenjixiebi*")
        self.cube = NewtonRigidBody(self.model, "cube*")
        self.goal_entity = NewtonRigidBody(self.model, "goal*")
        self._bind_state()

        # ── Viewer (``--vis``) ──
        # Newton's OpenGL viewer renders one window for the whole batched
        # model. Because the worlds are replicated with zero spacing every
        # environment is drawn on top of the others; ``--num_envs 1`` gives a
        # clean view of a single scene (see the world-layout note above).
        self.show_viewer = show_viewer
        self._render_time = 0.0
        if show_viewer:
            if num_envs > 1:
                print(
                    f"[viewer] --vis with num_envs={num_envs}: all environments are "
                    f"rendered superimposed (zero-spacing replication). "
                    f"Use --num_envs 1 for a clean view."
                )
            self.viewer = newton.viewer.ViewerGL()
            self.viewer.set_model(self.model)
            # Scene is centred at the origin: table top z=0.4, robot base
            # (-0.4, 0, 0.4), cube around z=0.42.
            self.viewer.set_camera(
                pos=wp.vec3(1.2, -1.2, 1.1), pitch=-30.0, yaw=135.0
            )
        else:
            self.viewer = None

        # ── DOF indices (local to the arm articulation) ──
        self.motor_dof_idx = torch.tensor(
            NewtonRobotLoader.resolve_dof_indices(
                builder, self.JOINT_NAMES, joint_start, joint_end
            ),
            dtype=torch.int32,
            device=self.device,
        )
        self.arm_dof_idx = self.motor_dof_idx[:self.ARM_DOF]
        self.gripper_dof_idx = self.motor_dof_idx[self.ARM_DOF:]

        # links - gripper fingers for EE position calculation (indices inside
        # the articulation, resolved by the batched facade)
        self._ee_link = self.robot.link_index(self.EE_LINK_NAME)
        self._gripper_left = self.robot.link_index(self.GRIPPER_LEFT_LINK)
        self._gripper_right = self.robot.link_index(self.GRIPPER_RIGHT_LINK)

        # ── Base friction on cube (for friction ratio randomization) ──
        if self.obj_randomizer.friction_enabled:
            raise NotImplementedError(
                "per-environment friction randomization is not supported by the "
                "Newton backend (materials are baked into the model)"
            )

        # ── Task-space controller (DLS IK + Adaptive PD) ──
        self.task_ctrl = TaskSpaceController(
            robot=self.robot,
            ee_link=self._ee_link,
            arm_dof_idx=self.arm_dof_idx,
            gripper_dof_idx=self.gripper_dof_idx,
            ik_solver=DLSSolver(damping=0.01),
            delta_pos_scale=0.05,
            delta_orient_scale=0.3,
            kp_range=(profile.kp[:ARM_DOF] * 0.5, profile.kp[:ARM_DOF] * 2.0),
            kd_range=(profile.kv[:ARM_DOF] * 0.5, profile.kv[:ARM_DOF] * 2.0),
            gripper_kp=400.0,
            gripper_kd=40.0,
        )
        # Set initial PD gains to mid-range values
        self.task_ctrl.set_initial_gains(num_envs)
        # Flush those initial gains into the MuJoCo backend too — the very
        # first ``step()`` would otherwise run on the gains baked in at model
        # construction time (see :meth:`_notify_gains_changed`). Forced, because
        # this is the baseline every later (decimated) sync builds on.
        self._notify_gains_changed(force=True)

        # ── Point cloud processor ──
        self.pc_config = get_rl_push_pointcloud_config()
        self.pc_processor = PointCloudProcessor(self.pc_config)

        # Per-env object sizes tensor for point cloud sampling
        # Initialize from sampled_sizes (set during object creation above)
        self._obj_sizes = torch.tensor(
            self._sampled_sizes, dtype=torch.float32, device=self.device
        )  # [n_envs, 3]

        # Current point cloud buffer
        self._current_pcd = torch.zeros(
            (num_envs, self.pc_config.n_points, 3), dtype=torch.float32, device=self.device
        )

        # Goal point cloud buffer (world frame, generated at reset)
        self._goal_pcd = torch.zeros_like(self._current_pcd)

        # FPS-downsampled point clouds for ICP (precomputed at reset)
        self._fps_pcd = torch.zeros(
            (num_envs, self.pc_config.icp_n_points, 3), dtype=torch.float32, device=self.device
        )
        self._fps_goal_pcd = torch.zeros_like(self._fps_pcd)

        # Cache for _is_success() result (avoid double computation per step)
        self._cached_is_success = None

        # default joint angles — from model.robot_config (RL_PUSH_DEFAULT_POSITION)
        # j1=-0.053, j2=2.079, j3=0.974, j4=0.429 → EE at (0.294, -0.015, 0.093)
        # This places EE in the center of cube workspace (X=0.2-0.4, Y=-0.2-0.2)
        self.default_dof_pos = torch.tensor(
            RL_PUSH_DEFAULT_POSITION,
            dtype=torch.float32,
            device=self.device,
        )

        # ── Move bounds tensors to device (from SceneProfile) ──
        self.obj_bounds_low = torch.tensor(sp.obj_bounds_low, device=self.device)
        self.obj_bounds_high = torch.tensor(sp.obj_bounds_high, device=self.device)
        self.obj_pos_low = torch.tensor(sp.obj_xy_low, device=self.device)
        self.obj_pos_high = torch.tensor(sp.obj_xy_high, device=self.device)
        self.goal_pos_low = torch.tensor(sp.goal_xy_low, device=self.device)
        self.goal_pos_high = torch.tensor(sp.goal_xy_high, device=self.device)

        # ── Environment origins (for local↔world coordinate conversion) ──
        # The worlds are replicated with zero spacing, so the world origin of
        # every environment is (0, 0, 0) and world coordinates already are
        # environment-local. The per-world offsets are still computed so that
        # local_to_world()/world_to_local() keep working for monitoring.
        world_offsets = newton.utils.compute_world_offsets(
            num_envs,
            spacing=self.ENV_SPACING,
            up_axis=self.model.up_axis,
        )
        self.env_origins = torch.as_tensor(
            np.asarray(world_offsets), dtype=torch.float32, device=self.device
        ).reshape(self.num_envs, 3)

        # ── buffers ──
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), dtype=torch.float32, device=self.device)
        self.rew_buf = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.reset_buf = torch.ones((self.num_envs,), dtype=torch.bool, device=self.device)
        self.time_out_buf = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.episode_length_buf = torch.zeros((self.num_envs,), dtype=torch.int32, device=self.device)
        self.actions = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float32, device=self.device)

        # goal positions (XY on ground surface)
        self.goal_pos_xy = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)

        # previous cube-to-goal distance for velocity-based push reward
        self.prev_cube_goal_dist = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)

        # success tracking
        self.success_buf = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)

        self.extras = dict()

        # episode reward sums for logging
        self.episode_sums = {
            "approach_reward": torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device),
            "push_reward": torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device),
            "dist_reward": torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device),
            "success_reward": torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device),
            "action_penalty": torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device),
        }

        # ── Initialise every environment ──
        # rsl_rl's ``OnPolicyRunner.learn()`` never calls ``env.reset()``: it
        # starts by reading ``env.get_observations()`` and then steps
        # (rsl_rl/runners/on_policy_runner.py:62). Because
        # ``_compute_termination()`` overwrites ``reset_buf`` *before*
        # ``_reset_idx()`` consumes it, the ``torch.ones`` above never triggers
        # a first-step reset — so without this call every world would start at
        # the URDF build pose (all joints zero) and only be initialised once its
        # first episode happened to time out, which is why most environments sat
        # at the zero pose during the first iterations. Reset here so training
        # always begins from the configured initial pose.
        self.reset()

    # ────────────────────── reset ──────────────────────

    def _notify_gains_changed(self, force: bool = False) -> None:
        """Push in-place PD-gain / effort-limit edits into the MuJoCo backend.

        Newton keeps the control gains on the *model* (``joint_target_ke`` /
        ``joint_target_kd``), while :class:`~newton.solvers.SolverMuJoCo`
        copies them into MuJoCo's own ``actuator_gainprm`` /
        ``actuator_biasprm`` tables while building ``mj_model``. That copy is
        only refreshed on an explicit model-change notification
        (``ModelFlags.JOINT_DOF_PROPERTIES`` covers ``joint_target_ke``,
        ``joint_target_kd`` and ``joint_effort_limit``).

        Both the adaptive gains written every step by
        :meth:`controller.task_space_controller.TaskSpaceController.compute_and_apply`
        and :meth:`~controller.task_space_controller.TaskSpaceController.set_initial_gains`
        therefore need this call, otherwise the solver keeps using stale gains.

        The notification is NOT cheap, though: ``JOINT_DOF_PROPERTIES`` forces
        ``need_const_0``, which re-runs ``set_const_0`` twice over every
        replicated world and invalidates the cached contact fast path. Profiling
        (profile2.svg) put it at 7.6% of the whole run when called every step.

        The adaptive gains only drift slowly compared with the 100 Hz control
        rate, so they are pushed to the solver every ``gain_sync_interval``
        steps instead (default 16 → ~6 Hz) and stay piecewise-constant in
        between. Call with ``force=True`` for the one-off flushes that must not
        be skipped (initial gains, reset).

        Args:
            force: Sync now regardless of the decimation counter.
        """
        if not force:
            self._steps_since_gain_sync += 1
            if self._steps_since_gain_sync < self.gain_sync_interval:
                return
        self._steps_since_gain_sync = 0
        self.solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)

    def _bind_state(self) -> None:
        """Rebind the facades to the state that holds the latest results."""
        self.robot.bind(self.state_0, self.control)
        self.cube.bind(self.state_0)
        self.goal_entity.bind(self.state_0)

    def _step_physics_raw(self) -> None:
        """Run the solver for one control period — GPU work only.

        Deliberately free of Python-side bookkeeping so it can be recorded into
        a Warp CUDA graph: it only launches kernels and never reassigns
        :attr:`state_0` / :attr:`state_1`, so both state buffers keep fixed
        addresses and a replayed graph writes to the very same memory. The
        ping-pong uses locals instead, and because ``sim_substeps`` is even the
        newest results end up back in :attr:`state_0`.
        """
        s0, s1 = self.state_0, self.state_1
        for _ in range(self.sim_substeps):
            s0.clear_forces()
            self.pipeline.collide(s0, self.contacts)
            self.solver.step(s0, s1, self.control, self.contacts, self.sim_dt)
            s0, s1 = s1, s0

    def _capture_physics_graph(self) -> None:
        """Record :meth:`_step_physics_raw` into a CUDA graph.

        One real launch runs first so every lazily-created array and compiled
        module exists before recording — a graph can neither allocate memory nor
        load modules while being replayed.

        Recording itself does not execute anything, so this warm-up launch is
        what actually advances the simulation on this call; the graph is only
        instantiated for later replay.
        """
        self._step_physics_raw()
        try:
            with wp.ScopedCapture(
                device=self._wp_device, force_module_load=True
            ) as capture:
                self._step_physics_raw()
        except Exception as exc:  # driver / CUDA-version dependent
            self.use_cuda_graph = False
            print(
                "[PushEnv] CUDA-graph capture failed "
                f"({type(exc).__name__}: {exc}); falling back to kernel launches."
            )
            return
        self._physics_graph = capture.graph
        print(
            f"[PushEnv] physics captured into a CUDA graph "
            f"({self.sim_substeps} substep(s) x {self.num_envs} envs)."
        )

    def _step_physics(self) -> None:
        """Advance the simulation by one control period (``sim_substeps``)."""
        if self.use_cuda_graph:
            if self._physics_graph is None:
                # First call: warms up, records the graph, advances once.
                self._capture_physics_graph()
            else:
                wp.capture_launch(self._physics_graph)
        else:
            self._step_physics_raw()
        self._bind_state()

        # Hard mechanical stops for the gripper fingers. Their prismatic travel
        # is [0, 0.05] m; the solver's joint-limit constraint alone does not
        # hold them (they slid steadily open and drifted far away from the
        # gripper), so the two gripper DOFs are projected back into their
        # travel and their velocity is zeroed whenever a stop is reached.
        self.robot.clamp_dofs_position(
            lower=(0.0, 0.0),
            upper=(0.05, 0.05),
            dofs_idx=(self.ARM_DOF, self.ARM_DOF + 1),
        )

    def reset(self) -> tuple:
        """Full reset of all environments."""
        self.reset_buf[:] = True
        self._reset_idx(self.reset_buf)
        self._step_physics()
        self._compute_obs()
        # Show the freshly reset scene immediately (no-op unless ``--vis``).
        self.render()
        return self.get_observations()

    def _reset_idx(self, envs_idx) -> None:
        """Reset selected environments."""
        num_reset = envs_idx.sum().item() if envs_idx.dtype == torch.bool else len(envs_idx)
        if num_reset == 0:
            self.extras["log"] = {}
            return

        # ── Compute per-episode stats BEFORE resetting ──
        log_dict = {}

        if envs_idx.dtype == torch.bool:
            success_rate = self.success_buf[envs_idx].mean()
        else:
            success_rate = self.success_buf[envs_idx].mean() if num_reset > 0 else torch.tensor(0.0, device=self.device)
        log_dict["/success_rate"] = success_rate.item()

        for key, value in self.episode_sums.items():
            if envs_idx.dtype == torch.bool:
                n = envs_idx.sum()
                mean = torch.where(n > 0, value[envs_idx].sum() / n, torch.tensor(0.0, device=self.device))
            else:
                mean = value[envs_idx].mean() if num_reset > 0 else torch.tensor(0.0, device=self.device)
            log_dict["/" + key] = mean.item()
        
        self.extras["log"] = log_dict

        # ── Reset robot ──
        # The facade always zeroes joint velocities, matching zero_velocity=True.
        self.robot.set_qpos(self.default_dof_pos, envs_idx=envs_idx)

        # ── Randomise push object position ──
        # NOTE: the batched writes use an articulation mask, so a bool envs_idx
        # requires a full-batch tensor of shape [num_envs, *].
        is_bool_idx = envs_idx.dtype == torch.bool
        rand_xy = torch.rand(num_reset, 2, device=self.device)
        obj_xy = self.obj_pos_low + rand_xy * (self.obj_pos_high - self.obj_pos_low)
        # Z height: use max possible size to ensure no penetration for all variants
        obj_z_val = self.obj_randomizer.get_max_obj_z(self.scene_profile.table_top_z, self.scene_profile.z_eps)
        obj_z = torch.full((num_reset, 1), obj_z_val, device=self.device)
        cube_pos_subset = torch.cat([obj_xy, obj_z], dim=-1)
        if is_bool_idx:
            cube_pos = torch.empty(self.num_envs, 3, device=self.device)
            cube_pos[envs_idx] = cube_pos_subset
        else:
            cube_pos = cube_pos_subset
        self.cube.set_pos(cube_pos, envs_idx=envs_idx)

        # Random Z-axis orientation (yaw range from SceneProfile: [-90°, 90°])
        rand_yaw = (torch.rand(num_reset, device=self.device) * 2 - 1) * self.scene_profile.obj_yaw_range
        half_yaw = rand_yaw * 0.5
        # Built in (w, x, y, z) as in Genesis, then converted to Warp (x, y, z, w).
        cube_quat_subset_wxyz = torch.stack([
            torch.cos(half_yaw),
            torch.zeros_like(half_yaw),
            torch.zeros_like(half_yaw),
            torch.sin(half_yaw),
        ], dim=-1)
        cube_quat_subset = torch.stack([
            cube_quat_subset_wxyz[:, 1],
            cube_quat_subset_wxyz[:, 2],
            cube_quat_subset_wxyz[:, 3],
            cube_quat_subset_wxyz[:, 0],
        ], dim=-1)
        if is_bool_idx:
            cube_quat = torch.zeros(self.num_envs, 4, device=self.device)
            cube_quat[envs_idx] = cube_quat_subset
        else:
            cube_quat = cube_quat_subset
        self.cube.set_quat(cube_quat, envs_idx=envs_idx)

        # Zero the cube's linear AND angular velocity so a reset starts from rest.
        # set_pos/set_quat only rewrite the free-joint coordinate block (joint_q)
        # and re-run FK, which derives body_qd from the (unchanged) joint_qd -- the
        # stale velocity from the previous episode therefore survives and is
        # integrated on the next step, making the cube keep sliding/spinning.
        # SolverMuJoCo keeps the true velocity in state.joint_qd, so it must be
        # zeroed here; body_qd is cleared in step with it for an immediately
        # consistent readout.
        self.cube.set_vel(torch.zeros(num_reset, 6, device=self.device), envs_idx=envs_idx)

        # ── Mass randomization ──
        mass_tensor = self.obj_randomizer.sample_masses(num_reset, self.device)
        if mass_tensor is not None:
            raise NotImplementedError(
                "per-environment mass randomization is not supported by the "
                "Newton backend (inertias are baked into the model)"
            )

        # ── Friction randomization ──
        friction_ratio = self.obj_randomizer.sample_friction_ratios(num_reset, self.device)
        if friction_ratio is not None:
            raise NotImplementedError(
                "per-environment friction randomization is not supported by the "
                "Newton backend (materials are baked into the model)"
            )

        # ── Randomise goal position ──
        # Ensure goal is at least MIN_INIT_DIST away from cube to avoid trivial success
        MIN_INIT_DIST = self.success_config.dist_threshold + 0.03  # > dist_threshold so episode doesn't start already done
        # Vectorized rejection sampling: generate candidates and keep valid ones
        goal_range = self.goal_pos_high - self.goal_pos_low
        new_goal_xy = self.goal_pos_low + torch.rand(num_reset, 2, device=self.device) * goal_range
        dist = torch.linalg.norm(new_goal_xy - obj_xy, dim=-1)
        invalid = dist < MIN_INIT_DIST
        # Retry invalid entries (up to 5 rounds of vectorized rejection)
        for _ in range(5):
            if not invalid.any():
                break
            n_invalid = invalid.sum()
            retry = self.goal_pos_low + torch.rand(n_invalid, 2, device=self.device) * goal_range
            new_goal_xy[invalid] = retry
            dist = torch.linalg.norm(new_goal_xy - obj_xy, dim=-1)
            invalid = dist < MIN_INIT_DIST
        if envs_idx.dtype == torch.bool:
            self.goal_pos_xy[envs_idx] = new_goal_xy
        else:
            self.goal_pos_xy[envs_idx] = new_goal_xy

        # Goal yaw (currently fixed at 0). To randomize the goal orientation,
        # sample it here and it will be picked up by both the marker rotation
        # below and the goal point-cloud generation.
        goal_yaw = torch.zeros(num_reset, device=self.device)

        # ── Sync goal marker (visualization only) ──
        # Same full-batch set_pos/set_quat pattern as the cube (Genesis'
        # zero-copy fast path requires [num_envs, *] shapes for bool masks).
        goal_z_val = self.scene_profile.goal_marker_z
        goal_pos_subset = torch.cat([
            new_goal_xy,
            torch.full((num_reset, 1), goal_z_val, device=self.device),
        ], dim=-1)
        if is_bool_idx:
            goal_pos = torch.empty(self.num_envs, 3, device=self.device)
            goal_pos[envs_idx] = goal_pos_subset
        else:
            goal_pos = goal_pos_subset
        self.goal_entity.set_pos(goal_pos, envs_idx=envs_idx)

        half_yaw = goal_yaw * 0.5
        # (w, x, y, z) -> Warp (x, y, z, w)
        goal_quat_subset_wxyz = torch.stack([
            torch.cos(half_yaw),
            torch.zeros_like(half_yaw),
            torch.zeros_like(half_yaw),
            torch.sin(half_yaw),
        ], dim=-1)
        goal_quat_subset = torch.stack([
            goal_quat_subset_wxyz[:, 1],
            goal_quat_subset_wxyz[:, 2],
            goal_quat_subset_wxyz[:, 3],
            goal_quat_subset_wxyz[:, 0],
        ], dim=-1)
        if is_bool_idx:
            goal_quat = torch.zeros(self.num_envs, 4, device=self.device)
            goal_quat[envs_idx] = goal_quat_subset
        else:
            goal_quat = goal_quat_subset
        self.goal_entity.set_quat(goal_quat, envs_idx=envs_idx)

        # ── Regenerate point cloud for reset envs ──
        if self.pc_config.enabled:
            reset_sizes = self._obj_sizes[envs_idx] if envs_idx.dtype != torch.bool else self._obj_sizes[envs_idx]
            new_pcd = self.pc_processor.generate_point_cloud(reset_sizes, num_reset)
            if envs_idx.dtype == torch.bool:
                self._current_pcd[envs_idx] = new_pcd
            else:
                self._current_pcd[envs_idx] = new_pcd

            # Generate goal point cloud (world frame)
            new_goal_pcd = self.pc_processor.generate_goal_point_cloud(
                new_pcd, new_goal_xy, goal_yaw
            )
            if envs_idx.dtype == torch.bool:
                self._goal_pcd[envs_idx] = new_goal_pcd
            else:
                self._goal_pcd[envs_idx] = new_goal_pcd

            # FPS downsample for ICP (precompute at reset)
            new_fps_pcd = farthest_point_sample(new_pcd, self.pc_config.icp_n_points)
            new_fps_goal_pcd = farthest_point_sample(new_goal_pcd, self.pc_config.icp_n_points)
            if envs_idx.dtype == torch.bool:
                self._fps_pcd[envs_idx] = new_fps_pcd
                self._fps_goal_pcd[envs_idx] = new_fps_goal_pcd
            else:
                self._fps_pcd[envs_idx] = new_fps_pcd
                self._fps_goal_pcd[envs_idx] = new_fps_goal_pcd

        # ── Reset episode buffers ──
        if envs_idx.dtype == torch.bool:
            self.episode_length_buf.masked_fill_(envs_idx, 0)
            self.actions.masked_fill_(envs_idx.unsqueeze(-1), 0.0)
            self.success_buf.masked_fill_(envs_idx, 0.0)
            self.prev_cube_goal_dist.masked_fill_(envs_idx, 0.0)
            for v in self.episode_sums.values():
                v.masked_fill_(envs_idx, 0.0)
        else:
            self.episode_length_buf[envs_idx] = 0
            self.actions[envs_idx] = 0.0
            self.success_buf[envs_idx] = 0.0
            self.prev_cube_goal_dist[envs_idx] = 0.0
            for v in self.episode_sums.values():
                v[envs_idx] = 0.0

    # ────────────────────── step ──────────────────────

    def step(self, actions: torch.Tensor) -> tuple:
        """Run one environment step."""
        self._cached_is_success = None  # Reset cache for new step
        self.actions = actions.clamp(-1.0, 1.0)
        self.episode_length_buf += 1

        # ── Parse 18D actions: [delta_pose(6), kp_raw(6), kd_raw(6)] ──
        delta_pose = self.actions[:, :6]
        kp_raw = self.actions[:, 6:12]
        kd_raw = self.actions[:, 12:18]

        # ── Task-space control: DLS IK + Adaptive PD ──
        self.task_ctrl.compute_and_apply(delta_pose, kp_raw, kd_raw)
        # ``compute_and_apply`` just wrote the adaptive gains in place into
        # ``model.joint_target_ke`` / ``model.joint_target_kd``. SolverMuJoCo
        # bakes those arrays into its own ``mj_model`` actuator gain parameters
        # when the model is built, so an in-place edit stays invisible to the
        # solver until it is told the DOF properties changed. Without this
        # notification the adaptive-gain half of the action space (kp_raw /
        # kd_raw — 12 of the 18 dimensions) is silently discarded and the arm
        # always runs on the gains that happened to be baked in at construction
        # time. See :meth:`_notify_gains_changed`.
        self._notify_gains_changed()

        # Step physics
        self._step_physics()

        # ── Divergence guard ──
        # A single environment whose state goes non-finite (solver divergence
        # under an extreme contact/gain configuration) must never leak NaN into
        # the rollout: NaN observations → NaN PPO loss → NaN gradients → NaN
        # policy parameters, which crashes the update with
        # "RuntimeError: normal expects all elements of std >= 0.0"
        # (std = exp(NaN) fails the >= 0 check). Flag those environments so the
        # regular reset path below re-initialises them this same step.
        state_bad = ~(
            torch.isfinite(self.robot.get_qpos()).all(dim=-1)
            & torch.isfinite(self.robot.get_dofs_velocity()).all(dim=-1)
            & torch.isfinite(self.cube.get_pos()).all(dim=-1)
        )
        if state_bad.any():
            self.reset_buf |= state_bad

        # ── Compute rewards (IsaacGym L1003-1033) ──
        self._compute_rewards()

        # ── Compute termination (IsaacGym L985-1001) ──
        self._compute_termination()

        # Reset done envs
        self._reset_idx(self.reset_buf)

        # ── Compute observations ──
        self._compute_obs()
        obs_td = self.get_observations()

        # Draw one frame (no-op unless ``--vis``).
        self.render()

        return obs_td, self.rew_buf, self.reset_buf, self.extras

    # ────────────────────── visualization ──────────────────────

    def render(self) -> None:
        """Draw one frame of the current simulation state.

        No-op unless the environment was created with ``show_viewer=True``
        (``--vis``). A closed window simply stops the drawing without
        interrupting the simulation/evaluation loop.
        """
        if self.viewer is None:
            return
        if hasattr(self.viewer, "is_running") and not self.viewer.is_running():
            return
        self.viewer.begin_frame(self._render_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()
        self._render_time += self.ctrl_dt

    # ────────────────────── rewards ──────────────────────

    def _compute_rewards(self) -> None:
        """Compute all reward components.

        Phased reward structure designed for effective push learning:

          1. approach_reward: dense EE→cube distance. Uses exp(-d/0.3) with
             wide sigma so there's always gradient even when EE is far.
             Weight: 0.01 (per-step). With ~500 steps/episode, accumulates
             to ~5.0, balanced with success_reward (~1.0) and dist_reward
             (~10.0). Same order of magnitude across the three signals.

          2. push_reward: velocity-based reward for cube moving toward goal.
             Positive when cube-goal distance decreases, negative when it
             increases. This is the KEY signal that teaches pushing.
             Weight: 5.0 (dominant to emphasize pushing). Clamped to [-1,1].

          3. dist_reward: dense cube→goal distance. Uses exp(-d/0.15).
             UNGATED — doesn't depend on EE position.
             Weight: 0.02 (per-step). Accumulates to ~10 per episode,
             matched to success_reward order of magnitude.

          4. success_reward: sparse bonus on task completion.
             Weight: +10 per step while succeeded. Accumulates to 0~10
             depending on fraction of episode spent in success state.

          5. action_penalty: small L2 penalty on actions.
             Weight: -0.005 (per-step). Accumulates to ~-2.5 per episode.

        Per-episode accumulation order of magnitude (target 1~10 each):
          approach_reward ~5    dist_reward ~10    success_reward ~1
        """
        cube_pos = self.cube.get_pos()
        cube_xy = cube_pos[:, :2]
        goal_xy = self.goal_pos_xy
        ee_pos = self._get_ee_pos()

        # 3D distance for approach (EE is 30cm above cube — need Z gradient!)
        ee_obj_dist_3d = torch.linalg.norm(ee_pos - cube_pos, dim=-1)
        cube_goal_dist = torch.linalg.norm(goal_xy - cube_xy, dim=-1)

        # 1. Approach reward: EE → cube in 3D (wider sigma for large Z gap)
        #    Weight 0.01 keeps it in same order-of-magnitude as success/dist.
        approach_rew = torch.exp(-ee_obj_dist_3d / 0.3) * 0.01 - 0.005

        # 2. Push reward: velocity-based, cube moving toward goal
        #    Positive when cube_goal_dist decreases (i.e. cube moves toward goal)
        #    On first step of episode (prev_dist=0), use current dist as baseline
        first_step = (self.episode_length_buf == 1)
        prev_dist = torch.where(first_step, cube_goal_dist, self.prev_cube_goal_dist)
        push_rew = (prev_dist - cube_goal_dist) * 0.10  # positive when approaching
        push_rew = push_rew.clamp(-1.0, 1.0)  # clip to avoid instability

        # 3. Distance reward: cube → goal (ungated, wide sigma)
        #    Weight 0.02 keeps it in same order-of-magnitude as success.
        dist_rew = torch.exp(-cube_goal_dist / 0.15) * 0.02 -0.0045

        # 4. Success reward: sparse bonus
        is_succ = self._is_success()
        self._cached_is_success = is_succ  # Cache for _compute_termination
        success_rew = is_succ * 10.0

        # 5. Action penalty
        action_pen = -0.0004 * torch.sum(self.actions ** 2, dim=-1)

        # Non-finite guard: an environment flagged by the divergence guard above
        # is reset this same step, but its reward is computed from the diverged
        # state first. Replace any NaN/Inf so neither the rollout buffer nor the
        # episode logs ever carry non-finite values into the PPO update.
        approach_rew = torch.nan_to_num(approach_rew, nan=0.0, posinf=0.0, neginf=0.0)
        push_rew = torch.nan_to_num(push_rew, nan=0.0, posinf=1.0, neginf=-1.0)
        dist_rew = torch.nan_to_num(dist_rew, nan=0.0, posinf=0.0, neginf=0.0)

        # Total reward
        self.rew_buf = approach_rew + push_rew + dist_rew + success_rew + action_pen

        # Save current distance for next step's velocity reward
        self.prev_cube_goal_dist = cube_goal_dist.detach().clone()

        # Accumulate for logging
        self.episode_sums["approach_reward"] += approach_rew
        self.episode_sums["push_reward"] += push_rew
        self.episode_sums["dist_reward"] += dist_rew
        self.episode_sums["success_reward"] += success_rew
        self.episode_sums["action_penalty"] += action_pen

        # Update per-episode success flag
        self.success_buf = torch.max(self.success_buf, is_succ)

    # ────────────────────── termination ──────────────────────

    def _get_ee_pos(self) -> torch.Tensor:
        """Get EE position using wrist link (6_Link).
        
        Note: Gripper fingers (7_Link, 8_Link) are mounted below the wrist,
        so their Z coordinates are negative. Using wrist position is more
        appropriate for pushing tasks.
        """
        pos, _ = self.robot.get_link_transforms()
        return pos[:, self._ee_link, :]

    def _get_ee_quat(self) -> torch.Tensor:
        """Get EE quaternion using wrist link, in Genesis (w, x, y, z) order."""
        _, quats = self.robot.get_link_transforms()
        quat_xyzw = quats[:, self._ee_link, :]
        return torch.stack(
            (quat_xyzw[:, 3], quat_xyzw[:, 0], quat_xyzw[:, 1], quat_xyzw[:, 2]),
            dim=-1,
        )

    def _compute_termination(self) -> None:
        """Compute termination conditions.
        
        Termination triggers on:
          1. Timeout (episode_length > max_episode_length)
          2. Object out of bounds (cube XY exits workspace)
          3. Success (cube reaches goal)
        
        Note: EE out-of-bounds termination is intentionally removed.
        The arm should be free to explore without premature termination.
        """
        self.time_out_buf = self.episode_length_buf > self.max_episode_length

        cube_pos = self.cube.get_pos()
        cube_xy = cube_pos[:, :2]
        obj_outbound = (
            torch.any(cube_xy < self.obj_bounds_low, dim=1) |
            torch.any(cube_xy > self.obj_bounds_high, dim=1)
        )

        if self._cached_is_success is not None:
            is_succ = self._cached_is_success.to(torch.bool)
            self._cached_is_success = None  # Clear cache
        else:
            is_succ = self._is_success().to(torch.bool)
        self.last_success = is_succ  # per-env success state (readable after _reset_idx clears success_buf)
        self.reset_buf = self.time_out_buf | obj_outbound | is_succ
        self.extras["time_outs"] = self.time_out_buf.to(dtype=torch.float32)

    # ────────────────────── success ──────────────────────

    def _is_success(self) -> torch.Tensor:
        """Success criterion: distance + velocity + ICP alignment (three conditions).

        Condition 1: object-to-goal XY distance < dist_threshold
        Condition 2: object XY linear velocity < vel_threshold (if enabled)
        Condition 3: ICP alignment error < icp_threshold (if enabled)
        """
        cfg = self.success_config
        cube_xy = self.cube.get_pos()[:, :2]
        obj_goal_dist = torch.linalg.norm(self.goal_pos_xy - cube_xy, dim=-1)
        dist_ok = obj_goal_dist < cfg.dist_threshold

        if cfg.vel_enabled:
            cube_vel = self.cube.get_vel()[:, :2]  # XY线速度
            cube_speed = torch.linalg.norm(cube_vel, dim=-1)
            vel_ok = cube_speed < cfg.vel_threshold
        else:
            vel_ok = torch.ones_like(dist_ok)

        if cfg.icp_enabled and self.pc_config.enabled:
            # Transform FPS-downsampled local pcd to current world frame
            cube_pos = self.cube.get_pos()
            cube_yaw = self._get_cube_yaw()
            current_world_pcd = transform_pcd_by_pose(
                self._fps_pcd, cube_pos[:, :2], cube_yaw, cube_pos[:, 2]
            )
            icp_error = self.pc_processor.compute_icp_error_downsampled(
                current_world_pcd, self._fps_goal_pcd
            )
            icp_ok = icp_error < cfg.icp_threshold
        else:
            icp_ok = torch.ones_like(dist_ok)

        return (dist_ok & vel_ok & icp_ok).to(torch.float32)

    def _get_cube_yaw(self) -> torch.Tensor:
        """Extract the Z-axis yaw angle from the object quaternion.

        Newton reports quaternions as (x, y, z, w); convert to (w, x, y, z)
        before applying the yaw formula.
        """
        quat_xyzw = self.cube.get_quat()  # [n_envs, 4] (x, y, z, w)
        w, x, y, z = quat_xyzw[:, 3], quat_xyzw[:, 0], quat_xyzw[:, 1], quat_xyzw[:, 2]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return yaw

    # ────────────────────── observations ──────────────────────

    def _compute_obs(self) -> None:
        """Observation vector (dim = 32), all in environment-local coordinates.

        Genesis guarantees: get_pos() returns local coordinates (excludes envs_offset).
        All positions are relative to each environment's own origin (robot base).
        This ensures the RL policy is invariant to environment placement on the grid.

        Observation breakdown:
          [0:3]   EE position (local XYZ)
          [3:7]   EE quaternion (local wxyz)
          [7:10]  Cube position (local XYZ)
          [10:12] Goal position (local XY)
          [12:15] EE→Cube vector (local XYZ)
          [15:17] Cube→Goal vector (local XY)
          [17:23] Arm joint positions (6)
          [23:29] Arm joint velocities (6)
          [29:32] Point cloud centroid (local XYZ)
        """
        ee_pos = self._get_ee_pos()       # local coords
        ee_quat = self._get_ee_quat()     # local coords
        cube_pos = self.cube.get_pos()    # local coords
        ee_to_cube = cube_pos - ee_pos
        cube_to_goal = self.goal_pos_xy - cube_pos[:, :2]

        # Joint states
        qpos = self.robot.get_qpos()[:, :self.ARM_DOF]
        qvel = self.robot.get_dofs_velocity()[:, :self.ARM_DOF]

        # Point cloud centroid
        if self.pc_config.enabled:
            centroid = PointCloudProcessor.compute_centroid(self._current_pcd)  # [n_envs, 3]
        else:
            centroid = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)

        # RL observation vector
        self.obs_buf = torch.cat([
            ee_pos,            # 3  — local
            ee_quat,           # 4  — local
            cube_pos,          # 3  — local
            self.goal_pos_xy,  # 2  — local (sampled in local ranges)
            ee_to_cube,        # 3  — local
            cube_to_goal,      # 2  — local
            qpos,              # 6  — joint positions
            qvel,              # 6  — joint velocities
            centroid,          # 3  — point cloud centroid
        ], dim=-1)

        # Final safety net: never let a non-finite value reach the policy
        # (one NaN observation would make the PPO loss and gradients NaN,
        # corrupting the network weights — including the noise log-std).
        self.obs_buf = torch.nan_to_num(self.obs_buf, nan=0.0, posinf=0.0, neginf=0.0)

    # ────────────────────── rsl_rl 4.x interface ──────────────────────

    def get_observations(self) -> TensorDict:
        """Return observations as TensorDict expected by rsl_rl 4.x."""
        return TensorDict(
            {"policy": self.obs_buf},
            batch_size=[self.num_envs],
            device=self.device,
        )

    def close(self) -> None:
        """Clean up simulation resources."""
        # Newton/Warp have no explicit teardown for a finalized Model; the
        # tensors below are released when the environment is garbage collected.
        self.robot = None
        self.cube = None
        self.goal_entity = None
        self.model = None
        self.solver = None
        self.pipeline = None

    # ────────────────────── randomization query ──────────────────────

    def get_object_randomization_params(self) -> dict:
        """Query per-env object randomization parameters.

        Returns:
            dict with keys:
                sizes: List[Tuple[float,float,float]] — per-env object sizes
                masses: torch.Tensor or None — per-env mass (n_envs,)
                frictions: torch.Tensor or None — per-env friction coefficient (n_envs,)
                randomizer: ObjectRandomizer — the randomization strategy config
        """
        return {
            "sizes": self._sampled_sizes,
            "masses": self._sampled_masses,
            "frictions": self._sampled_frictions,
            "randomizer": self.obj_randomizer,
        }

    # ────────────────────── point cloud query ──────────────────────

    def get_point_cloud(self) -> torch.Tensor:
        """Get current point cloud for all environments.

        Returns:
            Point cloud [n_envs, n_points, 3] in object-local frame (centered at origin)
        """
        return self._current_pcd.clone()

    def get_goal_point_cloud(self) -> torch.Tensor:
        """Get cached goal point cloud (world frame, generated at reset).

        Returns:
            Goal point cloud [n_envs, n_points, 3]
        """
        if not self.pc_config.enabled:
            return torch.zeros_like(self._current_pcd)
        return self._goal_pcd.clone()

    # ────────────────────── coordinate conversion (monitoring) ──────────────────────

    def local_to_world(self, local_pos: torch.Tensor) -> torch.Tensor:
        """Convert local coordinates to world coordinates (for monitoring).
        
        Args:
            local_pos: (..., 3) tensor in environment-local frame.
        
        Returns:
            (..., 3) tensor in world frame (adds env grid offset).
        """
        return local_pos + self.env_origins

    def world_to_local(self, world_pos: torch.Tensor) -> torch.Tensor:
        """Convert world coordinates to local coordinates.
        
        Args:
            world_pos: (..., 3) tensor in world frame.
        
        Returns:
            (..., 3) tensor in environment-local frame.
        """
        return world_pos - self.env_origins
