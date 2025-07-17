# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the 14-DOFs Simon Half-body Humanoid robot."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


##
# Configuration
##

simon_half_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="D:\\Isaac\\Simon\\models\\humanoid_28\\simon_half.usda",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            "abdomen_x": 0.0,
            "abdomen_y": 0.0,
            "abdomen_z": 0.0,
            "right_hip_x": 0.0,
            "right_hip_y": 0.0,
            "right_hip_z": 0.1,
            "right_knee": 0.2,
            "right_ankle_x": 0.0,
            "right_ankle_y": -0.1,
            "right_ankle_z": 0.0,
            "left_hip_x": 0.0,
            "left_hip_y": 0.0,
            "left_hip_z": -0.1,
            "left_knee": 0.2,
            "left_ankle_x": 0.0,
            "left_ankle_y": -0.1,
            "left_ankle_z": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*hip.*", ".*knee", ".*ankle.*"],
            effort_limit=300.0,
            velocity_limit=10.0,
            stiffness={
                ".*hip.*": 150.0,
                ".*knee": 200.0,
                ".*ankle.*": 100.0,
            },
            damping={
                ".*hip.*": 5.0,
                ".*knee": 6.0,
                ".*ankle.*": 4.0,
            },
        ),
        "torso": ImplicitActuatorCfg(
            joint_names_expr=["abdomen.*"],
            effort_limit=100.0,
            velocity_limit=10.0,
            stiffness={
                "abdomen.*": 80.0,
            },
            damping={
                "abdomen.*": 4.0,
            },
        ),
    },
)
