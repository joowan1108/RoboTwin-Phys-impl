# Method excerpts from NVIDIA cuRobo, distributed under the NVIDIA License.
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Revision d64c4b005459db10c5dd867d8b30a87d5bda9bdb. See CUROBO_LICENSE.
# Fixture modification: CPU stand-ins for planner inputs and results.
from __future__ import annotations
from types import SimpleNamespace
from dataclasses import dataclass
import time
from torch.autograd import profiler
import torch


class InterpolateType:
    LINEAR_CUDA = "LINEAR_CUDA"


@dataclass
class MotionGenResult:
    """Result obtained from motion generation."""

    #: success tensor with index referring to the batch index.
    success: Optional[torch.Tensor] = None

    #: returns true if the start state is not satisfying constraints (e.g., in collision)
    valid_query: bool = True

    #: optimized trajectory
    optimized_plan: Optional[JointState] = None

    #: dt between steps in the optimized plan
    optimized_dt: Optional[T_BValue_float] = None

    #: Cartesian position error at final timestep, returning l2 distance.
    position_error: Optional[T_BValue_float] = None

    #: Cartesian rotation error at final timestep, computed as q^(-1) * q_g
    rotation_error: Optional[T_BValue_float] = None

    #: Error in joint configuration, when planning in joint space, returning l2 distance.
    cspace_error: Optional[T_BValue_float] = None

    #: seconds taken  in the optimizer for solving the motion generation problem.
    solve_time: float = 0.0

    #: seconds taken to solve IK.
    ik_time: float = 0.0

    #: seconds taken to find graph plan.
    graph_time: float = 0.0

    #: seconds taken in trajectory optimization.
    trajopt_time: float = 0.0

    #: seconds to run finetune trajectory optimization.
    finetune_time: float = 0.0

    #: sum of ik_time, graph_time, and trajopt_time. This also includes any processing between
    #: calling the different modules.
    total_time: float = 0.0

    #: interpolated solution, useful for visualization.
    interpolated_plan: Optional[JointState] = None

    #: dt between steps in interpolated plan
    interpolation_dt: float = 0.02

    #: last timestep in interpolated plan per batch index. Only used for batched queries
    path_buffer_last_tstep: Optional[List[int]] = None

    #: Debug information
    debug_info: Any = None

    #: status of motion generation query.
    status: Optional[Union[MotionGenStatus, str]] = None

    #: number of attempts used before returning a solution.
    attempts: int = 1

    #: number of trajectory optimization attempts used per attempt.
    trajopt_attempts: int = 0

    #: returns true when a graph plan was used to seed trajectory optimization.
    used_graph: bool = False

    #: stores graph plan.
    graph_plan: Optional[JointState] = None

    #: stores the index of the goal pose reached when planning for a goalset.
    goalset_index: Optional[torch.Tensor] = None

    def clone(self):
        """Clone the current result."""
        m = MotionGenResult(
            self.success.clone(),
            valid_query=self.valid_query,
            optimized_plan=self.optimized_plan.clone() if self.optimized_plan is not None else None,
            optimized_dt=self.optimized_dt.clone() if self.optimized_dt is not None else None,
            position_error=self.position_error.clone() if self.position_error is not None else None,
            rotation_error=self.rotation_error.clone() if self.rotation_error is not None else None,
            cspace_error=self.cspace_error.clone() if self.cspace_error is not None else None,
            solve_time=self.solve_time,
            ik_time=self.ik_time,
            graph_time=self.graph_time,
            trajopt_time=self.trajopt_time,
            total_time=self.total_time,
            graph_plan=self.graph_plan.clone() if self.graph_plan is not None else None,
            debug_info=self.debug_info,
            status=self.status,
            attempts=self.attempts,
            trajopt_attempts=self.trajopt_attempts,
            used_graph=self.used_graph,
            path_buffer_last_tstep=self.path_buffer_last_tstep,
            interpolated_plan=(
                self.interpolated_plan.clone() if self.interpolated_plan is not None else None
            ),
            interpolation_dt=self.interpolation_dt,
            goalset_index=self.goalset_index.clone() if self.goalset_index is not None else None,
        )
        return m

    def copy_idx(self, idx: torch.Tensor, source_result: MotionGenResult):
        """Copy data from source result to current result at index.

        Args:
            idx: index to copy data at.
            source_result: source result to copy data from.

        Returns:
            MotionGenResult: copied result.
        """
        idx = idx.to(dtype=torch.long)
        # copy data from source result:
        self.success[idx] = source_result.success[idx]

        self.optimized_plan = self._check_none_and_copy_idx(
            self.optimized_plan, source_result.optimized_plan, idx
        )
        self.interpolated_plan = self._check_none_and_copy_idx(
            self.interpolated_plan, source_result.interpolated_plan, idx
        )

        self.position_error = self._check_none_and_copy_idx(
            self.position_error, source_result.position_error, idx
        )

        self.rotation_error = self._check_none_and_copy_idx(
            self.rotation_error, source_result.rotation_error, idx
        )
        self.cspace_error = self._check_none_and_copy_idx(
            self.cspace_error, source_result.cspace_error, idx
        )

        self.goalset_index = self._check_none_and_copy_idx(
            self.goalset_index, source_result.goalset_index, idx
        )
        # NOTE: graph plan will have different shape based on success.
        # self.graph_plan = self._check_none_and_copy_idx(
        #    self.graph_plan, source_result.graph_plan, idx
        # )

        idx_list = idx.cpu().tolist()
        if source_result.path_buffer_last_tstep is not None:
            if self.path_buffer_last_tstep is None:
                self.path_buffer_last_tstep = [0 for i in range(len(self.success))]

            for i in idx_list:
                self.path_buffer_last_tstep[i] = source_result.path_buffer_last_tstep[i]

        return self

    def get_paths(self) -> List[JointState]:
        """Get interpolated trajectories from the result. Use for batched queries.

        This will return unsuccessful trajectories as well. Use
        :meth:`MotionGenResult.get_successful_paths` to get only successful trajectories.

        Returns:
            List[JointState]: Interpolated trajectories. Check
                :attr:`MotionGenResult.interpolation_dt` for the time between steps.
        """
        path = [
            self.interpolated_plan[x].trim_trajectory(0, self.path_buffer_last_tstep[x])
            for x in range(len(self.interpolated_plan))
        ]
        return path

    def get_successful_paths(self) -> List[torch.Tensor]:
        """Get successful interpolated trajectories from the result. Use for batched queries.

        Returns:
            List[JointState]: Interpolated trajectories. Check
                :attr:`MotionGenResult.interpolation_dt` for the time between steps.
        """
        path = []
        nz_i = torch.nonzero(self.success.view(-1)).view(-1)
        path = self.interpolated_plan[nz_i]
        path_list = []

        if self.path_buffer_last_tstep is not None:
            for i in range(len(path)):
                last_tstep = self.path_buffer_last_tstep[nz_i[i]]
                path_list.append(path[i].trim_trajectory(0, last_tstep))
        else:
            path_list = [path[i, :, :] for i in range(path.shape[0])]

        return path_list

    def get_interpolated_plan(self) -> JointState:
        """Get interpolated trajectory from the result.

        Returns:
            JointState: Interpolated trajectory. Check :attr:`MotionGenResult.interpolation_dt` for
            the time between steps.
        """
        if self.path_buffer_last_tstep is None:
            return self.interpolated_plan
        if len(self.path_buffer_last_tstep) > 1:
            log_error("only single result is supported")

        return self.interpolated_plan.trim_trajectory(0, self.path_buffer_last_tstep[0])

    def retime_trajectory(
        self,
        time_dilation_factor: float,
        interpolate_trajectory: bool = True,
        interpolation_dt: Optional[float] = None,
        interpolation_kind: InterpolateType = InterpolateType.LINEAR_CUDA,
        create_interpolation_buffer: bool = True,
    ):
        """Retime the optimized trajectory by a dilation factor.

        Args:
            time_dilation_factor: Time factor to slow down the trajectory. Should be less than 1.0.
            interpolate_trajectory: Interpolate the trajectory after retiming.
            interpolation_dt: Time between steps in the interpolated trajectory. If None,
                :attr:`MotionGenResult.interpolation_dt` is used.
            interpolation_kind: Interpolation type to use.
            create_interpolation_buffer: Create a new buffer for interpolated trajectory. Set this
                to True if existing buffer is not large enough to store new trajectory.
        """

        if time_dilation_factor > 1.0:
            log_error("time_dilation_factor should be less than 1.0")
        if time_dilation_factor == 1.0:
            return
        if len(self.path_buffer_last_tstep) > 1:
            log_error("only single result is supported")

        new_dt = self.optimized_dt * (1.0 / time_dilation_factor)
        if len(self.optimized_plan.shape) == 3:
            new_dt = new_dt.view(-1, 1, 1)
        else:
            new_dt = new_dt.view(-1, 1)
        self.optimized_plan = self.optimized_plan.scale_by_dt(self.optimized_dt, new_dt)
        self.optimized_dt = new_dt.view(-1)
        if interpolate_trajectory:
            if interpolation_dt is not None:
                self.interpolation_dt = interpolation_dt
            self.interpolated_plan, last_tstep, _ = get_batch_interpolated_trajectory(
                self.optimized_plan,
                self.optimized_dt,
                self.interpolation_dt,
                kind=interpolation_kind,
                out_traj_state=self.interpolated_plan if not create_interpolation_buffer else None,
                tensor_args=self.interpolated_plan.tensor_args,
                optimize_dt=False,
            )
            self.path_buffer_last_tstep = [last_tstep[i] for i in range(len(last_tstep))]
            if len(self.optimized_plan.shape) == 2:
                self.interpolated_plan = self.interpolated_plan.squeeze(0)

    @property
    def motion_time(self) -> Union[float, torch.Tensor]:
        """Compute motion time in seconds."""

        # -2 as last three timesteps have the same value
        # 0, 1 also have the same position value.
        return self.optimized_dt * (self.optimized_plan.position.shape[-2] - 1 - 2 - 1)

    @staticmethod
    def _check_none_and_copy_idx(
        current_tensor: Union[torch.Tensor, JointState, None],
        source_tensor: Union[torch.Tensor, JointState, None],
        idx: int,
    ) -> Union[torch.Tensor, JointState]:
        """Helper function to copy data from source tensor to current tensor at index.

        Also supports copying from JointState to JointState.

        Args:
            current_tensor: tensor to copy data to.
            source_tensor: tensor to copy data from.
            idx: index to copy data at.

        Returns:
            Union[torch.Tensor, JointState]: copied tensor.
        """
        if source_tensor is not None:
            if current_tensor is None:
                current_tensor = source_tensor.clone()
            else:
                if isinstance(current_tensor, torch.Tensor) and isinstance(
                    source_tensor, torch.Tensor
                ):
                    current_tensor[idx] = source_tensor[idx]
                elif isinstance(current_tensor, JointState) and isinstance(
                    source_tensor, JointState
                ):
                    source_state = source_tensor[idx]
                    current_tensor.copy_at_index(source_state, idx)

        return current_tensor


class MotionGenPlanConfig:
    pass


class MotionGenStatus:
    IK_FAIL = "IK_FAIL"
    GRAPH_FAIL = "GRAPH_FAIL"
    TRAJOPT_FAIL = "TRAJOPT_FAIL"
    INVALID_QUERY = "INVALID_QUERY"


class Goal:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class JointState:
    @staticmethod
    def from_position(position):
        return SimpleNamespace(position=position)


class MotionGen:
    def _plan_from_solve_state(
        self,
        solve_state: ReacherSolveState,
        start_state: JointState,
        goal_pose: Pose,
        plan_config: MotionGenPlanConfig = MotionGenPlanConfig(),
        link_poses: Optional[Dict[str, Pose]] = None,
    ) -> MotionGenResult:
        """Plan from a given reacher solve state.

        Args:
            solve_state: Reacher solve state for planning.
            start_state: Start joint state for planning.
            goal_pose: Goal pose to reach for end-effector.
            plan_config: Planning parameters for motion generation.
            link_poses: Goal poses for other links in the robot.

        Returns:
            MotionGenResult: Result of planning.
        """
        trajopt_seed_traj = None
        trajopt_seed_success = None
        trajopt_newton_iters = None
        graph_success = 0

        if len(start_state.shape) == 1:
            log_error("Joint state should be not a vector (dof) should be (bxdof)")

        if goal_pose.shape[0] != 1:
            log_error(
                "Goal position should be of shape [1, n_goalset, -1], current shape: "
                + str(goal_pose.shape)
            )
        # plan ik:

        ik_result = self._solve_ik_from_solve_state(
            goal_pose,
            solve_state,
            start_state,
            plan_config.use_nn_ik_seed,
            plan_config.partial_ik_opt,
            link_poses,
        )

        if not plan_config.enable_graph and plan_config.partial_ik_opt:
            ik_result.success[:] = True

        # check for success:
        result = MotionGenResult(
            ik_result.success.view(-1)[0:1],  # This is not true for batch mode
            ik_time=ik_result.solve_time,
            solve_time=ik_result.solve_time,
        )

        if self.store_debug_in_result:
            result.debug_info = {"ik_result": ik_result}
        ik_success = torch.count_nonzero(ik_result.success)
        if ik_success == 0:
            result.status = MotionGenStatus.IK_FAIL
            return result

        # do graph search:
        with profiler.record_function("motion_gen/post_ik"):
            ik_out_seeds = solve_state.num_trajopt_seeds
            if plan_config.enable_graph:
                ik_out_seeds = min(solve_state.num_trajopt_seeds, ik_success)

            goal_config = ik_result.solution[ik_result.success].view(-1, self.ik_solver.dof)[
                :ik_success
            ]
            start_config = tensor_repeat_seeds(start_state.position, ik_out_seeds)
            if plan_config.enable_opt:
                self._trajopt_goal_config[:] = ik_result.solution

        # do graph search:
        if plan_config.enable_graph:
            interpolation_steps = None
            if plan_config.enable_opt:
                interpolation_steps = self.trajopt_solver.action_horizon
            log_info("MG: running GP")
            graph_result = self.graph_search(start_config, goal_config, interpolation_steps)
            trajopt_seed_success = graph_result.success

            graph_success = torch.count_nonzero(graph_result.success).item()
            result.graph_time = graph_result.solve_time
            result.solve_time += graph_result.solve_time
            if graph_success > 0:
                log_info("MG: GP Success")
                result.graph_plan = graph_result.interpolated_plan
                result.interpolated_plan = graph_result.interpolated_plan

                result.used_graph = True
                if plan_config.enable_opt:
                    trajopt_seed = (
                        result.graph_plan.position.view(
                            1,  # solve_state.batch_size,
                            graph_success,  # solve_state.num_trajopt_seeds,
                            interpolation_steps,
                            self._dof,
                        )
                        .transpose(0, 1)
                        .contiguous()
                    )
                    trajopt_seed_traj = torch.zeros(
                        (trajopt_seed.shape[0], 1, self.trajopt_solver.action_horizon, self._dof),
                        device=self.tensor_args.device,
                        dtype=self.tensor_args.dtype,
                    )
                    trajopt_seed_traj[:, :, :interpolation_steps, :] = trajopt_seed
                    trajopt_seed_success = ik_result.success.clone()
                    trajopt_seed_success[ik_result.success] = graph_result.success

                    trajopt_seed_success = trajopt_seed_success.view(
                        solve_state.batch_size, solve_state.num_trajopt_seeds
                    )
                    trajopt_newton_iters = self.graph_trajopt_iters
                else:
                    _, idx = torch.topk(
                        graph_result.path_length[graph_result.success], k=1, largest=False
                    )
                    result.interpolated_plan = result.interpolated_plan[idx].squeeze(0)
                    result.optimized_dt = self.tensor_args.to_device(self.interpolation_dt)
                    result.optimized_plan = result.interpolated_plan[
                        : graph_result.path_buffer_last_tstep[idx.item()]
                    ]
                    idx = idx.view(-1) + self._batch_col
                    result.position_error = ik_result.position_error[ik_result.success][
                        graph_result.success
                    ][idx]
                    result.rotation_error = ik_result.rotation_error[ik_result.success][
                        graph_result.success
                    ][idx]
                    result.path_buffer_last_tstep = graph_result.path_buffer_last_tstep[
                        idx.item() : idx.item() + 1
                    ]
                    result.success = result.success.view(-1)[0:1]
                    result.success[:] = True
                    return result
            else:
                result.success[:] = False
                result.status = MotionGenStatus.GRAPH_FAIL
                if not graph_result.valid_query:
                    result.valid_query = False
                    if self.store_debug_in_result:
                        result.debug_info["graph_debug"] = graph_result.debug_info
                    return result
                if plan_config.need_graph_success:
                    return result

        # do trajopt::

        if plan_config.enable_opt:
            with profiler.record_function("motion_gen/setup_trajopt_seeds"):
                self._trajopt_goal_config[:, :ik_success] = goal_config
                retract_config = None
                if plan_config.use_start_state_as_retract:
                    retract_config = start_state.position.clone()
                goal = Goal(
                    goal_pose=goal_pose,
                    current_state=start_state,
                    links_goal_pose=link_poses,
                    retract_state=retract_config,
                )

                if (
                    trajopt_seed_traj is None
                    or graph_success < solve_state.num_trajopt_seeds * self.noisy_trajopt_seeds
                ):
                    goal_config = self._trajopt_goal_config[0]  # batch index == 0

                    goal_state = JointState.from_position(
                        goal_config,
                    )
                    seed_link_poses = None
                    if link_poses is not None:
                        seed_link_poses = {}

                        for k in link_poses.keys():
                            seed_link_poses[k] = link_poses[k].repeat_seeds(
                                solve_state.num_trajopt_seeds
                            )
                    if goal_pose.shape[0] != 1:
                        log_error(
                            "Batch size of goal pose should be 1, current shape: "
                            + str(goal_pose.shape)
                        )
                    seed_goal = Goal(
                        goal_pose=goal_pose.repeat_seeds(solve_state.num_trajopt_seeds),
                        current_state=start_state.repeat_seeds(solve_state.num_trajopt_seeds),
                        goal_state=goal_state,
                        links_goal_pose=seed_link_poses,
                    )
                    if trajopt_seed_traj is not None:
                        trajopt_seed_traj = trajopt_seed_traj.transpose(0, 1).contiguous()
                        # batch, num_seeds, h, dof
                        if (
                            trajopt_seed_success.shape[1]
                            < solve_state.num_trajopt_seeds * self.noisy_trajopt_seeds
                        ):
                            trajopt_seed_success_new = torch.zeros(
                                (1, solve_state.num_trajopt_seeds * self.noisy_trajopt_seeds),
                                device=self.tensor_args.device,
                                dtype=torch.bool,
                            )
                            trajopt_seed_success_new[0, : trajopt_seed_success.shape[1]] = (
                                trajopt_seed_success
                            )
                            trajopt_seed_success = trajopt_seed_success_new
                    # create seeds here:
                    trajopt_seed_traj = self.trajopt_solver.get_seed_set(
                        seed_goal,
                        trajopt_seed_traj,  # batch, num_seeds, h, dof
                        num_seeds=self.noisy_trajopt_seeds,
                        batch_mode=solve_state.batch_mode,
                        seed_success=trajopt_seed_success,
                    )
                    trajopt_seed_traj = trajopt_seed_traj.view(
                        solve_state.num_trajopt_seeds * self.noisy_trajopt_seeds,
                        solve_state.batch_size,
                        self.trajopt_solver.action_horizon,
                        self._dof,
                    ).contiguous()
            if plan_config.enable_finetune_trajopt:
                og_value = self.trajopt_solver.interpolation_type
                self.trajopt_solver.interpolation_type = InterpolateType.LINEAR_CUDA
            with profiler.record_function("motion_gen/trajopt"):
                log_info("MG: running TO")
                traj_result = self._solve_trajopt_from_solve_state(
                    goal,
                    solve_state,
                    trajopt_seed_traj,
                    num_seeds_override=solve_state.num_trajopt_seeds * self.noisy_trajopt_seeds,
                    newton_iters=trajopt_newton_iters,
                    return_all_solutions=plan_config.parallel_finetune
                    and plan_config.enable_finetune_trajopt,
                )
            if plan_config.enable_finetune_trajopt:
                self.trajopt_solver.interpolation_type = og_value
            if self.store_debug_in_result:
                result.debug_info["trajopt_result"] = traj_result
            # run finetune
            if plan_config.enable_finetune_trajopt and torch.count_nonzero(traj_result.success) > 0:
                with profiler.record_function("motion_gen/finetune_trajopt"):
                    seed_traj = traj_result.raw_action.clone()  # solution.position.clone()
                    seed_traj = seed_traj.contiguous()
                    og_solve_time = traj_result.solve_time
                    seed_override = 1
                    opt_dt = traj_result.optimized_dt

                    if plan_config.parallel_finetune:
                        seed_override = solve_state.num_trajopt_seeds * self.noisy_trajopt_seeds
                        if self.optimize_dt:
                            opt_dt = torch.min(opt_dt[traj_result.success])

                    finetune_time = 0
                    newton_iters = None

                    for k in range(plan_config.finetune_attempts):
                        if self.optimize_dt:

                            scaled_dt = torch.clamp(
                                opt_dt
                                * plan_config.finetune_dt_scale
                                * (plan_config.finetune_dt_decay ** (k)),
                                self.trajopt_solver.minimum_trajectory_dt,
                            )
                            self.finetune_trajopt_solver.update_solver_dt(scaled_dt.item())

                        traj_result = self._solve_trajopt_from_solve_state(
                            goal,
                            solve_state,
                            seed_traj,
                            trajopt_instance=self.finetune_trajopt_solver,
                            num_seeds_override=seed_override,
                            newton_iters=newton_iters,
                        )
                        finetune_time += traj_result.solve_time
                        if torch.count_nonzero(traj_result.success) > 0 or not self.optimize_dt:
                            break
                        seed_traj = traj_result.optimized_seeds.detach().clone()
                        newton_iters = 4

                    traj_result.solve_time = finetune_time

                result.finetune_time = traj_result.solve_time

                traj_result.solve_time = og_solve_time
                if self.store_debug_in_result:
                    result.debug_info["finetune_trajopt_result"] = traj_result
            elif plan_config.enable_finetune_trajopt:
                traj_result.success = traj_result.success[0:1]
                # if torch.count_nonzero(result.success) == 0:
                result.status = MotionGenStatus.TRAJOPT_FAIL
            result.solve_time += traj_result.solve_time + result.finetune_time
            result.trajopt_time = traj_result.solve_time
            result.trajopt_attempts = 1
            result.success = traj_result.success

            if plan_config.enable_finetune_trajopt and torch.count_nonzero(result.success) == 0:

                result.status = MotionGenStatus.FINETUNE_TRAJOPT_FAIL
                if (
                    traj_result.debug_info is not None
                    and "dt_exception" in traj_result.debug_info
                    and traj_result.debug_info["dt_exception"]
                ):
                    result.status = MotionGenStatus.DT_EXCEPTION

            result.interpolated_plan = traj_result.interpolated_solution.trim_trajectory(
                0, traj_result.path_buffer_last_tstep[0]
            )
            result.interpolation_dt = self.trajopt_solver.interpolation_dt
            result.path_buffer_last_tstep = traj_result.path_buffer_last_tstep
            result.position_error = traj_result.position_error
            result.rotation_error = traj_result.rotation_error
            result.optimized_dt = traj_result.optimized_dt
            result.optimized_plan = traj_result.solution
            result.goalset_index = traj_result.goalset_index
        return result

    def _plan_from_solve_state_batch(
        self,
        solve_state: ReacherSolveState,
        start_state: JointState,
        goal_pose: Pose,
        plan_config: MotionGenPlanConfig = MotionGenPlanConfig(),
        link_poses: Optional[Dict[str, Pose]] = None,
    ) -> MotionGenResult:
        """Plan from a given reacher solve state in batch mode.

        Args:
            solve_state: Reacher solve state for planning.
            start_state: Start joint state for planning.
            goal_pose: Goal poses to reach for end-effector.
            plan_config: Planning parameters for motion generation.
            link_poses: Goal poses for other links in the robot.

        Returns:
            MotionGenResult: Result of planning.
        """
        self._trajopt_goal_config[:] = self.get_retract_config().view(1, 1, self._dof)
        trajopt_seed_traj = None
        trajopt_seed_success = None
        trajopt_newton_iters = None
        graph_success = 0

        # plan ik:
        ik_result = self._solve_ik_from_solve_state(
            goal_pose,
            solve_state,
            start_state,
            plan_config.use_nn_ik_seed,
            plan_config.partial_ik_opt,
            link_poses,
        )

        if not plan_config.enable_graph and plan_config.partial_ik_opt:
            ik_result.success[:] = True

        # check for success:
        result = MotionGenResult(
            ik_result.success,
            position_error=ik_result.position_error,
            rotation_error=ik_result.rotation_error,
            ik_time=ik_result.solve_time,
            solve_time=ik_result.solve_time,
            debug_info={},
            # goalset_index=ik_result.goalset_index,
        )

        ik_success = torch.count_nonzero(ik_result.success)
        if ik_success == 0:
            result.status = MotionGenStatus.IK_FAIL
            result.success = result.success[:, 0]
            return result

        # do graph search:
        ik_out_seeds = solve_state.num_trajopt_seeds
        if plan_config.enable_graph:
            ik_out_seeds = min(solve_state.num_trajopt_seeds, ik_success)

        # if not plan_config.enable_opt and plan_config.enable_graph:
        #    self.graph_planner.interpolation_steps = self.interpolation_steps
        #    self.graph_planner.interpolation_type = self.interpolation_type
        # elif plan_config.enable_graph:
        #    self.graph_planner.interpolation_steps = self.trajopt_solver.traj_tsteps
        #    self.graph_planner.interpolation_type = InterpolateType.LINEAR
        goal_config = ik_result.solution[ik_result.success].view(-1, self.ik_solver.dof)

        # get shortest path
        if plan_config.enable_graph:
            interpolation_steps = None
            if plan_config.enable_opt:
                interpolation_steps = self.trajopt_solver.action_horizon

            start_graph_state = start_state.repeat_seeds(ik_out_seeds)
            start_config = start_graph_state.position[ik_result.success.view(-1)].view(
                -1, self.ik_solver.dof
            )
            graph_result = self.graph_search(start_config, goal_config, interpolation_steps)
            graph_success = torch.count_nonzero(graph_result.success).item()

            result.graph_time = graph_result.solve_time
            result.solve_time += graph_result.solve_time
            if graph_success > 0:
                # path = graph_result.interpolated_plan
                result.graph_plan = graph_result.interpolated_plan
                result.interpolated_plan = graph_result.interpolated_plan
                result.used_graph = True

                if plan_config.enable_opt:
                    trajopt_seed = result.graph_plan.position.view(
                        graph_success,  # solve_state.num_trajopt_seeds,
                        interpolation_steps,
                        self._dof,
                    ).contiguous()
                    trajopt_seed_traj = torch.zeros(
                        (1, trajopt_seed.shape[0], self.trajopt_solver.action_horizon, self._dof),
                        device=self.tensor_args.device,
                        dtype=self.tensor_args.dtype,
                    )
                    trajopt_seed_traj[0, :, :interpolation_steps, :] = trajopt_seed
                    trajopt_seed_traj = trajopt_seed_traj.transpose(0, 1).contiguous()
                    trajopt_seed_success = ik_result.success.clone()
                    trajopt_seed_success[ik_result.success] = graph_result.success

                    trajopt_seed_success = trajopt_seed_success.view(
                        solve_state.num_trajopt_seeds, solve_state.batch_size
                    )
                    trajopt_newton_iters = self.graph_trajopt_iters

                else:
                    ik_success = ik_result.success.view(-1).clone()

                    # only some might be successful:

                    g_dim = torch.nonzero(ik_success).view(-1)[graph_result.success]

                    self._batch_graph_search_buffer.copy_at_index(
                        graph_result.interpolated_plan, g_dim
                    )

                    # result.graph_plan = JointState.from_position(
                    #    self._batch_graph_search_buffer,
                    #    joint_names=graph_result.interpolated_plan.joint_names,
                    # )
                    result.interpolated_plan = self._batch_graph_search_buffer
                    g_dim = g_dim.cpu().squeeze().tolist()
                    if isinstance(g_dim, int):
                        g_dim = [g_dim]
                    for x, x_val in enumerate(g_dim):
                        self._batch_path_buffer_last_tstep[x_val] = (
                            graph_result.path_buffer_last_tstep[x]
                        )
                    result.path_buffer_last_tstep = self._batch_path_buffer_last_tstep
                    result.optimized_plan = result.interpolated_plan
                    result.optimized_dt = torch.as_tensor(
                        [
                            self.interpolation_dt
                            for i in range(result.interpolated_plan.position.shape[0])
                        ],
                        device=self.tensor_args.device,
                        dtype=self.tensor_args.dtype,
                    )
                    result.success = result.success.view(-1).clone()
                    result.success[ik_success][graph_result.success] = True
                    return result

            else:
                result.success[:] = False
                result.success = result.success[:, 0]
                result.status = MotionGenStatus.GRAPH_FAIL
                if not graph_result.valid_query:
                    result.valid_query = False
                    if self.store_debug_in_result:
                        result.debug_info = {"graph_debug": graph_result.debug_info}
                    return result

        if plan_config.enable_opt:
            # get goal configs based on ik success:
            self._trajopt_goal_config[ik_result.success] = goal_config

            goal_config = self._trajopt_goal_config  # batch index == 0

            goal = Goal(
                goal_pose=goal_pose,
                current_state=start_state,
                links_goal_pose=link_poses,
            )
            # generate seeds:
            if trajopt_seed_traj is None or (
                plan_config.enable_graph and graph_success < solve_state.batch_size
            ):
                seed_link_poses = None
                if link_poses is not None:
                    seed_link_poses = {}

                    for k in link_poses.keys():
                        seed_link_poses[k] = link_poses[k].repeat_seeds(
                            solve_state.num_trajopt_seeds
                        )
                seed_goal = Goal(
                    goal_pose=goal_pose.repeat_seeds(solve_state.num_trajopt_seeds),
                    current_state=start_state.repeat_seeds(solve_state.num_trajopt_seeds),
                    goal_state=JointState.from_position(goal_config.view(-1, self._dof)),
                    links_goal_pose=seed_link_poses,
                )
                if trajopt_seed_traj is not None:
                    trajopt_seed_traj = trajopt_seed_traj.transpose(0, 1).contiguous()

                # create seeds here:
                trajopt_seed_traj = self.trajopt_solver.get_seed_set(
                    seed_goal,
                    trajopt_seed_traj,  # batch, num_seeds, h, dof
                    num_seeds=1,
                    batch_mode=solve_state.batch_mode,
                    seed_success=trajopt_seed_success,
                )
                trajopt_seed_traj = trajopt_seed_traj.view(
                    solve_state.num_trajopt_seeds,
                    solve_state.batch_size,
                    self.trajopt_solver.action_horizon,
                    self._dof,
                ).contiguous()
            if plan_config.enable_finetune_trajopt:
                og_value = self.trajopt_solver.interpolation_type
                self.trajopt_solver.interpolation_type = InterpolateType.LINEAR_CUDA

            traj_result = self._solve_trajopt_from_solve_state(
                goal,
                solve_state,
                trajopt_seed_traj,
                newton_iters=trajopt_newton_iters,
                return_all_solutions=plan_config.enable_finetune_trajopt,
            )

            # output of traj result will have 1 solution per batch

            # run finetune opt on 1 solution per batch:
            if plan_config.enable_finetune_trajopt:
                self.trajopt_solver.interpolation_type = og_value
            if self.store_debug_in_result:
                result.debug_info["trajopt_result"] = traj_result

            # run finetune
            if plan_config.enable_finetune_trajopt and torch.count_nonzero(traj_result.success) > 0:
                with profiler.record_function("motion_gen/finetune_trajopt"):
                    seed_traj = traj_result.raw_action.clone()  # solution.position.clone()
                    seed_traj = seed_traj.contiguous()
                    og_solve_time = traj_result.solve_time

                    scaled_dt = torch.clamp(
                        torch.max(traj_result.optimized_dt[traj_result.success])
                        * self.finetune_dt_scale,
                        self.trajopt_solver.minimum_trajectory_dt,
                    )
                    self.finetune_trajopt_solver.update_solver_dt(scaled_dt.item())

                    traj_result = self._solve_trajopt_from_solve_state(
                        goal,
                        solve_state,
                        seed_traj,
                        trajopt_instance=self.finetune_trajopt_solver,
                        num_seeds_override=solve_state.num_trajopt_seeds,
                    )

                result.finetune_time = traj_result.solve_time

                traj_result.solve_time = og_solve_time
                if self.store_debug_in_result:
                    result.debug_info["finetune_trajopt_result"] = traj_result
            elif plan_config.enable_finetune_trajopt and len(traj_result.success.shape) == 2:
                traj_result.success = traj_result.success[:, 0]

            result.success = traj_result.success

            result.interpolated_plan = traj_result.interpolated_solution
            result.solve_time += traj_result.solve_time
            result.trajopt_time = traj_result.solve_time
            result.position_error = traj_result.position_error
            result.rotation_error = traj_result.rotation_error
            result.cspace_error = traj_result.cspace_error
            result.goalset_index = traj_result.goalset_index
            result.path_buffer_last_tstep = traj_result.path_buffer_last_tstep
            result.optimized_plan = traj_result.solution
            result.optimized_dt = traj_result.optimized_dt
            if torch.count_nonzero(traj_result.success) == 0:
                result.status = MotionGenStatus.TRAJOPT_FAIL
                result.success[:] = False
            if self.store_debug_in_result:
                result.debug_info = {"trajopt_result": traj_result}
        return result



    def _plan_batch_attempts(
        self,
        solve_state: ReacherSolveState,
        start_state: JointState,
        goal_pose: Pose,
        plan_config: MotionGenPlanConfig = MotionGenPlanConfig(),
        link_poses: Optional[Dict[str, Pose]] = None,
    ):
        """Plan batch attempts for a given reacher solve state.

        Args:
            solve_state: Reacher solve state for planning.
            start_state: Start joint state for planning.
            goal_pose: Goal pose to reach for end-effector.
            plan_config: Planning parameters for motion generation.
            link_poses: Goal poses for other links in the robot.

        Returns:
            MotionGenResult: Result of batched planning.
        """
        start_time = time.time()
        plan_config = plan_config.clone()
        goal_pose = goal_pose.clone()
        if plan_config.pose_cost_metric is not None:
            valid_query = self.update_pose_cost_metric(
                plan_config.pose_cost_metric, start_state, goal_pose
            )
            if not valid_query:
                result = MotionGenResult(
                    success=torch.as_tensor(
                        [False for _ in solve_state.batch_size],
                        device=self.motion_gen.tensor_args.device,
                    ),
                    valid_query=valid_query,
                    status="Invalid Hold Partial Pose",
                )
                return result

        if solve_state.batch_env:
            if solve_state.batch_size > self.world_coll_checker.n_envs:
                log_error("Batch Env is less that goal batch")
            if plan_config.enable_graph:
                log_error("Graph Search / Geometric Planner not supported in batch_env mode")

        if plan_config.enable_graph or (
            plan_config.enable_graph_attempt is not None
            and plan_config.max_attempts >= plan_config.enable_graph_attempt
        ):
            log_warn("Batch mode enable graph is only supported with num_graph_seeds==1")
            plan_config.num_trajopt_seeds = 1
            plan_config.num_graph_seeds = 1
            solve_state.num_trajopt_seeds = 1
            solve_state.num_graph_seeds = 1
        self.update_batch_size(seeds=solve_state.num_trajopt_seeds, batch=solve_state.batch_size)

        ik_fail_count = 0
        force_graph = plan_config.enable_graph
        partial_ik = plan_config.partial_ik_opt

        time_dict = {
            "solve_time": 0,
            "ik_time": 0,
            "graph_time": 0,
            "trajopt_time": 0,
            "trajopt_attempts": 0,
        }
        best_result = None

        for n in range(plan_config.max_attempts):
            result = self._plan_from_solve_state_batch(
                solve_state,
                start_state,
                goal_pose,
                plan_config,
                link_poses=link_poses,
            )

            time_dict["solve_time"] += result.solve_time
            time_dict["ik_time"] += result.ik_time

            time_dict["graph_time"] += result.graph_time
            time_dict["trajopt_time"] += result.trajopt_time
            time_dict["trajopt_attempts"] += result.trajopt_attempts

            # if not all have succeeded, store the successful ones and re attempt:
            # TODO: update stored based on error
            if best_result is None:
                best_result = result.clone()
            else:
                # get success idx:
                idx = torch.nonzero(result.success).reshape(-1)
                if len(idx) > 0:
                    best_result.copy_idx(idx, result)

            if (
                result.status == MotionGenStatus.IK_FAIL and plan_config.ik_fail_return is not None
            ):  # IF IK fails the first time, we exit assuming the goal is not reachable
                ik_fail_count += 1
                if ik_fail_count > plan_config.ik_fail_return:
                    break

            if (
                torch.count_nonzero(best_result.success)
                >= goal_pose.batch * plan_config.success_ratio
            ):  # we want 90% targets to succeed
                best_result.status = None
                break
            if plan_config.enable_graph_attempt is not None and (
                n >= plan_config.enable_graph_attempt - 1
                and result.status != MotionGenStatus.IK_FAIL
                and not plan_config.enable_graph
            ):
                plan_config.enable_graph = True
                plan_config.partial_ik_opt = False

            if plan_config.disable_graph_attempt is not None and (
                n >= plan_config.disable_graph_attempt - 1
                and result.status in [MotionGenStatus.TRAJOPT_FAIL, MotionGenStatus.GRAPH_FAIL]
                and not force_graph
            ):
                plan_config.enable_graph = False
                plan_config.partial_ik_opt = partial_ik

            if plan_config.fail_on_invalid_query:
                if not result.valid_query:
                    best_result.valid_query = False
                    best_result.status = "Invalid Problem"
                    break
            if time.time() - start_time > plan_config.timeout:
                break
        best_result.solve_time = time_dict["solve_time"]
        best_result.ik_time = time_dict["ik_time"]
        best_result.graph_time = time_dict["graph_time"]
        best_result.trajopt_time = time_dict["trajopt_time"]
        best_result.trajopt_attempts = time_dict["trajopt_attempts"]
        best_result.attempts = n + 1
        torch.cuda.synchronize(device=self.tensor_args.device)
        if plan_config.pose_cost_metric is not None:
            self.update_pose_cost_metric(PoseCostMetric.reset_metric())

        if plan_config.time_dilation_factor is not None and torch.count_nonzero(result.success) > 0:
            result.retime_trajectory(
                plan_config.time_dilation_factor,
                interpolation_kind=self.finetune_trajopt_solver.interpolation_type,
            )
        best_result.total_time = time.time() - start_time
        return best_result

log_warn = log_info = lambda message: None

def log_error(message):
    raise RuntimeError(message)
