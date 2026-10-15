# Method excerpts from NVIDIA cuRobo, distributed under the NVIDIA License.
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Revision d64c4b005459db10c5dd867d8b30a87d5bda9bdb. See CUROBO_LICENSE.
# Fixture modification: CPU stand-ins for optimizer inputs/result container and JIT.
from __future__ import annotations
import time
from types import SimpleNamespace
import torch
from torch.autograd import profiler


def get_torch_jit_decorator(**kwargs):
    return lambda fn: fn


class TrajOptResult(SimpleNamespace):
    pass


def log_error(message):
    raise RuntimeError(message)


log_info = lambda message: None


class TrajOptSolver:
    @profiler.record_function("trajopt/get_result")
    def _get_result(
        self,
        result: WrapResult,
        return_all_solutions: bool,
        goal: Goal,
        seed_traj: JointState,
        num_seeds: int,
        batch_mode: bool = False,
    ) -> TrajOptResult:
        """Get result from the optimization problem.

        Args:
            result: Result of the optimization problem.
            return_all_solutions: Return solutions for all seeds.
            goal: Goal object containing convergence parameters.
            seed_traj: Seed trajectory used for optimization.
            num_seeds: Number of seeds used for optimization.
            batch_mode: Batch mode for problems.

        Returns:
            TrajOptResult: Result of the trajectory optimization.
        """
        st_time = time.time()
        if self.trim_steps is not None:
            result.action = result.action.trim_trajectory(self.trim_steps[0], self.trim_steps[1])
        interpolated_trajs, last_tstep, opt_dt, buffer_change = self.get_interpolated_trajectory(
            result.action
        )

        if self.sync_cuda_time:
            torch.cuda.synchronize(device=self.tensor_args.device)
        interpolation_time = time.time() - st_time
        if self.evaluate_interpolated_trajectory:
            with profiler.record_function("trajopt/evaluate_interpolated"):
                if self.use_cuda_graph_metrics and buffer_change:
                    if is_cuda_graph_reset_available():
                        self.interpolate_rollout.reset_cuda_graph()
                        buffer_change = False
                    else:
                        self.interpolate_rollout.break_cuda_graph()
                if self.use_cuda_graph_metrics and not buffer_change:
                    metrics = self.interpolate_rollout.get_metrics_cuda_graph(interpolated_trajs)
                else:
                    metrics = self.interpolate_rollout.get_metrics(interpolated_trajs)
                result.metrics.feasible = metrics.feasible
                result.metrics.position_error = metrics.position_error
                result.metrics.rotation_error = metrics.rotation_error
                result.metrics.cspace_error = metrics.cspace_error
                result.metrics.goalset_index = metrics.goalset_index

        st_time = time.time()
        if result.metrics.cspace_error is None and result.metrics.position_error is None:
            log_error("convergence check requires either goal_pose or goal_state")

        success = jit_feasible_success(
            result.metrics.feasible,
            result.metrics.position_error,
            result.metrics.rotation_error,
            result.metrics.cspace_error,
            self.position_threshold,
            self.rotation_threshold,
            self.cspace_threshold,
        )

        if False:
            feasible = torch.all(result.metrics.feasible, dim=-1)

            if result.metrics.position_error is not None:
                converge = torch.logical_and(
                    result.metrics.position_error[..., -1] <= self.position_threshold,
                    result.metrics.rotation_error[..., -1] <= self.rotation_threshold,
                )
            elif result.metrics.cspace_error is not None:
                converge = result.metrics.cspace_error[..., -1] <= self.cspace_threshold
            else:
                log_error("convergence check requires either goal_pose or goal_state")

            success = torch.logical_and(feasible, converge)
        if return_all_solutions:
            traj_result = TrajOptResult(
                success=success,
                goal=goal,
                solution=result.action.scale_by_dt(self.solver_dt_tensor, opt_dt.view(-1, 1, 1)),
                seed=seed_traj,
                position_error=(
                    result.metrics.position_error[..., -1]
                    if result.metrics.position_error is not None
                    else None
                ),
                rotation_error=(
                    result.metrics.rotation_error[..., -1]
                    if result.metrics.rotation_error is not None
                    else None
                ),
                solve_time=result.solve_time,
                metrics=result.metrics,  # TODO: index this also
                interpolated_solution=interpolated_trajs,
                debug_info={"solver": result.debug, "interpolation_time": interpolation_time},
                path_buffer_last_tstep=last_tstep,
                cspace_error=(
                    result.metrics.cspace_error[..., -1]
                    if result.metrics.cspace_error is not None
                    else None
                ),
                optimized_dt=opt_dt,
                raw_solution=result.action,
                raw_action=result.raw_action,
                goalset_index=result.metrics.goalset_index,
                optimized_seeds=result.raw_action,
            )
        else:
            # get path length:
            if self.evaluate_interpolated_trajectory:
                smooth_label, smooth_cost = self.traj_evaluator.evaluate_interpolated_smootheness(
                    interpolated_trajs,
                    opt_dt,
                    self.solver.rollout_fn.dynamics_model.cspace_distance_weight,
                    self._velocity_bounds,
                )

            else:
                smooth_label, smooth_cost = self.traj_evaluator.evaluate(
                    result.action,
                    self.solver.rollout_fn.traj_dt,
                    self.solver.rollout_fn.dynamics_model.cspace_distance_weight,
                    self._velocity_bounds,
                )

            with profiler.record_function("trajopt/best_select"):
                (
                    idx,
                    position_error,
                    rotation_error,
                    cspace_error,
                    goalset_index,
                    opt_dt,
                    success,
                ) = jit_trajopt_best_select(
                    success,
                    smooth_label,
                    result.metrics.cspace_error,
                    result.metrics.pose_error,
                    result.metrics.position_error,
                    result.metrics.rotation_error,
                    result.metrics.goalset_index,
                    result.metrics.cost,
                    smooth_cost,
                    batch_mode,
                    goal.batch,
                    num_seeds,
                    self._col,
                    opt_dt,
                )
                if batch_mode:
                    last_tstep = [last_tstep[i] for i in idx]
                else:
                    last_tstep = [last_tstep[idx.item()]]
                best_act_seq = result.action[idx]
                best_raw_action = result.raw_action[idx]
                interpolated_traj = interpolated_trajs[idx]

            if self.sync_cuda_time:
                torch.cuda.synchronize(device=self.tensor_args.device)
            if len(best_act_seq.shape) == 3:
                opt_dt_v = opt_dt.view(-1, 1, 1)
            else:
                opt_dt_v = opt_dt.view(1, 1)
            opt_solution = best_act_seq.scale_by_dt(self.solver_dt_tensor, opt_dt_v)
            select_time = time.time() - st_time
            debug_info = {}
            if self.store_debug_in_result:
                debug_info = {
                    "solver": result.debug,
                    "interpolation_time": interpolation_time,
                    "select_time": select_time,
                }
            if not torch.count_nonzero(success) > 0:
                max_dt = torch.max(opt_dt).item()
                if max_dt >= self.traj_evaluator_config.max_dt:

                    log_info(
                        "Optimized dt is above maximum_trajectory_dt, consider \
                            increasing max_trajectory_dt"
                    )
                    debug_info["dt_exception"] = True
            traj_result = TrajOptResult(
                success=success,
                goal=goal,
                solution=opt_solution,
                seed=seed_traj,
                position_error=position_error,
                rotation_error=rotation_error,
                cspace_error=cspace_error,
                solve_time=result.solve_time,
                metrics=result.metrics,  # TODO: index this also
                interpolated_solution=interpolated_traj,
                debug_info=debug_info,
                path_buffer_last_tstep=last_tstep,
                smooth_error=smooth_cost,
                smooth_label=smooth_label,
                optimized_dt=opt_dt,
                raw_solution=best_act_seq,
                raw_action=best_raw_action,
                goalset_index=goalset_index,
                optimized_seeds=result.raw_action,
            )
        return traj_result

@get_torch_jit_decorator()
def jit_feasible_success(
    feasible,
    position_error: Union[torch.Tensor, None],
    rotation_error: Union[torch.Tensor, None],
    cspace_error: Union[torch.Tensor, None],
    position_threshold: float,
    rotation_threshold: float,
    cspace_threshold: float,
):
    """JIT function to check if the optimization is successful."""
    feasible = torch.all(feasible, dim=-1)
    converge = feasible
    if position_error is not None and rotation_error is not None:
        converge = torch.logical_and(
            position_error[..., -1] <= position_threshold,
            rotation_error[..., -1] <= rotation_threshold,
        )
    elif cspace_error is not None:
        converge = cspace_error[..., -1] <= cspace_threshold

    success = torch.logical_and(feasible, converge)
    return success

@get_torch_jit_decorator(only_valid_for_compile=True)
def jit_trajopt_best_select(
    success,
    smooth_label,
    cspace_error: Union[torch.Tensor, None],
    pose_error: Union[torch.Tensor, None],
    position_error: Union[torch.Tensor, None],
    rotation_error: Union[torch.Tensor, None],
    goalset_index: Union[torch.Tensor, None],
    cost,
    smooth_cost,
    batch_mode: bool,
    batch: int,
    num_seeds: int,
    col,
    opt_dt,
):
    """JIT function to select the best solution from optimized seeds."""
    success[~smooth_label] = False
    convergence_error = 0
    # get the best solution:
    if pose_error is not None:
        convergence_error = pose_error[..., -1]
    elif cspace_error is not None:
        convergence_error = cspace_error[..., -1]

    running_cost = torch.mean(cost, dim=-1) * 0.0001
    error = convergence_error + smooth_cost + running_cost
    error[~success] += 10000.0
    if batch_mode:
        idx = torch.argmin(error.view(batch, num_seeds), dim=-1)
        idx = idx + num_seeds * col
        success = success[idx]
    else:
        idx = torch.argmin(error, dim=0)

        success = success[idx : idx + 1]

    # goalset_index = position_error = rotation_error = cspace_error = None
    if position_error is not None:
        position_error = position_error[idx, -1]
    if rotation_error is not None:
        rotation_error = rotation_error[idx, -1]
    if cspace_error is not None:
        cspace_error = cspace_error[idx, -1]
    if goalset_index is not None:
        goalset_index = goalset_index[idx, -1]

    opt_dt = opt_dt[idx]

    return idx, position_error, rotation_error, cspace_error, goalset_index, opt_dt, success

