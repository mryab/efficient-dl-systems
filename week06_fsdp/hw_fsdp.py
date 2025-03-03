import contextlib
import logging
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from typing import Any, Callable, Dict, List, NoReturn, Optional, Tuple, Type, cast

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd.profiler import record_function
from torch.distributed._composable.fsdp._fsdp_param import ParamModuleInfo
from torch.distributed._composable.fsdp._fsdp_param_group import _get_param_module_infos
from torch.distributed.device_mesh import DeviceMesh, _get_device_handle
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.utils._pytree import tree_flatten, tree_unflatten

cls_to_fsdp_cls: Dict[Type, Type] = {}

logger = logging.getLogger("hw_fsdp")


class TrainingState(Enum):
    FORWARD = auto()
    PRE_BACKWARD = auto()
    POST_BACKWARD = auto()
    IDLE = auto()


@dataclass(frozen=True)
class MixedPrecisionPolicy:
    param_dtype: Optional[torch.dtype] = None
    reduce_dtype: Optional[torch.dtype] = None


class ShardedState(Enum):
    SHARDED = auto()
    SHARDED_POST_FORWARD = auto()
    UNSHARDED = auto()


class FSDPParam:
    orig_dtype: torch.dtype
    param_dtype: Optional[torch.dtype]
    reduce_dtype: Optional[torch.dtype]
    _orig_size: torch.Size
    sharded_size: torch.Size
    sharded_param: nn.Parameter
    _unsharded_param: nn.Parameter
    _sharding_spec: DTensorSpec

    def __init__(
        self,
        param: nn.Parameter,
        module_info: ParamModuleInfo,
        param_fqn: str,
        mesh: DeviceMesh,
        device: torch.device,
        mp_policy: MixedPrecisionPolicy,
    ):
        self._module_info = module_info
        self._param_fqn = param_fqn
        self.mesh = mesh
        self.device = device
        self._init_sharded_param(param)
        self._init_dtype_attrs(mp_policy)

    @torch.no_grad()
    def _init_sharded_param(self, param: nn.Parameter):
        self.fsdp_placement = Shard(0)
        shard_dim = self.fsdp_placement.dim
        self._sharding_spec = DTensorSpec(
            self.mesh,
            (self.fsdp_placement,),
            tensor_meta=TensorMeta(param.size(), param.stride(), param.dtype),
        )
        self._orig_size = param.size()
        shard_rank = self.mesh.get_local_rank()
        shard_world_size = self.mesh.size(0)
        assert param.size(shard_dim) % shard_world_size == 0
        # TODO: split param into shards, save local shard to `self.sharded_param`
        # (make sharded param a `DTensor`)

        self._setattr_on_module(self.sharded_param)
        self.sharded_state = ShardedState.SHARDED

    def _init_dtype_attrs(self, mp_policy: MixedPrecisionPolicy):
        param_dtype, reduce_dtype = (mp_policy.param_dtype, mp_policy.reduce_dtype)
        self.orig_dtype = self.sharded_param.dtype
        # Clamp `reduce_dtype` to `None` if no casting is required: since
        # gradients are computed in `param_dtype`, if `reduce_dtype` matches,
        # then we do not need extra casting
        if reduce_dtype == param_dtype:
            reduce_dtype = None
        # Clamp `param_dtype` to `None` if no casting is required
        if param_dtype == self.orig_dtype:
            param_dtype = None
        self.param_dtype = param_dtype
        self.reduce_dtype = reduce_dtype
        # None indicates that the mixed precision is not enabled

    def to_sharded(self) -> None:
        self._setattr_on_module(self.sharded_param)
        # TODO: free unsharded param
        self.sharded_state = ShardedState.SHARDED

    def to_unsharded(self) -> None:
        # Assume that the data has been allocated and all-gathered
        self._setattr_on_module(self.unsharded_param)
        self._unsharded_param = nn.Parameter(
            self.unsharded_param.data,
            requires_grad=self.unsharded_param.requires_grad,
        )
        self.sharded_state = ShardedState.UNSHARDED

    def _setattr_on_module(self, param: nn.Parameter) -> None:
        unsafe_setattr_param(
            self._module_info.module, self._module_info.param_name, param
        )

    @property
    def unsharded_param(self) -> nn.Parameter:  # ND
        if not hasattr(self, "_unsharded_param"):
            pass
            # TODO: create unsharded param and save it to `self._unsharded_param`
        return self._unsharded_param

    def __repr__(self):
        return f"FSDPParam(fqn={self._param_fqn}, orig_size={self._orig_size})"


def alloc_storage(tensor: torch.Tensor) -> None:
    size = tensor.numel() * tensor.itemsize
    if (storage := tensor.untyped_storage()).size() != size:
        storage.resize_(size)


def free_storage(tensor: torch.Tensor) -> None:
    if (storage := tensor.untyped_storage()).size() != 0:
        storage.resize_(0)


# NOTE: These bypass `nn.Module.__setattr__` checks, which incur non-trivial
# CPU overhead, if the module did not override it. For FSDP, we know we do not
# need those checks when transitioning between sharded/unsharded parameters.
def unsafe_setattr_param(
    module: nn.Module, param_name: str, param: nn.Parameter
) -> None:
    if getattr(module.__setattr__, "__func__", None) is nn.Module.__setattr__:
        module._parameters[param_name] = param
    else:  # slow path
        setattr(module, param_name, param)


class FSDPCommContext:
    def __init__(self, device_type: str):
        self.device_handle = _get_device_handle(device_type)
        high_priority = -1
        self.all_gather_stream = self.device_handle.Stream(priority=high_priority)
        self.reduce_scatter_stream = self.device_handle.Stream(priority=high_priority)
        # Post-forward order for explicit backward prefetching
        self.post_forward_order: List[FSDPModule] = []  # will cause ref cycles


def fully_shard(
    module: nn.Module,
    *,
    module_fqn: str,
    comm_ctx: FSDPCommContext,
    mesh: DeviceMesh,
    reshard_after_forward: bool = True,
    mp_policy: MixedPrecisionPolicy = MixedPrecisionPolicy(),
):
    if isinstance(module, (nn.ModuleList, nn.ModuleDict)):
        raise ValueError(
            f"fully_shard does not support containers that do not implement forward: {module}"
        )
    if mesh.ndim != 1:
        raise ValueError(f"fully_shard expects a 1D DeviceMesh but got {mesh}")

    device_handle = _get_device_handle(mesh.device_type)
    device = torch.device(mesh.device_type, device_handle.current_device())

    module.register_forward_pre_hook(pre_forward, prepend=True, with_kwargs=True)
    module.register_forward_hook(post_forward, prepend=False)

    module.to(device)

    module.fsdp_params = [
        FSDPParam(
            param,
            module_info,
            f"{module_fqn}.{name}",
            mesh,
            device,
            mp_policy,
        )
        for (name, param), module_info in zip(
            module.named_parameters(),
            _get_param_module_infos(list(module.parameters()), (module,)),
        )
    ]
    module._training_state = TrainingState.IDLE
    module._sharded_state = ShardedState.SHARDED
    module._module_fqn = module_fqn
    module.comm_ctx = comm_ctx
    module._post_forward_indices = []
    module._reshard_after_forward = reshard_after_forward
    module._all_gather_event = None
    module._post_reduce_event = None

    # Place FSDP leftmost for highest priority in the method resolution order
    cls = module.__class__
    new_cls = cls_to_fsdp_cls.get(cls, None)
    if not new_cls:
        dct = {"__deepcopy__": unimplemented_deepcopy}
        new_cls = type(f"FSDP{cls.__name__}", (FSDPModule, cls), dct)
        cls_to_fsdp_cls[cls] = new_cls
    module.__class__ = new_cls
    return module


def unimplemented_deepcopy(*args: Any, **kwargs: Any) -> NoReturn:
    raise AssertionError(
        "FSDP does not support deepcopy. Please use state dict for serialization."
    )


class FSDPModule:
    fsdp_params: List[FSDPParam]
    _training_state: TrainingState
    _sharded_state: ShardedState
    _module_fqn: str
    comm_ctx: FSDPCommContext
    _post_forward_indices: List[int]
    _reshard_after_forward: bool
    _all_gather_event: Optional[torch.Event]
    _post_reduce_event: Optional[torch.Event]

    def __new__(cls, *args, **kwargs):
        # Use index 2 since 0 is the dynamically constructed `FSDP<...>` class
        # and index 1 is the `FSDPModule` class itself
        orig_cls = cls.__mro__[2]
        self = orig_cls.__new__(orig_cls, *args, **kwargs)
        self.__init__(*args, **kwargs)
        return self

    def unshard(self):
        if self._all_gather_event is not None:  # already called, pending wait
            return
        if self.is_unsharded:
            return  # no-op
        with record_function(self.with_fqn("FSDP::all_gather")):
            pass
            # TODO: allocate unsharded param data
            # TODO: all-gather sharded params into unsharded params

    def wait_for_unshard(self):
        # TODO: wait for all-gather to complete
        # TODO: set unsharded params on module
        self._to_unsharded()

    def reshard(self):
        if self._training_state == TrainingState.FORWARD:
            if not self._reshard_after_forward:
                return
        self._to_sharded()

    def record_post_forward(self) -> None:
        post_forward_index = len(self.comm_ctx.post_forward_order)
        self.comm_ctx.post_forward_order.append(self)
        self._post_forward_indices.append(post_forward_index)

    def register_post_backward_final_callback(self):
        Variable._execution_engine.queue_callback(self._post_backward_final_callback)

    def _post_backward_final_callback(self) -> None:
        if self.is_unsharded:
            # Run post-backward in case forward inputs did not require
            # gradient so the autograd backward did not run
            post_backward(self)
        self._training_state = TrainingState.IDLE
        # TODO: wait for reduce-scatter to complete
        self._post_forward_indices.clear()
        self.comm_ctx.post_forward_order.clear()

    def _backward_prefetch(self) -> None:
        # TODO (task-3): explicitly prefetch the next module during backward
        pass

    @staticmethod
    def _prefetch_unshard(target_fsdp_module: "FSDPModule") -> None:
        with record_function(
            f"FSDP::backward_prefetch for {target_fsdp_module._module_fqn}"
        ), target_fsdp_module.use_training_state(TrainingState.PRE_BACKWARD):
            target_fsdp_module.unshard()

    def _to_sharded(self):
        if not self.is_sharded:
            for fsdp_param in self.fsdp_params:
                fsdp_param.to_sharded()
            self._sharded_state = ShardedState.SHARDED

    def _to_unsharded(self):
        for fsdp_param in self.fsdp_params:
            fsdp_param.to_unsharded()
        if not self.is_unsharded:
            self._sharded_state = ShardedState.UNSHARDED

    @property
    def is_sharded(self) -> bool:
        return self._sharded_state == ShardedState.SHARDED

    @property
    def is_unsharded(self) -> bool:
        return self._sharded_state == ShardedState.UNSHARDED

    @contextlib.contextmanager
    def use_training_state(self, training_state: TrainingState):
        old_training_state = self._training_state
        self._training_state = training_state
        try:
            yield
        finally:
            self._training_state = old_training_state

    def with_fqn(self, label: str) -> str:
        if self._module_fqn:
            return f"{label} ({self._module_fqn})"
        return label


def pre_forward(module: FSDPModule, args: Tuple[Any, ...], kwargs: Dict[str, Any]):
    # When composing with module-hook-based activation checkpointing, the
    # the pre-backward hook is responsible for the unshard
    if module._training_state == TrainingState.PRE_BACKWARD:
        return args, kwargs
    logger.debug("%s", module.with_fqn("FSDP::pre_forward"))
    with record_function(module.with_fqn("FSDP::pre_forward")):
        module._training_state = TrainingState.FORWARD
        module.unshard()
        args, kwargs = register_post_backward_hook(module, args, kwargs)
        module.wait_for_unshard()
        return args, kwargs


def post_forward(module: FSDPModule, input: Any, output: Any):
    # When composing with module-hook-based activation checkpointing, the
    # post-backward hook is responsible for the reshard
    if module._training_state == TrainingState.PRE_BACKWARD:
        return output
    logger.debug("%s", module.with_fqn("FSDP::post_forward"))
    with record_function(module.with_fqn("FSDP::post_forward")):
        module.reshard()
        module.record_post_forward()
        module._training_state = TrainingState.IDLE
    output = register_pre_backward_hook(partial(pre_backward, module), output)
    return output


def pre_backward(module: FSDPModule, grad: torch.Tensor):
    module.register_post_backward_final_callback()
    logger.debug("%s", module.with_fqn("FSDP::pre_backward"))
    if module._training_state == TrainingState.PRE_BACKWARD:
        return
    with record_function(module.with_fqn("FSDP::pre_backward")):
        module._training_state = TrainingState.PRE_BACKWARD
        module.unshard()  # no-op if prefetched
        module.wait_for_unshard()
        # module._backward_prefetch()
    return grad


def post_backward(module: FSDPModule):
    logger.debug("%s", module.with_fqn("FSDP::post_backward"))
    module._training_state = TrainingState.POST_BACKWARD
    with record_function(module.with_fqn("FSDP::post_backward_reshard")):
        module.reshard()
    with record_function(module.with_fqn("FSDP::post_backward_reduce")):
        pass
        # TODO: reduce-scatter unsharded grads into sharded grads
        # TODO: free unsharded grads


def register_pre_backward_hook(hook: Callable, output: Any) -> Any:
    if not torch.is_grad_enabled():
        return output
    flat_outputs, _ = tree_flatten(output)
    for t in flat_outputs:
        if torch.is_tensor(t) and t.requires_grad:
            t.register_hook(hook)
    return output


def register_post_backward_hook(
    module: FSDPModule, args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    if not torch.is_grad_enabled():
        return args, kwargs
    args_list, args_spec = tree_flatten(args)
    kwargs_list, kwargs_spec = tree_flatten(kwargs)
    args_kwargs_list = list(args_list) + list(kwargs_list)
    inp_tensor_indices: List[int] = []
    inp_tensors: List[torch.Tensor] = []
    for i, obj in enumerate(args_kwargs_list):
        if torch.is_tensor(obj) and obj.requires_grad:
            inp_tensor_indices.append(i)
            inp_tensors.append(obj)
    inp_tensors = RegisterPostBackwardFunction.apply(
        module,
        *(fsdp_param.unsharded_param for fsdp_param in module.fsdp_params),
        *inp_tensors,
    )
    unsharded_params, inp_tensors = (
        inp_tensors[: len(module.fsdp_params)],
        inp_tensors[len(module.fsdp_params) :],
    )
    for fsdp_param, unsharded_param in zip(module.fsdp_params, unsharded_params):
        unsharded_param._is_param = True
        fsdp_param._unsharded_param = cast(nn.Parameter, unsharded_param)
    if len(inp_tensors) == 0:
        return args, kwargs  # no tensors that require gradients
    for inp_tensor_idx, inp_tensor in zip(inp_tensor_indices, inp_tensors):
        args_kwargs_list[inp_tensor_idx] = inp_tensor
    args_list = args_kwargs_list[: len(args_list)]
    kwargs_list = args_kwargs_list[len(args_list) :]
    args = tree_unflatten(args_list, args_spec)
    kwargs = tree_unflatten(kwargs_list, kwargs_spec)
    return args, kwargs


class RegisterPostBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module: FSDPModule, *inputs: torch.Tensor):
        # All tensors in `inputs` should require gradient
        ctx.module = module
        return inputs

    @staticmethod
    def backward(ctx, *grads: torch.Tensor):
        unsharded_param_grads, inp_grads = (
            grads[: len(ctx.module.fsdp_params)],
            grads[len(ctx.module.fsdp_params) :],
        )
        for fsdp_param, unsharded_param_grad in zip(
            ctx.module.fsdp_params, unsharded_param_grads, strict=True
        ):
            if unsharded_param_grad is None:
                raise ValueError(
                    f"{fsdp_param._param_fqn} got unsharded during forward, but got no gradient after backward."
                )
            fsdp_param._unsharded_param.grad = unsharded_param_grad
        post_backward(ctx.module)
        return (
            None,
            *(None for _ in unsharded_param_grads),
            *inp_grads,
        )
