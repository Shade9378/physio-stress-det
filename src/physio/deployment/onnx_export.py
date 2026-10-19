"""FP32 ONNX export with training-only normalization folded into the weights."""
from typing import Any
import numpy as np
from physio.training.tabular import FittedModel

def export_model(fitted: FittedModel, path: Any) -> int:
    """Fold training-only scaling into a tiny FP32 ONNX opset-17 graph."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    if fitted.candidate["family"] == "linear":
        weight = fitted.model[:-1] / fitted.scaler.scale_
        bias = fitted.model[-1] - np.sum(weight * fitted.scaler.mean_)
        arrays = [("weight", weight.astype(np.float32)[:, None]),
                  ("bias", np.array([bias], dtype=np.float32))]
        nodes = [helper.make_node("MatMul", ["features", "weight"], ["linear"]),
                 helper.make_node("Add", ["linear", "bias"], ["logit"])]
        parameter_count = len(weight) + 1
    else:
        first, second = fitted.model.layers[0], fitted.model.layers[2]
        first_weight = first.weight.detach().numpy().astype(np.float64)
        folded = first_weight / fitted.scaler.scale_[None]
        bias = first.bias.detach().numpy() - folded @ fitted.scaler.mean_
        arrays = [("weight1", folded.T.astype(np.float32)), ("bias1", bias.astype(np.float32)),
                  ("weight2", second.weight.detach().numpy().T.copy()),
                  ("bias2", second.bias.detach().numpy())]
        nodes = [helper.make_node("MatMul", ["features", "weight1"], ["hidden_linear"]),
                 helper.make_node("Add", ["hidden_linear", "bias1"], ["hidden_bias"]),
                 helper.make_node("Relu", ["hidden_bias"], ["hidden"]),
                 helper.make_node("MatMul", ["hidden", "weight2"], ["linear"]),
                 helper.make_node("Add", ["linear", "bias2"], ["logit"])]
        parameter_count = sum(value.numel() for value in fitted.model.parameters())
    nodes.append(helper.make_node("Sigmoid", ["logit"], ["probability"]))
    graph = helper.make_graph(nodes, "stress_classifier",
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, len(fitted.scaler.mean_)])],
        [helper.make_tensor_value_info("probability", TensorProto.FLOAT, [None, 1])],
        [numpy_helper.from_array(array, name) for name, array in arrays])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return parameter_count
