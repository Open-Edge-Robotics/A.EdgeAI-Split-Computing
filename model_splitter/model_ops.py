import torch
import onnx
import onnx.utils
import onnx.shape_inference

def pt_to_onnx(pt_model_path, onnx_model_path):
    model = torch.load(pt_model_path, map_location='cpu')
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy_input, onnx_model_path)

def get_split_points(model_path):
    model = onnx.load(model_path)
    graph = model.graph
    inputs = set()
    split_points = []
    flag = False
    for i, node in reversed(list(enumerate(graph.node))):
        if flag:
            flag = False
            split_points.append(node.output)
        for input_ in node.input:
            if all(x not in input_ for x in ['onnx::', 'fc.weight', 'fc.bias']):
                inputs.add(input_)
        inputs = inputs.difference(node.output)
        if len(inputs) == 1:
            flag = True
    split_points.reverse()
    return split_points

def infer_output_shapes(model_path, outputs):
    inferred_model = onnx.shape_inference.infer_shapes(onnx.load(model_path))
    name_to_dim = {}
    for v in inferred_model.graph.value_info:
        dims = [d.dim_value for d in v.type.tensor_type.shape.dim]
        name_to_dim[v.name] = dims
    for o in inferred_model.graph.output:
        dims = [d.dim_value for d in o.type.tensor_type.shape.dim]
        name_to_dim[o.name] = dims

    shapes = []
    for out_name in outputs:
        shape = name_to_dim.get(out_name[0], None)
        if shape is None:
            shape = [1,64,56,56]
        shapes.append((out_name[0], shape))
    return shapes

def split_model(model_path, split_point, head_path, tail_path):
    model = onnx.load(model_path)
    input_name = model.graph.input[0].name
    output_name = model.graph.output[0].name
    onnx.utils.extract_model(model_path, head_path, [input_name], [split_point], check_model=True)
    onnx.utils.extract_model(model_path, tail_path, [split_point], [output_name], check_model=True)
