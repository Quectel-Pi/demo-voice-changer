import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper, numpy_helper

    from voice_changer.ai.llvc_rknn_graph import rewrite_static_scatter_nd
except ImportError:
    onnx = None


@unittest.skipIf(onnx is None, "ONNX conversion dependencies are not installed")
class LlvcRknnGraphTests(unittest.TestCase):
    def make_scatter_graph(self, data_shape, indices, updates_shape):
        data = helper.make_tensor_value_info("data", TensorProto.FLOAT, data_shape)
        updates = helper.make_tensor_value_info(
            "updates", TensorProto.FLOAT, updates_shape,
        )
        output = helper.make_tensor_value_info("result", TensorProto.FLOAT, data_shape)
        indices_tensor = numpy_helper.from_array(
            np.asarray(indices, dtype=np.int64),
            name="indices",
        )
        scatter = helper.make_node(
            "ScatterND",
            ["data", "indices", "updates"],
            ["result"],
            name="scatter",
        )
        graph = helper.make_graph(
            [scatter],
            "scatter-test",
            [data, updates],
            [output],
            initializer=[indices_tensor],
        )
        return helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 17)],
        )

    def compare_rewrite(self, model, feeds, expected):
        with tempfile.TemporaryDirectory() as temporary:
            original_path = Path(temporary) / "original.onnx"
            rewritten_path = Path(temporary) / "rewritten.onnx"
            onnx.save(model, original_path)
            self.assertEqual(rewrite_static_scatter_nd(model), 1)
            self.assertFalse(any(node.op_type == "ScatterND" for node in model.graph.node))
            onnx.save(model, rewritten_path)
            original = ort.InferenceSession(
                str(original_path), providers=["CPUExecutionProvider"],
            )
            rewritten = ort.InferenceSession(
                str(rewritten_path), providers=["CPUExecutionProvider"],
            )
            original_output = original.run(["result"], feeds)[0]
            rewritten_output = rewritten.run(["result"], feeds)[0]
            np.testing.assert_array_equal(original_output, expected)
            np.testing.assert_array_equal(rewritten_output, expected)
            np.testing.assert_array_equal(original_output, rewritten_output)

    def test_partial_contiguous_update_becomes_slice_and_concat(self):
        model = self.make_scatter_graph(
            (1, 5),
            [[0, 2], [0, 3]],
            (2,),
        )
        self.compare_rewrite(
            model,
            {
                "data": np.asarray([[1, 2, 3, 4, 5]], dtype=np.float32),
                "updates": np.asarray([7, 8], dtype=np.float32),
            },
            np.asarray([[1, 2, 7, 8, 5]], dtype=np.float32),
        )

    def test_full_update_becomes_reshape_without_scatter(self):
        model = self.make_scatter_graph(
            (2, 2),
            [[0, 0], [0, 1], [1, 0], [1, 1]],
            (4,),
        )
        self.compare_rewrite(
            model,
            {
                "data": np.zeros((2, 2), dtype=np.float32),
                "updates": np.asarray([1, 2, 3, 4], dtype=np.float32),
            },
            np.asarray([[1, 2], [3, 4]], dtype=np.float32),
        )

    def test_noncontiguous_indices_are_rejected(self):
        model = self.make_scatter_graph(
            (1, 5),
            [[0, 0], [0, 2]],
            (2,),
        )
        with self.assertRaisesRegex(ValueError, "not contiguous"):
            rewrite_static_scatter_nd(model)


if __name__ == "__main__":
    unittest.main()
