import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path='handover_predictor_int8.tflite')
interpreter.allocate_tensors()

all_ops = set()
for i in range(len(interpreter._get_ops_details())):
    pass

# Use the low-level model to walk every subgraph
model = interpreter._interpreter
import flatbuffers
from tensorflow.lite.python import schema_py_generated as schema_fb

with open('handover_predictor_int8.tflite', 'rb') as f:
    buf = f.read()
tflite_model = schema_fb.Model.GetRootAsModel(buf, 0)

op_codes = []
for i in range(tflite_model.OperatorCodesLength()):
    oc = tflite_model.OperatorCodes(i)
    builtin = oc.BuiltinCode()
    name = schema_fb.BuiltinOperator.__dict__
    op_name = [k for k, v in name.items() if v == builtin]
    op_codes.append(op_name[0] if op_name else f"CUSTOM({builtin})")

print("All unique ops across all subgraphs:")
for op in sorted(set(op_codes)):
    print(" -", op)