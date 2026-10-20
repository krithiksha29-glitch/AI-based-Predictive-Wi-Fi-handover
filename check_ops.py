import tensorflow as tf

interp = tf.lite.Interpreter(model_path="handover_predictor_int8.tflite")
interp.allocate_tensors()

print("Ops used in this model:")
ops_used = set()
for op in interp._get_ops_details():
    ops_used.add(op['op_name'])

for op_name in sorted(ops_used):
    print(" -", op_name)

print()
if "BATCH_MATMUL" in ops_used:
    print("⚠️  BATCH_MATMUL is present — this op MUST be supported by the library.")
else:
    print("✅ No BATCH_MATMUL in this model.")