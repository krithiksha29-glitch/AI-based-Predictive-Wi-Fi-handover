data = open('handover_predictor_int8.tflite', 'rb').read()

with open('model_data.h', 'w') as f:
    f.write('#ifndef MODEL_DATA_H\n#define MODEL_DATA_H\n\n')
    f.write('alignas(8) const unsigned char model_data[] = {\n')
    for i, b in enumerate(data):
        f.write(f'0x{b:02x}, ')
        if (i+1) % 12 == 0:
            f.write('\n')
    f.write('\n};\n')
    f.write(f'const int model_data_len = {len(data)};\n\n')
    f.write('#endif\n')

print('Saved model_data.h —', len(data), 'bytes')