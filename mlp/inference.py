import torch
from mlp import LinearRegressionModel  # This is a must import for torch to load model
import struct


def load_model(model_path=''):
    """
    Load saved model from file
    :param model_path: mlp.pth prepared using mlp.py
    :return net: loaded model
    """
    print(f'[INFO]: Loading saved model...')
    net = torch.load(model_path)
    net = net.to('cuda:0')
    net.eval()
    return net


def test_model(mlp_model):
    """
    Test model on custom input
    :param mlp_model: pre-trained model
    :return:
    """
    print(f'[INFO]: Testing model on sample input...')
    tmp = torch.ones(1, 1).to('cuda:0')
    out = mlp_model(tmp)
    print(f'[INFO]: Test Result is: ', out.detach().cpu().numpy())


def convert_to_wts(mlp_model):
    """
    Convert weights to .wts format for TensorRT Engine
    Weights are written in the following format:
        <total-weights-count>
        weight.name <weight-count> <weight-val1> <weight-val2> ...

        -- total-weights-count: is an integer
        -- weight.name:         is used as key in TensorRT engine
        -- weight-count:        no. of weights for current layer
        -- weight-valxx:        float to c-bytes to hexadecimal

    :param mlp_model: pre-trained model
    :return:
    """
    print(f'[INFO]: Writing weights to .wts ...')
    with open('mlp.wts', 'w') as f:
        f.write(f'{len(mlp_model.state_dict().keys())}\n')
        for k, v in mlp_model.state_dict().items():
            vr = v.reshape(-1).cpu().numpy()
            f.write(f'{k} {len(vr)}')
            for vv in vr:
                f.write(" ")
                # convert weights to c-structs
                # Big-Endian (byte values) to Hex
                f.write(struct.pack('>f', float(vv)).hex())
            f.write('\n')
    print('[INFO]: Successfully, converted weights to WTS ')


def main():
    mlp_model = load_model('mlp.pth')
    test_model(mlp_model)
    convert_to_wts(mlp_model)


if __name__ == '__main__':
    main()

