from collections import OrderedDict
import torch

def load_state(net, checkpoint):
    source_state = checkpoint['state_dict']
    target_state = net.state_dict()
    new_target_state = collections.OrderedDict()
    for target_key, target_value in target_state.items():
        if target_key in source_state and source_state[target_key].size() == target_state[target_key].size():
            new_target_state[target_key] = source_state[target_key]
        else:
            new_target_state[target_key] = target_state[target_key]
            print('[WARNING] Not found pre-trained parameters for {}'.format(target_key))

    net.load_state_dict(new_target_state)


def load_from_mobilenet(net, checkpoint):
    source_state = checkpoint['state_dict']
    target_state = net.state_dict()
    new_target_state = collections.OrderedDict()
    for target_key, target_value in target_state.items():
        k = target_key
        if k.find('model') != -1:
            k = k.replace('model', 'module.model')
        if k in source_state and source_state[k].size() == target_state[target_key].size():
            new_target_state[target_key] = source_state[k]
        else:
            new_target_state[target_key] = target_state[target_key]
            print('[WARNING] Not found pre-trained parameters for {}'.format(target_key))

    net.load_state_dict(new_target_state)

    
def load_state_mnv2_pretrained(net):
    source_state = torch.hub.load('pytorch/vision:v0.8.0', 'mobilenet_v2', pretrained=True).state_dict()
    target_state = net.state_dict()
    new_target_state = OrderedDict()
    for target_key, target_value in target_state.items():
        mapped_key = target_key.replace("model", "features")
        if mapped_key in source_state and source_state[mapped_key].size() == target_state[target_key].size():
            new_target_state[target_key] = source_state[mapped_key]
        else:
            new_target_state[target_key] = target_state[target_key]
            print('[WARNING] Not found pre-trained parameters for {}'.format(target_key))
    print("All weights have been matched.")
    net.load_state_dict(new_target_state)