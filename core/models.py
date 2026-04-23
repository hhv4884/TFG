from collections import OrderedDict
import os
import torch.nn as nn
import torch

# TODO: sigmoide, Loss: BCE
class MLP_base(nn.Module):
    def __init__(self, input_size, hidden_size, n_class):
        super().__init__()
        self.net = nn.Sequential(OrderedDict([
            ('input_layer', nn.Linear(input_size, hidden_size)),
            ('first_activation_layer', nn.ReLU()),
            ('last_layer', nn.Linear(hidden_size, 1)),
            ('last_activation_layer', nn.Sigmoid())
        ]))

    def forward(self, x):
        output = self.net(x)
        return output
    
    def threshold_weight_forward(self, x, method='smallest', fraction_non_zero=1.0, to_signs=False):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                weight = layer.weight.data.clone()
                bias = layer.bias.data.clone()
                total_weights = weight.numel()
                num_non_zero = int(fraction_non_zero * total_weights)
                total_biases = bias.numel()
                num_non_zero_bias = int(fraction_non_zero * total_biases)
                if method in ['smallest', 'highest']:
                    abs_weight = torch.abs(weight).view(-1)
                    if method == 'smallest':
                        _, indices = torch.topk(abs_weight, num_non_zero, largest=True)
                    elif method == 'highest':
                        _, indices = torch.topk(abs_weight, num_non_zero, largest=False)
                    mask = torch.zeros_like(abs_weight)
                    mask[indices] = 1
                    weight = weight.view(-1) * mask
                    weight = weight.view_as(layer.weight)
                    abs_bias = torch.abs(bias)
                    if method == 'smallest':
                        _, indices_bias = torch.topk(abs_bias, num_non_zero_bias, largest=True)
                    elif method == 'highest':
                        _, indices_bias = torch.topk(abs_bias, num_non_zero_bias, largest=False)
                    mask_bias = torch.zeros_like(bias)
                    mask_bias[indices_bias] = 1
                    bias = bias * mask_bias
                elif method == 'random':
                    mask = torch.zeros_like(weight.view(-1))
                    mask[:num_non_zero] = 1
                    mask = mask[torch.randperm(mask.size(0))]
                    weight = weight.view(-1) * mask
                    weight = weight.view_as(layer.weight)
                    mask_bias = torch.zeros_like(bias)
                    mask_bias[:num_non_zero_bias] = 1
                    mask_bias = mask_bias[torch.randperm(mask_bias.size(0))]
                    bias = bias * mask_bias
                else:
                    raise ValueError("Invalid method")
                if to_signs:
                    weight = torch.sign(weight)
                    bias = torch.sign(bias)
                x = nn.functional.linear(x, weight, bias)

            else:
                x = layer(x)

        return x

    def extract_weights(self, tensor, origin_layer):
        edgelist = {}
        '''
        tensor([[ 0.6799,  0.3017],
        [-0.4485,  0.6246]], requires_grad=True)
        '''
        input_len = tensor.shape[1]
        output_len = tensor.shape[0]

        for x in range(tensor.shape[0]):
            for y in range(tensor.shape[1]):
                input_output_text = f'L{origin_layer}_{y} L{origin_layer+1}_{x}'
                edgelist[input_output_text] = tensor[x][y].item()
        return edgelist, input_len, output_len

    def extract_bias(self, tensor, origin_layer):
        biaslist = {}
        bias_len = tensor.shape[0]
        for i in range(bias_len):
            bias_text = f'L{origin_layer}_{i}'
            biaslist[bias_text] = tensor[i].item()
        return biaslist, bias_len

    def save_edgelist(self, path, epoch):
        # print(self)
        '''
        MLP_base(
          (net): Sequential(
            (input_layer): Linear(in_features=2, out_features=2, bias=True)
            (first_activation_layer): ReLU()
            (last_layer): Linear(in_features=2, out_features=2, bias=True)
            (last_activation_layer): LogSoftmax(dim=1)
          )
        )
        '''
        os.makedirs(path, exist_ok = True)
        edgelist = []
        weights_input_layer = self.net.input_layer.weight
        weights_output_layer = self.net.last_layer.weight
        # print(weights_input_layer)
        # print(weights_output_layer)
        dict_input, origin_i, target_i = self.extract_weights(weights_input_layer, 0)
        dict_output, origin_o, target_o = self.extract_weights(weights_output_layer, 1)
        edgelist.append(dict_input)
        edgelist.append(dict_output)
        origin = [origin_i, origin_o]
        target = [target_i, target_o]
        for d in range(len(edgelist)):
            file = f'{path}/epoch_{epoch}_layer{d}_edgelist.txt'
            with open(file, 'w') as f:
                # neuronas de entrada y de salida
                f.write(f'number origin neurons: {origin[d]} || number target neurons: {target[d]}\n')
                for key, value in edgelist[d].items():
                    f.write(f'{key} -> {value}\n')

    def save_biaslist(self, path, epoch):
        os.makedirs(path, exist_ok=True)
        biaslist = []
        bias_input_layer = self.net.input_layer.bias
        bias_output_layer = self.net.last_layer.bias
        dict_input, len_i = self.extract_bias(bias_input_layer, 0)
        dict_output, len_o = self.extract_bias(bias_output_layer, 1)
        biaslist.append(dict_input)
        biaslist.append(dict_output)
        sizes = [len_i, len_o]
        for d in range(len(biaslist)):
            file = f'{path}/epoch_{epoch}_layer{d}_biaslist.txt'
            with open(file, 'w') as f:
                f.write(f'number of biases: {sizes[d]}\n')
                for key, value in biaslist[d].items():
                    f.write(f'{key} -> {value}\n')

class MLP_superior(nn.Module):
    def __init__(self, input_size, hidden_sizes, n_class):
        # hidden_sizes = [32, 64, 128]
        print(hidden_sizes)
        super().__init__()
        self.flatten = nn.Flatten()
        for hidden_size in hidden_sizes:
            self.net = nn.Sequential(OrderedDict([
                ('input_layer', nn.Linear(input_size, hidden_size)),
                ('first_activation_layer', nn.ReLU()),
                ('last_layer', nn.Linear(hidden_size, n_class)),
                ('last_activation_layer', nn.LogSoftmax(dim = 1))
            ]))
        # print(self.net)
        '''
        Sequential(
        (input_layer): Linear(in_features=784, out_features=2, bias=True)
        (first_activation_layer): ReLU()
        (last_layer): Linear(in_features=2, out_features=2, bias=True)
        (last_activation_layer): LogSoftmax(dim=1)
        )
        '''
    def forward(self, x):
        x = self.flatten(x)
        output = self.net(x)
        return output

    def extract_weights(self, layer, origin_layer):
        edgelist = {}
        input_len = layer.shape[1]
        output_len = layer.shape[0]

        '''
        tensor([[ 0.6799,  0.3017],
        [-0.4485,  0.6246]], requires_grad=True)
        '''
        for x in range(layer.shape[0]):
            for y in range(layer.shape[1]):
                input_output_text = f'L{origin_layer}_{y} L{origin_layer+1}_{x}'
                edgelist[input_output_text] = layer[x][y].item()
                # y neurona de entrada x neurona de salida
        return edgelist, input_len, output_len

    def extract_bias(self, layer, origin_layer):
        biaslist = {}
        layer_bias = layer.shape[0]
        for x in range(layer.shape[0]):
            bias_text = f'L{origin_layer}_{x}'
            biaslist[bias_text] = layer[x].item()
        return biaslist, layer_bias

    def save_edgelist(self, path, epoch):
        os.makedirs(path, exist_ok = True)
        # edgelist por capa
        for name, layer in self.net.named_parameters():
            # print(f'name {name}')
            # print(f'layer {layer}')
            edgelist = {}
            origin_layer = None
            if name == "input_layer.weight":
                origin_layer = 0
                edgelist, origin, target = self.extract_weights(layer, origin_layer) #diccionario
            elif name == "last_layer.weight":
                origin_layer = 1
                edgelist, origin, target = self.extract_weights(layer, origin_layer)
            else:
                pass
            if len(edgelist.keys()) == 0:
                pass
            else:
                file = f'{path}/epoch_{epoch}_layer{origin_layer}_edgelist.txt'
                with open(file, 'w') as f:
                    # neuronas de entrada y de salida
                    f.write(f'number origin neurons: {origin} || number target neurons: {target}\n')
                    for key, value in edgelist.items():
                        f.write(f'{key} -> {value}\n')

    def save_biaslist(self, path, epoch):
        os.makedirs(path, exist_ok = True)
        for name, layer in self.net.named_parameters():
            biaslist = {}
            origin_layer = None
            if name == "input_layer.bias":
                origin_layer = 0
                biaslist, bias_layer = self.extract_bias(layer, origin_layer) #diccionario
            elif name == "last_layer.bias":
                origin_layer = 1
                biaslist, bias_layer = self.extract_bias(layer, origin_layer)
            else:
                pass
            if len(biaslist.keys()) == 0:
                pass
            else:
                file = f'{path}/epoch_{epoch}_layer{origin_layer}_biaslist.txt'
                with open(file, 'w') as f:
                    f.write(f'number of biases: {bias_layer}\n')
                    for key, value in biaslist.items():
                        f.write(f'{key} -> {value}\n')

    # forward()
    # fraction non zero = fracción no binarizada
    def threshold_weight_forward(self, x, method='smallest', fraction_non_zero=1.0, to_signs=False):
        x = self.flatten(x)
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                weight = layer.weight.data.clone()
                bias = layer.bias.data.clone()
                total_weights = weight.numel()
                num_non_zero = int(fraction_non_zero * total_weights)
                total_biases = bias.numel()
                num_non_zero_bias = int(fraction_non_zero * total_biases)
                if method in ['smallest', 'highest']:
                    abs_weight = torch.abs(weight).view(-1)
                    if method == 'smallest':
                        _, indices = torch.topk(abs_weight, num_non_zero, largest=True)
                    elif method == 'highest':
                        _, indices = torch.topk(abs_weight, num_non_zero, largest=False)
                    mask = torch.zeros_like(abs_weight)
                    mask[indices] = 1
                    weight = weight.view(-1) * mask
                    weight = weight.view_as(layer.weight)
                    abs_bias = torch.abs(bias)
                    if method == 'smallest':
                        _, indices_bias = torch.topk(abs_bias, num_non_zero_bias, largest=True)
                    elif method == 'highest':
                        _, indices_bias = torch.topk(abs_bias, num_non_zero_bias, largest=False)
                    mask_bias = torch.zeros_like(bias)
                    mask_bias[indices_bias] = 1
                    bias = bias * mask_bias
                elif method == 'random':
                    mask = torch.zeros_like(weight.view(-1))
                    mask[:num_non_zero] = 1
                    mask = mask[torch.randperm(mask.size(0))]
                    weight = weight.view(-1) * mask
                    weight = weight.view_as(layer.weight)
                    mask_bias = torch.zeros_like(bias)
                    mask_bias[:num_non_zero_bias] = 1
                    mask_bias = mask_bias[torch.randperm(mask_bias.size(0))]
                    bias = bias * mask_bias
                else:
                    raise ValueError("Invalid method")
                if to_signs:
                    weight = torch.sign(weight)
                    bias = torch.sign(bias)
                # forward
                x = nn.functional.linear(x, weight, bias)
            else:
                x = layer(x)
        return x