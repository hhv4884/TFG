from core.load_dataset import load_mnist
from core.models import MLP_superior
import torch
from core.new_dataset import NewDataset

if __name__ == '__main__':
    hidden_size = [2, 5, 10]
    model = MLP_superior(20, hidden_size, 2)
    # D E B U G
    print(model)
    for param in model.parameters():
       print(param)
    # 4 tensores ->
    # tensor1 = [[20],[20],...,[20]] -> 10 veces
    # tensor2 = [10]
    # tensor3 = [[10],[10]]
    # tensor4 = salida (2)

    # PRUEBA DATASET LIMPIO
    dataset_train, dataset_test = load_mnist(batch_size = 128, classes = [0,7])
    for input, output in dataset_train:
        print(input) #tensor
        print(output) #tensor
    prueba = NewDataset(dataset_test, [0,7])
    pruebalen = prueba.__len__()
    pruebain = prueba.__getitem__(0)
    print(pruebalen)
    print(pruebain)

    # PRUEBA TENSORES
    # por neuronas
    # 0.6799 peso de la 0 a la 0
    # 0.3012 peso de la 0 a la 1
    w = torch.tensor([[0.6799,  0.3017], [-0.4485,  0.6246]])
    print(w.shape)
    print(w.shape[0])
    print(w.size())

    # PRUEBA GUARDADO DE INFO RED
    model = MLP_superior(784, [2], 2)
    model.save_edgelist("prueba", 0)
    model.save_biaslist("prueba", 0)

    # print(x_train.shape)
    # con 20 99%
    # con 10 50%
    # con 15 50%
    # con 17 tmp
    # con 18 100% -> con NLLoss()
    # el 2 y el 5 apenas

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Hello! Using {device} device :)')
    print(torch.__version__)
    print(torch.version.cuda)
