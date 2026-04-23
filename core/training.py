import torch
import torch.nn as nn
import torch.optim as optim
import copy
from torch.nn import BCELoss
from .utils import clean_data, save_logs
from .models import MLP_superior, MLP_base

# TODO
import copy
import torch
import torch.nn as nn
import torch.optim as optim

def train_model_base_with_early_stop(model, train_loader, test_loader, path, device):
    max_epochs = 10
    lr = 0.001
    patience = 5
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    best_accuracy = 0
    best_epoch = 0
    best_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0
    for epoch in range(max_epochs):
        log_file = save_logs(epoch, path)
        model.train()
        correct_train = 0
        total_train = 0
        best_test_accuracy_epoch = 0
        for step, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            preds = (outputs > 0.5).float()
            correct_train += (preds == y_batch).sum().item()
            total_train += y_batch.size(0)
            train_acc = 100 * correct_train / total_train
            model.eval()
            correct_test = 0
            total_test = 0
            with torch.no_grad():
                for t_x, t_y in test_loader:
                    t_x = t_x.to(device)
                    t_y = t_y.to(device).float().unsqueeze(1)
                    t_out = model(t_x)
                    t_preds = (t_out > 0.5).float()
                    correct_test += (t_preds == t_y).sum().item()
                    total_test += t_y.size(0)
            test_accuracy = 100 * correct_test / total_test
            if test_accuracy > best_test_accuracy_epoch:
                best_test_accuracy_epoch = test_accuracy
            model.train()
            msg = (
                f"Epoch [{epoch+1}/{max_epochs}], Step [{step+1}/{len(train_loader)}] | "
                f"Loss: {loss.item():.4f} | Train Acc: {train_acc:.2f}% | "
                f"Test Acc: {test_accuracy:.2f}%"
            )
            print(msg)
            print(msg, file=log_file)
        if best_test_accuracy_epoch > best_accuracy:
            best_accuracy = best_test_accuracy_epoch
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            print(f"Early stopping at epoch {epoch+1}", file=log_file)
            log_file.close()
            break
        log_file.close()
        scheduler.step()
    model.load_state_dict(best_weights)
    model.save_edgelist(path, best_epoch)
    model.save_biaslist(path, best_epoch)

def train_model_base_without_early_stop(model, train_loader, test_loader, path, device):
    max_epochs = 10
    lr = 0.001
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    best_accuracy = 0
    best_epoch = 0
    best_weights = copy.deepcopy(model.state_dict())
    for epoch in range(max_epochs):
        log_file = save_logs(epoch, path)
        model.train()
        correct_train = 0
        total_train = 0
        for step, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            preds = (outputs > 0.5).float()
            correct_train += (preds == y_batch).sum().item()
            total_train += y_batch.size(0)
            train_acc = 100 * correct_train / total_train
            model.eval()
            correct_test = 0
            total_test = 0
            with torch.no_grad():
                for t_x, t_y in test_loader:
                    t_x = t_x.to(device)
                    t_y = t_y.to(device).float().unsqueeze(1)
                    t_out = model(t_x)
                    t_preds = (t_out > 0.5).float()
                    correct_test += (t_preds == t_y).sum().item()
                    total_test += t_y.size(0)
            test_accuracy = 100 * correct_test / total_test
            model.train()
            msg = (
                f"Epoch [{epoch+1}/{max_epochs}], "
                f"Step [{step+1}/{len(train_loader)}] | "
                f"Loss: {loss.item():.4f} | "
                f"Train Acc: {train_acc:.2f}% | "
                f"Test Acc: {test_accuracy:.2f}%"
            )
            print(msg)
            print(msg, file=log_file)
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_epoch = epoch
                best_weights = copy.deepcopy(model.state_dict())
        log_file.close()
        scheduler.step()
    model.load_state_dict(best_weights)
    model.save_edgelist(path, best_epoch)
    model.save_biaslist(path, best_epoch)

def train_model_superior_without_early_stop(model, train_data, test_data, path, device):
    max_epochs = 10
    learning_rate = 0.01
    criterion = nn.NLLLoss()
    # remapeo de clases -> 0<= target < num_clases
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= max_epochs)
    best_weights = model.state_dict()
    best_test_accuracy = 0
    best_test_epoch = 0
    best_test_step = 0
    for epoch in range(max_epochs):
        # modo train
        model.train()
        correct_train = 0
        total_train = 0
        log_file = save_logs(epoch, path)
        for step,(input, output) in enumerate(train_data):
            # print(input)
            # por qué la mayoría del input es -1? -> normalización del transforms del load
            input = input.to(device)
            output = output.to(device)
            optimizer.zero_grad()
            # error -> self.flatten(x)
            actual_output = model(input)
            total_train += output.size(0)
            # respecto al batch
            values, class_predicted_train = torch.max(actual_output, 1)
            correct_train += (class_predicted_train == output).sum().item()
            # print(f'actual output: {actual_output}')
            # tensor([[2],[2]...,[2]])
            # print(f'output: {output}')
            # tensor con todas las labels de salida -> limpiar con las del caso de estudio
            loss = criterion(actual_output, output)
            loss.backward()
            optimizer.step()

            model.eval()
            correct_test = 0
            total_test = 0
            with torch.no_grad():
                for t_input, t_output in test_data:
                    t_input = t_input.to(device)
                    t_output = t_output.to(device)
                    # tensor con labels
                    t_actual_output = model(t_input)
                    # tensor con las salidas de las dos últimas neuronas para cada label
                    # print(t_output.shape)
                    # print(f'actual output: {t_actual_output}')
                    # print(f'output: {t_output}')
                    values, class_predicted = torch.max(t_actual_output, 1)
                    total_test += t_output.size(0)
                    correct_test += (class_predicted == t_output).sum().item()
            test_accuracy = 100 * correct_test / total_test
            train_accuracy = 100 * correct_train / total_train
            # final de la epoca
            model.train()
            print(f"Epoch [{epoch+1}/{max_epochs}], Step [{step+1}/{len(train_data)}], "
                  f"Loss: {loss.item():.4f}, Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%",
                  file= log_file)
            print(f"Epoch [{epoch+1}/{max_epochs}], Step [{step+1}/{len(train_data)}], "
                  f"Loss: {loss.item():.4f}, Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%")

            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
                best_test_epoch = epoch
                best_test_step = step
                best_weights = copy.deepcopy(model.state_dict())
        log_file.close()
        scheduler.step()
    model.load_state_dict(best_weights)
    model.save_edgelist(path, best_test_epoch)
    model.save_biaslist(path, best_test_epoch)

def train_model_superior_with_early_stop(model, train_data, test_data, path, device):
    max_epochs = 10
    learning_rate = 0.01
    patience = 5
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    best_weights = copy.deepcopy(model.state_dict())
    best_test_accuracy = 0
    best_test_epoch = 0
    counter = 0
    for epoch in range(max_epochs):
        model.train()
        correct_train = 0
        total_train = 0
        log_file = save_logs(epoch, path)
        best_test_accuracy_epoch = 0
        for step, (input, output) in enumerate(train_data):
            input = input.to(device)
            output = output.to(device)
            optimizer.zero_grad()
            actual_output = model(input)
            loss = criterion(actual_output, output)
            loss.backward()
            optimizer.step()
            values, class_predicted_train = torch.max(actual_output, 1)
            correct_train += (class_predicted_train == output).sum().item()
            total_train += output.size(0)
            model.eval()
            correct_test = 0
            total_test = 0
            with torch.no_grad():
                for t_input, t_output in test_data:
                    t_input = t_input.to(device)
                    t_output = t_output.to(device)
                    t_actual_output = model(t_input)
                    values, class_predicted = torch.max(t_actual_output, 1)
                    correct_test += (class_predicted == t_output).sum().item()
                    total_test += t_output.size(0)
            test_accuracy = 100 * correct_test / total_test
            if test_accuracy > best_test_accuracy_epoch:
                best_test_accuracy_epoch = test_accuracy
            train_accuracy = 100 * correct_train / total_train
            model.train()
            msg = (
                f"Epoch [{epoch+1}/{max_epochs}], Step [{step+1}/{len(train_data)}], "
                f"Loss: {loss.item():.4f}, Train Acc: {train_accuracy:.2f}%, "
                f"Test Acc: {test_accuracy:.2f}%"
            )
            print(msg)
            print(msg, file=log_file)
        if best_test_accuracy_epoch > best_test_accuracy:
            best_test_accuracy = best_test_accuracy_epoch
            best_test_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            print(f"Early stopping at epoch {epoch+1}", file=log_file)
            log_file.close()
            break
        log_file.close()
        scheduler.step()
    model.load_state_dict(best_weights)
    model.save_edgelist(path, best_test_epoch)
    model.save_biaslist(path, best_test_epoch)