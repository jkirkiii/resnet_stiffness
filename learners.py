import torch
from pgdattack import pgd_attack


def score(outputs, labels): return torch.sum(torch.eq(labels, torch.argmax(outputs, dim=1))).item()


def train(model, data, device, objective, optimizer, regularizer=None, pgd=False):
    model.train(True)

    running_correct, running_loss, running_regularization = 0, 0, 0

    for inputs, labels in data:
        inputs, labels = inputs.to(device), labels.to(device)

        if pgd:
            inputs = pgd_attack(model, torch.nn.CrossEntropyLoss(), inputs, labels, .01, .005, 5)

        outputs = model(inputs)
        loss = objective(outputs, labels)

        if regularizer != None:
            regularization, _ = regularizer(model, inputs)
            loss += regularization
            running_regularization += regularization.item() * inputs.shape[0]

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_correct += score(outputs, labels)
        running_loss += loss.item() * inputs.shape[0]

    average_accuracy = running_correct*100 / len(data.dataset)
    average_loss = running_loss / len(data.dataset)
    average_regularization = (running_regularization / len(data.dataset)) if regularizer != None else 0
    return average_accuracy, average_loss, average_regularization


def test(model, data, device, objective=None, regularizer=None):
    model.train(False)

    running_correct, running_loss, running_regularization = 0, 0, 0

    for inputs, labels in data:
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            running_correct += score(outputs, labels)

            if objective != None:
                loss = objective(outputs, labels)

                if regularizer != None:
                    regularization, _ = regularizer(model, inputs)
                    loss += regularization
                    running_regularization += regularization.item() * inputs.shape[0]

                running_loss += loss.item() * inputs.shape[0]

    average_accuracy = running_correct*100 / len(data.dataset)
    average_loss = (running_loss / len(data.dataset)) if objective != None else float('inf')
    average_regularization = (running_regularization / len(data.dataset)) if regularizer != None else 0
    return average_accuracy, average_loss, average_regularization
