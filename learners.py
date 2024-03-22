import torch

from pgdattack import pgd_attack


def score(outputs, labels): return torch.sum(torch.eq(labels, torch.argmax(outputs, dim=1))).item()


def train(model, data, device, objective, optimizer, index=None, regularizer=None, sample=1, pgd=False):
    model.train(True)

    running_correct, running_loss, running_regularization, running_deltas = 0, 0, 0, None

    for inputs, labels in data:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(pgd_attack(model, objective, inputs, labels) if pgd else inputs)
        loss = objective(outputs, labels)

        if index != None:
            _, deltas = index(model, inputs)
            if running_deltas != None: running_deltas += deltas
            else: running_deltas = deltas

        # if regularizer != None:
        #     model.train(False)
        #     samples = inputs[:sample, :, :, :]

        #     for i in range(samples.shape[0]): # samples must be run one at a time
        #         input = samples[i].unsqueeze(0)
        #         regularization, _ = regularizer(model, input)
        #         regularization /= sample
        #         loss += regularization
        #         running_regularization += regularization.item()
        
        model.train(True)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_correct += score(outputs, labels)
        running_loss += loss.item() * inputs.shape[0]

    average_accuracy = running_correct*100 / len(data.dataset)
    average_loss = running_loss / len(data.dataset)
    average_regularization = (running_regularization / len(data.dataset)) if regularizer != None else 0
    average_deltas = (running_deltas / len(data.dataset)) if running_deltas != None else None
    return average_accuracy, average_loss, average_regularization, average_deltas


def test(model, data, device, objective=None, index=None, regularizer=None, sample=1, pgd=False):
    model.train(False)

    running_correct, running_loss, running_regularization, running_deltas = 0, 0, 0, None

    for inputs, labels in data:
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(pgd_attack(model, objective, inputs, labels) if pgd else inputs)
            running_correct += score(outputs, labels)

            if objective != None:
                loss = objective(outputs, labels)

                if index != None:
                    _, deltas = index(model, inputs)
                    if running_deltas != None: running_deltas += deltas
                    else: running_deltas = deltas

                # if regularizer != None:
                #     samples = inputs[:sample, :, :, :]

                #     for i in range(samples.shape[0]):
                #         input = samples[i].unsqueeze(0)
                #         regularization, _ = regularizer(model, input)
                #         regularization /= sample
                #         loss += regularization
                #         running_regularization += regularization.item()

                running_loss += loss.item() * inputs.shape[0]

    average_accuracy = running_correct*100 / len(data.dataset)
    average_loss = (running_loss / len(data.dataset)) if objective != None else float('inf')
    average_regularization = (running_regularization / len(data.dataset)) if regularizer != None else 0
    average_deltas = (running_deltas / len(data.dataset)) if running_deltas != None else None
    return average_accuracy, average_loss, average_regularization, average_deltas
