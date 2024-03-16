import torch

def score(outputs, labels): return torch.sum(torch.eq(labels, torch.argmax(outputs, dim=1))).item()

def train(model, data, device, objective, optimizer, our_loss=None):
    model.train(True)

    running_correct, running_loss, running_our_loss = 0, 0, 0

    for inputs, labels in data:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = objective(outputs, labels)

        if our_loss != None:
            with torch.no_grad():
                loss_ = our_loss(model, inputs)
                running_our_loss += loss_
                loss += loss_

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_correct += score(outputs, labels)
        running_loss += loss.item() * inputs.shape[0]
        running_our_loss += loss.item() * inputs.shape[0]

    average_accuracy = running_correct*100 / len(data.dataset)
    average_loss = running_loss / len(data.dataset)
    average_our_loss = running_our_loss / len(data.dataset) if our_loss != None else float('inf')
    return average_accuracy, average_loss, average_our_loss

def test(model, data, device, objective=None):
    model.train(False)

    running_correct, running_loss = 0, 0

    for inputs, labels in data:
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            running_correct += score(outputs, labels)

            if objective != None:
                loss = objective(outputs, labels)
                running_loss += loss.item() * inputs.shape[0]

    average_accuracy = running_correct*100 / len(data.dataset)
    average_loss = (running_loss / len(data.dataset)) if objective != None else float('inf')
    return average_accuracy, average_loss
