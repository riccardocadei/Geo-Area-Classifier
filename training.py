import torch 
from plot import plot_train_val

train_losses = []
valid_losses = []

def training(model, device, train_loader, valid_loader, optimizer,  criterion, num_epochs, model_name='model', plot=True):
    for epoch in range(1, num_epochs + 1):
        # keep-track-of-training-and-validation-loss
        train_loss = 0.0
        valid_loss = 0.0
        
        # training-the-model
        model.train()
        for data, target in train_loader:
            # move-tensors-to-GPU 
            data = data.to(device)
            target = target.to(device)
            
            # clear-the-gradients-of-all-optimized-variables
            optimizer.zero_grad()
            # forward-pass: compute-predicted-outputs-by-passing-inputs-to-the-model
            output = model(data)
            # calculate-the-batch-loss
            loss = criterion(output, target)
            # backward-pass: compute-gradient-of-the-loss-wrt-model-parameters
            loss.backward()
            # perform-a-ingle-optimization-step (parameter-update)
            optimizer.step()
            # update-training-loss
            train_loss += loss.item() * data.size(0)
            
        # validate-the-model
        model.eval()
        for data, target in valid_loader:
            
            data = data.to(device)
            target = target.to(device)
            
            output = model(data)
            
            loss = criterion(output, target)
            
            # update-average-validation-loss 
            valid_loss += loss.item() * data.size(0)
        
        # calculate-average-losses
        train_loss = train_loss/len(train_loader.sampler)
        valid_loss = valid_loss/len(valid_loader.sampler)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        if valid_loss <= min(valid_losses):
            torch.save(model.state_dict(), "./models/" + model_name + ".pth")
        
        # print-training/validation-statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
             epoch, train_loss, valid_loss))

    if plot==True: 
        plot_train_val(train_losses, valid_losses)

    return 

def testing(model, device, data_loader, name_data):
    # load the model
    model.load_state_dict(torch.load("./models/model.pth"))
    # test-the-model
    model.eval()  # it-disables-dropout
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        print(name_data+'Accuracy of the model: {} %'.format(100 * correct / total))
    return 
