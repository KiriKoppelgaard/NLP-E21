from typing import Iterable
from torch import nn 

def train(model:nn.Module, epochs: int, batches: Iterable, loss_fn: Callable, val_X, val_y, patience: int) -> nn.Module:
    '''
    Documentation

    Callable is anything, you can put () after, i.e. functions
    '''
    #initialise optimizer
    optimizer = ...
    val_losses = []
    best_val_loss = None

    for epoch in epochs: 
        for batch in batches: 
            X, y = prepare_batch(batch)

            #forward pass 
            y_hat = model.forward(X)
            loss = loss_fn(y, y_hat)
 
            #backward 
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        #early stopping
        val_y_hat = model.forward(val_X)
        val_loss = loss(val_y, val_y_hat)
        val_losses.append(val_loss)

        if val_loss < best_val_loss or best_val_loss is None: 
            best_val_loss = val_loss
            #save the model

        better =  [vl for vl in val_losses[-patience:] if val_loss >= vl]

        #Another way to compare it is to take the mean
        if len(better) == patience: 
            break

    #model = load_model()

    return model



