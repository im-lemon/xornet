from torch import tensor, float32, nn, optim, save
from netxor import netXOR

x = tensor(
    [
        [0,1],
        [1,0],
        [0,0],
        [1,1]
    ], dtype=float32
)

y = tensor([
    [1],
    [1],
    [0],
    [0]
], dtype=float32)

m = netXOR()
the_goat_optim = optim.AdamW(m.parameters(), lr=0.01)

def train(x,y, epoch:int):
    global m, the_goat_optim

    # gather x+y shape must match network layers.

    # get loss fn and optim ready

    loss_fun = nn.BCEWithLogitsLoss()
    the_goat_optim = optim.AdamW(m.parameters(), lr=0.01)

    # get model's prediction and calculate loss.

    prediction = m(x)
    loss = loss_fun(prediction, y)
    print(f'Epoch: {epoch} | Loss rounded to 5 decimals: {loss:.5f}')
    save(m, 'xornet.pt')
    # backward + optim step
    the_goat_optim.zero_grad()
    loss.backward()

    the_goat_optim.step()

def trainloop(epochs:int):
    for epch in range(1,epochs+1):
        train(x,y,epch)

trainloop(1000)