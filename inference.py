import torch

model = torch.load('netxor.pt', weights_only=False)


def inference(a,b):
    global model

    e  = model(torch.tensor([a,b], dtype=torch.float32))

    return e

while True:
    torch.no_grad(inference)
    x1 = int(input('x 1> '))
    x2 = int(input('x 2> '))

    e = inference(x1, x2)
    if e > -1:
        e = 1
    else:
        e = 0
    print(e)