import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(764, 100),
    nn.ReLU(),
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Linear(50, 10),
    nn.Sigmoid()
)

model = nn.Sequential()
model.add_module("dense1", nn.Linear(8, 12))
model.add_module("act1", nn.ReLU())
model.add_module("dense2", nn.Linear(12, 8))
model.add_module("act2", nn.ReLU())
model.add_module("output", nn.Linear(8, 1))
model.add_module("outact", nn.Sigmoid())

nn.Linear(764, 100, device="cpu")

loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(y_pred, y)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for n in range(num_epochs):
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(model)

torch.save(model.state_dict(), "my_model.pickle") #save only the model parameters

model = nn.Sequential(
    nn.Linear(764, 100),
    nn.ReLU(),
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Linear(50, 10),
    nn.Sigmoid()
)
model.load_state_dict(torch.load("my_model.pickle"))  # load the model parameters