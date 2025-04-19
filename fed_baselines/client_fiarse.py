from fed_baselines.client_base import FedClient
import copy
from collections import Counter
from utils.models import *

from torch.utils.data import DataLoader


class FIARSEClient(FedClient):

    def __init__(self, name, epoch, dataset_id, model_name, batch_size, lr):
        super().__init__(name, epoch, dataset_id, model_name, batch_size, lr)

    def update(self, model_state_dict, mask):
        super().update(model_state_dict)
        self.mask = mask

    def train(self):
        """
        Client trains the model on local dataset
        :return: Local updated model, number of local data points, training loss
        """
        train_loader = DataLoader(
            self.trainset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        self.model.to(self._device)
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self._lr,
            momentum=self._momentum,
        )
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self._lr, weight_decay=1e-4)
        loss_func = nn.CrossEntropyLoss()

        # nonzero_count, total_count, sparsity = self.get_sparsity()
        # print(f"Client {self.name} sparsity: {sparsity:.2f}%")

        # Training process
        for epoch in range(self._epoch):
            for step, (x, y) in enumerate(train_loader):
                with torch.no_grad():
                    b_x = x.to(self._device)  # Tensor on GPU
                    b_y = y.to(self._device)  # Tensor on GPU

                with torch.enable_grad():
                    self.model.train()
                    output = self.model(b_x)
                    loss = loss_func(output, b_y.long())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # Apply mask to the model
                    for name, param in self.model.named_parameters():
                        # print(f"Try to applying mask to {name}")
                        if name in self.mask:
                            # print(f"Applying mask to {name}")
                            param.data = param.data * self.mask[name]
        self.loss = loss

        # nonzero_count, total_count, sparsity = self.get_sparsity()
        # print(f"Client {self.name} sparsity: {sparsity:.2f}%")

        return self.model.state_dict(), self.n_data, loss.data.cpu().numpy()
