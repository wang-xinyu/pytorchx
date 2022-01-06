import torch
from torch.autograd import Variable


class LinearRegressionModel(torch.nn.Module):
    """
    Linear Regression Model:
        * - input a number
        * - output its double
    """

    def __init__(self):
        """
        Initialize the model
        """
        super(LinearRegressionModel, self).__init__()

        # A single linear layer with one weight and one bias
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        """
        Forward function of linear model
        :param x: input
        :return y_pred: predictions
        """
        y_pred = self.linear(x)
        return y_pred


def get_dataset():
    """
    Create a simple dataset
        * - x: numbers
        * - y: the labels as 2x of numbers
    :return dataset:
    """
    print("[INFO]: Building dataset...")
    x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
    y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))
    return x_data, y_data


def get_hyperparams(model):
    """
    Create some configurations for model
    :param model:
    :return loss_function and optimizer:
    """
    print("[INFO]: Preparing configurations...")
    criterion = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    return criterion, optimizer


def train(model=None,
          x=None,
          y=None,
          criterion=None,
          optimizer=None):
    """
    Train the given model
    :param model: a pytorch model
    :param x: dataset
    :param y: labels
    :param criterion: loss function
    :param optimizer: optimization technique
    :return:
    """
    print("[INFO]: Starting training...\n")
    for epoch in range(500):
        # Forward pass: Compute predicted y by passing x to the model
        pred_y = model(x)

        # Compute and print loss
        loss = criterion(pred_y, y)

        # Zero gradients,
        optimizer.zero_grad()

        # perform a backward pass
        loss.backward()

        # update the weights.
        optimizer.step()

        # print loss after 50 epochs
        if epoch % 50 == 0:
            print('[INFO]: Epoch {}, Loss {}'.format(epoch, loss.item()))


def check_inference(mlp_model):
    """
    Check inference on the given model, currently with a static input
    :param mlp_model:
    :return:
    """
    new_var = Variable(torch.Tensor([[4.0]]))
    print("\n[INFO]: Predicted (after training) \n\tinput: ", 4,"\n\toutput:", mlp_model(new_var).item())


def main():
    # initialize the model
    mlp_model = LinearRegressionModel()
    # create the dataset
    x, y = get_dataset()
    # prepare configs
    criterion, optim = get_hyperparams(mlp_model)
    # train the model
    train(model=mlp_model,
          x=x, y=y,
          criterion=criterion, optimizer=optim)
    # check inference
    check_inference(mlp_model)
    # save the model
    torch.save(mlp_model, "mlp.pth")


if __name__ == "__main__":
    main()

