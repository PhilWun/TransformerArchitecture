import torch
from torch import nn
from loguru import logger

from .model import Transformer


def get_device() -> str:
    """
    Returns the device name of the default accelerator or cpu if no accelerator is available.

    :return: device name
    """
    if torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator().type
    else:
        device = "cpu"

    logger.info(f"Using device: {device}")

    return device


def transformer_training():
    block_count = 2
    heads_count = 8
    model_size = 64
    key_size = 8
    value_size = 8
    inner_size = 256
    dictionary_size = 16
    input_sequence_length = 8
    target_sequence_length = 8
    batch_size = 8
    learning_rate = 0.002

    device = get_device()
    dtype = torch.bfloat16

    transformer = Transformer(
        block_count,
        heads_count,
        model_size,
        key_size,
        value_size,
        inner_size,
        dictionary_size,
        device,
        dtype,
    )

    # token with ID 0 is the output padding token and is not used otherwise
    input_tokens = torch.randint(
        1,
        dictionary_size,
        (batch_size, input_sequence_length),
        dtype=torch.long,
        device=device,
    )
    # length + 1 because we add the output padding token to the front
    target_tokens = torch.randint(
        1,
        dictionary_size,
        (batch_size, target_sequence_length + 1),
        dtype=torch.long,
        device=device,
    )
    target_tokens[:, 0] = 0  # sets the first token to the output padding token

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(transformer.parameters(), lr=learning_rate)
    transformer.train(True)

    for i in range(10000):
        # uses the padded target tokens without the last token as the "output" that is used as input to the decoder stack
        output_logits: torch.Tensor = transformer(
            input_tokens, target_tokens[:, 0:-1], output_logits=True
        )
        # uses all the target tokens, without the padding token at the beginning, as targets
        loss = loss_fn(output_logits.transpose(1, 2), target_tokens[:, 1:])

        if i % 100 == 0:
            logger.info(f"loss: {loss.item()}")

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


if __name__ == "__main__":
    transformer_training()
