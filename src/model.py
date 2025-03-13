import math

import torch
from torch import nn


def scaled_dot_product_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    masks: torch.Tensor | None,
) -> torch.Tensor:
    """
    Query size must be equal to the key size. Key count and value count must be equal.

    :param queries: [batch size, query count, query size]
    :param keys: [batch size, key count, key size]
    :param values: [batch size, value count, value size]
    :param masks: boolean tensor [batch size, query count, key count], true = allow attention

    :return: output: [batch size, query count, value size]
    """
    assert (
        queries.shape[0] == keys.shape[0] == values.shape[0]
    ), "batch size must be equal for all inputs"
    assert queries.shape[2] == keys.shape[2], "query size and key size must be equal"
    assert keys.shape[1] == values.shape[1], "Key count and value count must be equal"

    # dot-product over all queries and key at once (matrix multiplication) to calculate the compatibility of each query
    # with each key
    # output shape: [batch size, query count, key count]
    compatibility = torch.matmul(
        queries,
        torch.transpose(keys, 1, 2),
    )

    # scaling to avoid large dot products when the key size is large
    key_size = keys.shape[2]
    compatibility = (1.0 / math.sqrt(key_size)) * compatibility

    if masks is not None:
        # set prohibited connections to negative infinity which results in a compatibility of 0 after softmax
        compatibility[masks.logical_not()] = float("-inf")

    # for each query calculates the final weights for every key
    weights = nn.Softmax(dim=2)(compatibility)

    # calculates the results of each query (weighted sum of values)
    # output shape: [batch size, query count, value size]
    result = torch.matmul(weights, values)

    return result


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        heads_count: int,
        model_size: int,
        key_size: int,
        value_size: int,
        device: str,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.heads_count = heads_count

        self.query_projections = nn.ModuleList()
        self.key_projections = nn.ModuleList()
        self.value_projections = nn.ModuleList()
        self.output_projection = nn.Linear(
            heads_count * value_size, model_size, device=device, dtype=dtype
        )

        for _ in range(heads_count):
            self.query_projections.append(
                nn.Linear(model_size, key_size, bias=False, device=device, dtype=dtype)
            )
            self.key_projections.append(
                nn.Linear(model_size, key_size, bias=False, device=device, dtype=dtype)
            )
            self.value_projections.append(
                nn.Linear(
                    model_size, value_size, bias=False, device=device, dtype=dtype
                )
            )

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Key count and value count must be equal.

        :param queries: [batch size, query count, model size]
        :param keys: [batch size, key count, model size]
        :param values: [batch size, value count, model size]
        :param masks: [batch size, query count, key count]

        :return: [batch size, query count, model size]
        """

        intermediate_results = []

        for i in range(self.heads_count):
            intermediate = scaled_dot_product_attention(
                self.query_projections[i](queries),
                self.key_projections[i](keys),
                self.value_projections[i](values),
                masks=masks,
            )
            intermediate_results.append(intermediate)

        output = self.output_projection(torch.concat(intermediate_results, dim=2))

        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(
        self, model_size: int, inner_size: int, device: str, dtype: torch.dtype
    ):
        super().__init__()

        self.first_linear = nn.Linear(
            model_size, inner_size, device=device, dtype=dtype
        )
        self.second_linear = nn.Linear(
            inner_size, model_size, device=device, dtype=dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.second_linear(nn.ReLU()(self.first_linear(x)))


def create_positional_embedding(
    sequence_length: int, model_size: int, device: str, dtype: torch.dtype
) -> torch.Tensor:
    """
    :param sequence_length:
    :param model_size:
    :param device:
    :param dtype:

    :return: [sequence length, model size]
    """
    positions = torch.arange(sequence_length, device=device, dtype=dtype).reshape(
        (sequence_length, 1)
    )
    dimensions = torch.arange(model_size, device=device, dtype=dtype).reshape(
        (1, model_size)
    )

    uneven_mask = torch.arange(model_size, device=device, dtype=dtype) % 2 == 1

    pos_emb = torch.sin(positions / (10000 ** (2 * dimensions / model_size)))
    pos_emb_cos = torch.cos(positions / (10000 ** (2 * dimensions / model_size)))

    pos_emb[:, uneven_mask] = pos_emb_cos[
        :, uneven_mask
    ]  # use cos for every second index in the second dimension

    return pos_emb


def create_attention_mask(sequence_length: int, device: str) -> torch.Tensor:
    """
    Creates the mask tensor for the masked multi-head attention. It's a matrix where the diagonal and lower triangle are
    set to true and the rest to false.

    :param sequence_length:
    :param device:

    :return: mask, shape=[sequence length, sequence length]
    """
    return torch.tril(
        torch.ones((sequence_length, sequence_length), device=device, dtype=torch.bool)
    )


class Embedding(nn.Module):
    def __init__(
        self, dictionary_size: int, model_size: int, device: str, dtype: torch.dtype
    ) -> None:
        super().__init__()

        self.model_size = model_size
        self.embeddings = nn.Parameter(
            torch.randn(dictionary_size, model_size, device=device, dtype=dtype)
        )

    def forward(self, token_indices: torch.Tensor) -> torch.Tensor:
        """
        :param token_indices: dtype=long, shape=[*]
        :return: embeddings: [*, model_size]
        """
        assert token_indices.dtype == torch.long

        return self.embeddings[token_indices, :] * math.sqrt(self.model_size)


class OutputPredictor(nn.Module):
    def __init__(self, embeddings: nn.Parameter) -> None:
        super().__init__()

        self.embeddings = embeddings

    def forward(
        self, output_embeddings: torch.Tensor, output_logits: bool = False
    ) -> torch.Tensor:
        """
        :param output_embeddings: [*, model_size]
        :param output_logits: boolean, if true the output are logits instead of probabilities

        :return: [*, dictionary size]
        """

        logits = output_embeddings.matmul(self.embeddings.transpose(0, 1))

        if output_logits:
            return logits
        else:
            return nn.functional.softmax(logits, dim=-1)


class EncoderBlock(nn.Module):
    def __init__(
        self,
        heads_count: int,
        model_size: int,
        key_size: int,
        value_size: int,
        inner_size: int,
        device: str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()

        self.mha = MultiHeadAttention(
            heads_count, model_size, key_size, value_size, device, dtype
        )
        self.ff = PositionWiseFeedForward(model_size, inner_size, device, dtype)
        self.norm1 = nn.LayerNorm(model_size, device=device, dtype=dtype)
        self.norm2 = nn.LayerNorm(model_size, device=device, dtype=dtype)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """

        :param input_: [batch size, input sequence length, model size]
        :return: [batch size, input sequence length, model size]
        """
        intermediate = self.mha(input_, input_, input_)  # self-attention
        intermediate = intermediate + input_  # residual connection
        intermediate = self.norm1(intermediate)  # normalization

        intermediate2 = self.ff(intermediate)  # point-wise feed-forward
        intermediate2 = intermediate2 + intermediate  # residual connection
        intermediate2 = self.norm2(intermediate2)  # normalization

        return intermediate2


class DecoderBlock(nn.Module):
    def __init__(
        self,
        heads_count: int,
        model_size: int,
        key_size: int,
        value_size: int,
        inner_size: int,
        device: str,
        dtype: torch.dtype,
    ):
        super().__init__()

        self.mmha = MultiHeadAttention(
            heads_count, model_size, key_size, value_size, device, dtype
        )
        self.norm1 = nn.LayerNorm(model_size, device=device, dtype=dtype)

        self.mha = MultiHeadAttention(
            heads_count, model_size, key_size, value_size, device, dtype
        )
        self.norm2 = nn.LayerNorm(model_size, device=device, dtype=dtype)

        self.ff = PositionWiseFeedForward(model_size, inner_size, device, dtype)
        self.norm3 = nn.LayerNorm(model_size, device=device, dtype=dtype)

    def forward(
        self,
        input_: torch.Tensor,
        encoder_stack_output: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """

        :param input_: [batch size, output sequence length, model size]
        :param encoder_stack_output: [batch size, input sequence length, model size]
        :param masks: [batch size, output sequence length, output sequence length]
        :return: [batch size, output sequence length, model size]
        """
        # shape=[batch size, output sequence length, model size]
        intermediate = self.mmha(input_, input_, input_, masks)  # masked self-attention
        intermediate = intermediate + input_  # residual connection
        intermediate = self.norm1(intermediate)  # normalization

        # shape=[batch size, output sequence length, model size]
        intermediate2 = self.mha(
            intermediate,
            encoder_stack_output,
            encoder_stack_output,
        )  #  attention from encoder
        intermediate2 = intermediate2 + intermediate  # residual connection
        intermediate2 = self.norm2(intermediate2)  # normalization

        # shape=[batch size, output sequence length, model size]
        intermediate3 = self.ff(intermediate2)  # point-wise feed-forward
        intermediate3 = intermediate3 + intermediate2  # residual connection
        intermediate3 = self.norm3(intermediate3)  # normalization

        return intermediate3


class Transformer(nn.Module):
    def __init__(
        self,
        block_count: int,
        heads_count: int,
        model_size: int,
        key_size: int,
        value_size: int,
        inner_size: int,
        dictionary_size: int,
        device: str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.model_size = model_size
        self.device = device
        self.dtype = dtype

        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(
                    heads_count,
                    model_size,
                    key_size,
                    value_size,
                    inner_size,
                    device,
                    dtype,
                )
                for _ in range(block_count)
            ]
        )
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    heads_count,
                    model_size,
                    key_size,
                    value_size,
                    inner_size,
                    device,
                    dtype,
                )
                for _ in range(block_count)
            ]
        )
        self.embedding_layer = Embedding(dictionary_size, model_size, device, dtype)
        self.output_layer = OutputPredictor(self.embedding_layer.embeddings)

    def forward(
        self,
        input_tokens: torch.Tensor,
        output_tokens: torch.Tensor,
        output_logits: bool = False,
    ) -> torch.Tensor:
        """

        :param input_tokens: dtype=long, shape=[batch size, input sequence length]
        :param output_tokens: dtype=long, shape=[batch size, output sequence length]
        :param output_logits: boolean, if true the output are logits instead of probabilities

        :return: output token probabilities: [batch size, dictionary size]
        """
        # input token processing
        input_sequence_length = input_tokens.shape[-1]
        # shape=[batch size, input sequence length, model size]
        input_embeddings: torch.Tensor = self.embedding_layer(input_tokens)
        input_embeddings += create_positional_embedding(
            input_sequence_length, self.model_size, self.device, dtype=self.dtype
        )

        # encoder stack
        encoder_stack_output: torch.Tensor = input_embeddings

        for encoder_block in self.encoder_blocks:
            encoder_stack_output = encoder_block(encoder_stack_output)

        # output token processing
        output_sequence_length = output_tokens.shape[-1]
        # shape=[batch size, output sequence length, model size]
        output_embeddings = self.embedding_layer(output_tokens)
        output_embeddings += create_positional_embedding(
            output_sequence_length, self.model_size, self.device, self.dtype
        )

        # decoder stack
        decoder_block_output = output_embeddings
        mask_shape = tuple(output_tokens.shape[:-1]) + (
            output_sequence_length,
            output_sequence_length,
        )
        # shape=[batch size, output sequence length, output sequence length]
        attention_mask = create_attention_mask(
            output_sequence_length, self.device
        ).expand(mask_shape)

        for decoder_block in self.decoder_blocks:
            decoder_block_output = decoder_block(
                decoder_block_output, encoder_stack_output, attention_mask
            )

        # output layers
        predictions = self.output_layer(decoder_block_output, output_logits)

        return predictions
