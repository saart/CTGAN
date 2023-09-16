"""CTGAN module."""
import time

import warnings
from typing import Union, Optional, List
import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, functional
import torch_geometric

from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.base import BaseSynthesizer


class Noise(Module):
    def __init__(
        self, metadata_dim, with_gcn, n_nodes, graph_dim, device, graph_index_to_edges,
        graph_data: Optional[torch.Tensor] = None,
        chain_data: Optional[torch.Tensor] = None,
        trigger_data: Optional[torch.Tensor] = None,
        tx_start_time: torch.Tensor = None,
        noise_embedding_dim=128
    ):
        super().__init__()
        # Config
        self.noise_embedding_dim = noise_embedding_dim
        self.with_gcn = with_gcn
        self.n_nodes = n_nodes
        self.device = device
        self.graph_index_to_edges = graph_index_to_edges
        gcn_node_embedding_size = 1
        if self.with_gcn:
            self.graph_dim = n_nodes * gcn_node_embedding_size
        else:
            self.graph_dim = graph_dim
        self.use_tx_time = tx_start_time is not None
        self.use_metadata = trigger_data is not None

        # Layers
        self.gcn = torch_geometric.nn.GCNConv(1, gcn_node_embedding_size)
        if metadata_dim:
            self.triggering_layer = Sequential(Linear(metadata_dim, 32), torch.nn.ReLU())
        else:
            self.triggering_layer = None
        self.noise_embedding = Linear(
            (
                self.graph_dim
                + 1  # chain
                + (32 if self.use_metadata else 0)
                + (tx_start_time.shape[1] if self.use_tx_time else 0)
            ),
            noise_embedding_dim
        )

        # Data
        self.graph_data = graph_data
        self.chain_data = chain_data
        self.trigger_data = trigger_data
        self.tx_start_time = tx_start_time

    def get_dim(self) -> int:
        return self.noise_embedding_dim

    def forward(self, input_):
        """Apply the Discriminator to the `input_`."""
        assert input_.size()[0] % self.pac == 0
        return self.seq(input_.view(-1, self.pacdim))

    def generate_noise_from_indexes(self, idx: List[int]):
        chain = self.chain_data[idx]
        trigger_data = None if self.trigger_data is None else self.trigger_data[idx]
        graph = self.graph_data[idx]
        tx_start_time = self.tx_start_time[idx] if self.use_tx_time else None

        # TODO: Do we want to use the same layers for generator and discriminator?
        return self.generate_noise_from_metadata(
            batched_graphs=graph, batched_chain=chain, trigger_data=trigger_data,
            tx_start_time=tx_start_time
        )

    def generate_noise_from_metadata(
        self, trigger_data: Optional[torch.Tensor], batched_chain: torch.Tensor,
        batched_graphs: torch.Tensor, tx_start_time: torch.Tensor
    ):
        if self.with_gcn:
            ones = torch.ones((self.n_nodes, 1)).to(self.device)
            unique_gcn = {
                int(graph): self.gcn(ones, self.graph_index_to_edges[int(graph)]).flatten().unsqueeze(0)
                for graph in batched_graphs.unique()
            }
            graph_embedding = torch.concatenate([unique_gcn[int(graph)] for graph in batched_graphs])
        else:
            graph_embedding = torch.zeros((len(batched_graphs), self.graph_dim))
            for row_index, graph in enumerate(batched_graphs):
                graph_embedding[row_index][int(graph)] = 1

        embed_to_noise = torch.cat(
            [graph_embedding, batched_chain.unsqueeze(1)]
            + ([self.triggering_layer(trigger_data)] if self.use_metadata else [])
            + ([tx_start_time] if self.use_tx_time else []),
            dim=1
        ).float()
        return self.noise_embedding(embed_to_noise)


class Discriminator(Module):
    """Discriminator for the CTGAN."""

    def __init__(self, input_dim, discriminator_dim, pac=10):
        super(Discriminator, self).__init__()
        dim = input_dim * pac
        self.pac = pac
        self.pacdim = dim
        seq = []
        for item in list(discriminator_dim):
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item

        seq += [Linear(dim, 1)]
        self.seq = Sequential(*seq)

    def calc_gradient_penalty(self, real_data, fake_data, device='cpu', pac=10, lambda_=10):
        """Compute the gradient penalty."""
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        gradients_view = gradients.reshape(-1, pac * real_data.size(1)).norm(2, dim=1) - 1
        gradient_penalty = ((gradients_view) ** 2).mean() * lambda_

        return gradient_penalty

    def forward(self, input_):
        """Apply the Discriminator to the `input_`."""
        assert input_.size()[0] % self.pac == 0
        return self.seq(input_.view(-1, self.pacdim))


class Residual(Module):
    """Residual layer for the CTGAN."""

    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input_):
        """Apply the Residual layer to the `input_`."""
        out = self.fc(input_)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input_], dim=1)


class Generator(Module):
    """Generator for the CTGAN."""

    def __init__(self, embedding_dim, generator_dim, data_dim):
        super(Generator, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(generator_dim):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, input_):
        """Apply the Generator to the `input_`."""
        data = self.seq(input_)
        return data


class CTGAN(BaseSynthesizer):
    """Conditional Table GAN Synthesizer.

    This is the core class of the CTGAN project, where the different components
    are orchestrated together.
    For more details about the process, please check the [Modeling Tabular data using
    Conditional GAN](https://arxiv.org/abs/1907.00503) paper.

    Args:
        embedding_dim (int):
            Size of the random sample passed to the Generator. Defaults to 128.
        generator_dim (tuple or list of ints):
            Size of the output samples for each one of the Residuals. A Residual Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        discriminator_dim (tuple or list of ints):
            Size of the output samples for each one of the Discriminator Layers. A Linear Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        generator_lr (float):
            Learning rate for the generator. Defaults to 2e-4.
        generator_decay (float):
            Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
        discriminator_lr (float):
            Learning rate for the discriminator. Defaults to 2e-4.
        discriminator_decay (float):
            Discriminator weight decay for the Adam Optimizer. Defaults to 1e-6.
        batch_size (int):
            Number of data samples to process in each step.
        discriminator_steps (int):
            Number of discriminator updates to do for each generator update.
            From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper
            default is 5. Default used is 1 to match original CTGAN implementation.
        log_frequency (boolean):
            Whether to use log frequency of categorical levels in conditional
            sampling. Defaults to ``True``.
        verbose (boolean):
            Whether to have print statements for progress results. Defaults to ``False``.
        epochs (int):
            Number of training epochs. Defaults to 300.
        pac (int):
            Number of samples to group together when applying the discriminator.
            Defaults to 10.
        cuda (bool):
            Whether to attempt to use cuda for GPU computation.
            If this is False or CUDA is not available, CPU will be used.
            Defaults to ``True``.
    """

    def __init__(self, embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256),
                 generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
                 discriminator_decay=1e-6, batch_size=500, discriminator_steps=1,
                 log_frequency=True, verbose=False, epochs=300, pac=10, cuda=True,
                 graph_index_to_edges=None, n_nodes=32, functional_loss=None,
                 functional_loss_freq=None, with_gcn=True, device: torch.device = None,
                 name: str = 'ctgan'):

        assert batch_size % 2 == 0
        self.name = name

        self._embedding_dim = embedding_dim
        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim

        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay

        self._batch_size = batch_size
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._epochs = epochs
        self.pac = pac

        self.functional_loss = functional_loss
        self.functional_loss_freq = functional_loss_freq
        self.graph_index_to_edges = graph_index_to_edges
        self.n_nodes = n_nodes
        self.metadata_dim = 0
        self.with_gcn = with_gcn

        if device:
            pass
        elif not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self._device = torch.device(device)
        print(f"Use device: {self._device}")

        self._transformer = None
        self._data_sampler = None
        self._noise: Optional[Noise] = None
        self._generator = None

    def get_device(self):
        return self._device

    @staticmethod
    def _gumbel_softmax(logits, tau=1., hard=False, eps=1e-10, dim=-1):
        """Deals with the instability of the gumbel_softmax for older versions of torch.

        For more details about the issue:
        https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing

        Args:
            logits […, num_features]:
                Unnormalized log probabilities
            tau:
                Non-negative scalar temperature
            hard (bool):
                If True, the returned samples will be discretized as one-hot vectors,
                but will be differentiated as if it is the soft sample in autograd
            dim (int):
                A dimension along which softmax will be computed. Default: -1.

        Returns:
            Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
        """
        for _ in range(10):
            transformed = functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)
            if not torch.isnan(transformed).any():
                return transformed

        raise ValueError('gumbel_softmax returning NaN.')

    def _apply_activate(self, data):
        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == 'tanh':
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'softmax':
                    ed = st + span_info.dim
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                else:
                    raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')

        return torch.cat(data_t, dim=1)

    def _cond_loss(self, data, c, m):
        """Compute the cross entropy loss on the fixed discrete column."""
        loss = []
        st = 0
        st_c = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if len(column_info) != 1 or span_info.activation_fn != 'softmax':
                    # not discrete column
                    st += span_info.dim
                else:
                    ed = st + span_info.dim
                    ed_c = st_c + span_info.dim
                    tmp = functional.cross_entropy(
                        data[:, st:ed],
                        torch.argmax(c[:, st_c:ed_c], dim=1),
                        reduction='none'
                    )
                    loss.append(tmp)
                    st = ed
                    st_c = ed_c

        loss = torch.stack(loss, dim=1)  # noqa: PD013

        return (loss * m).sum() / data.size()[0]

    def _validate_discrete_columns(self, train_data, discrete_columns):
        """Check whether ``discrete_columns`` exists in ``train_data``.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        if isinstance(train_data, pd.DataFrame):
            invalid_columns = set(discrete_columns) - set(train_data.columns)
        elif isinstance(train_data, np.ndarray):
            invalid_columns = []
            for column in discrete_columns:
                if column < 0 or column >= train_data.shape[1]:
                    invalid_columns.append(column)
        else:
            raise TypeError('``train_data`` should be either pd.DataFrame or np.array.')

        if invalid_columns:
            raise ValueError(f'Invalid columns found: {invalid_columns}')

    # @random_state
    def fit(self, train_data, graph_data, chain_data, tx_start_time=None, discrete_columns=(), metadata_discrete_columns=(), epochs=None, metadata=None):
        """Fit the CTGAN Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        self._validate_discrete_columns(train_data, discrete_columns)

        if epochs is None:
            epochs = self._epochs
        else:
            warnings.warn(
                ('`epochs` argument in `fit` method has been deprecated and will be removed '
                 'in a future version. Please pass `epochs` to the constructor instead'),
                DeprecationWarning
            )

        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)

        train_data = self._transformer.transform(train_data)

        self._metadata_transformer = DataTransformer()
        if metadata is not None:
            self._metadata_transformer.fit(metadata, metadata_discrete_columns)
            metadata = self._metadata_transformer.transform(metadata)

        self._start_time_transformer = DataTransformer()
        if tx_start_time is not None:
            if isinstance(tx_start_time, pd.Series):
                tx_start_time = tx_start_time.to_frame()
            self._start_time_transformer.fit(tx_start_time)
            tx_start_time = self._start_time_transformer.transform(tx_start_time)

        self._data_sampler = DataSampler(
            train_data,
            self._transformer.output_info_list,
            self._log_frequency)
        self.metadata_dim = metadata.shape[1] if metadata is not None else 0

        data_dim = self._transformer.output_dimensions

        self._noise = Noise(
            n_nodes=self.n_nodes,
            metadata_dim=self.metadata_dim,
            graph_dim=graph_data.max() + 1,
            with_gcn=self.with_gcn,
            device=self._device,
            graph_index_to_edges=self.graph_index_to_edges,
            graph_data=torch.Tensor(graph_data.values).to(self._device),
            chain_data=torch.Tensor(chain_data.values).to(self._device),
            trigger_data=torch.from_numpy(metadata).type(torch.float32).to(self._device) if metadata is not None else None,
            tx_start_time=torch.from_numpy(tx_start_time).to(self._device) if tx_start_time is not None else None,
        ).to(self._device)

        self._generator = Generator(
            self._embedding_dim + self._noise.get_dim(),
            self._generator_dim,
            data_dim
        ).to(self._device)

        discriminator = Discriminator(
            data_dim + self._noise.get_dim(),
            self._discriminator_dim,
            pac=self.pac
        ).to(self._device)

        optimizerG = optim.Adam(
            self._generator.parameters(), lr=self._generator_lr, betas=(0.5, 0.9),
            weight_decay=self._generator_decay
        )

        optimizerD = optim.Adam(
            discriminator.parameters(), lr=self._discriminator_lr,
            betas=(0.5, 0.9), weight_decay=self._discriminator_decay
        )
        self._fit_for(train_data, epochs, discriminator, optimizerG, optimizerD)
        return {
            "discriminator": discriminator,
            "optimizerG": optimizerG,
            "optimizerD": optimizerD,
        }

    def _fit_for(self, train_data, epochs, discriminator, optimizerG, optimizerD):
        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        start_time = time.time()
        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        for i in range(epochs):
            for id_ in range(steps_per_epoch):

                for n in range(self._discriminator_steps):
                    fakez = torch.normal(mean=mean, std=std)

                    condvec = self._data_sampler.sample_condvec(self._batch_size)
                    if condvec is None:
                        c1, m1, col, opt, c2 = None, None, None, None, None
                        idx = self._data_sampler.sample_data(self._batch_size, col, opt)
                        real = self._data_sampler._data[idx]
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)

                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        idx = self._data_sampler.sample_data(
                            self._batch_size, col[perm], opt[perm])
                        real = self._data_sampler._data[idx]

                        c1 = self._noise.generate_noise_from_indexes(idx)
                        c2 = c1[perm]
                        fakez = torch.cat([fakez, c1], dim=1)

                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)

                    real = torch.from_numpy(real.astype('float32')).to(self._device)

                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    else:
                        real_cat = real
                        fake_cat = fakeact

                    y_fake = discriminator(fake_cat)
                    y_real = discriminator(real_cat)

                    pen = discriminator.calc_gradient_penalty(
                        real_cat, fake_cat, self._device, self.pac)
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                    optimizerD.zero_grad(set_to_none=False)
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    optimizerD.step()

                fakez = torch.normal(mean=mean, std=std)
                condvec = self._data_sampler.sample_condvec(self._batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)

                    perm = np.arange(self._batch_size)
                    np.random.shuffle(perm)
                    idx = self._data_sampler.sample_data(self._batch_size, col[perm],
                                                         opt[perm])

                    c1 = self._noise.generate_noise_from_indexes(idx)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)

                if c1 is not None:
                    y_fake = discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = discriminator(fakeact)

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1)

                loss_g = -torch.mean(y_fake) + cross_entropy

                optimizerG.zero_grad(set_to_none=False)
                loss_g.backward()
                optimizerG.step()

            if self._verbose:
                print(f'{self.name}-{i+1},', end=' ', flush=True)

            if self.functional_loss_freq and self.functional_loss and (i+1) % self.functional_loss_freq == 0:
                self.functional_loss(i)
        print(f'\n\n{self.name} _fit_for took {time.time() - start_time} seconds\n\n')

    def continue_fit(self, train_data, graph_data, chain_data, tx_start_time, metadata, discriminator, optimizerG, optimizerD):
        train_data = self._transformer.transform(train_data)
        if metadata is not None:
            metadata = self._metadata_transformer.transform(metadata)
        if tx_start_time is not None:
            if isinstance(tx_start_time, pd.Series):
                tx_start_time = tx_start_time.to_frame()
            tx_start_time = self._start_time_transformer.transform(tx_start_time)

        self._noise.graph_data = torch.Tensor(graph_data.values).to(self._device)
        self._noise.chain_data = torch.Tensor(chain_data.values).to(self._device)
        self._noise.trigger_data = torch.from_numpy(metadata).type(torch.float32).to(self._device) if metadata is not None else None
        self._noise.tx_start_time = torch.from_numpy(tx_start_time).type(torch.float32).to(self._device) if tx_start_time is not None else None

        self._data_sampler = DataSampler(
            train_data,
            self._transformer.output_info_list,
            self._log_frequency,
        )
        self._fit_for(train_data, self._epochs, discriminator, optimizerG, optimizerD)

    # @random_state
    def sample_one(self, graph_index: int, chain_index: Optional[int],
                   metadata: Optional[Union[pd.Series, pd.DataFrame]] = None,
                   tx_start_time: Optional[int] = None,
                   columns: Optional[List[str]] = None,
                   normalize_trigger_data: bool = True) -> pd.DataFrame:
        n = 2
        if n == 1:
            raise ValueError('n must be greater than 1')
        graph_index_list = torch.Tensor([graph_index for _ in range(n)])
        chain_index_list = torch.Tensor([chain_index or 0 for _ in range(n)])
        tx_start_time_list = torch.Tensor([tx_start_time for _ in range(n)]) if tx_start_time is not None else None
        metadata_list: Optional[List[np.ndarray]] = None
        if metadata is not None:
            metadata_list = metadata.values.repeat(n, axis=0)
        return self.sample(
            graph_index_list=graph_index_list, chain_index_list=chain_index_list,
            metadata_list=metadata_list, tx_start_time_list=tx_start_time_list, columns=columns,
            normalize_trigger_data=normalize_trigger_data
        )

    def sample(self, graph_index_list: torch.Tensor, chain_index_list: torch.Tensor,
               metadata_list: Optional[List[np.ndarray]] = None,
               tx_start_time_list: Optional[pd.DataFrame] = None,
               columns: Optional[List[str]] = None,
               normalize_trigger_data: bool = True) -> pd.DataFrame:
        """Sample data similar to the training data.

        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        n = len(graph_index_list)

        graph = graph_index_list.to(self._device)

        chain = chain_index_list.type(torch.float32).to(self._device)
        tx_start_time = tx_start_time_list if tx_start_time_list is not None else None
        if metadata_list is not None:
            if normalize_trigger_data:
                metadata_list = torch.from_numpy(self._metadata_transformer.transform(metadata_list))
            metadata = metadata_list.type(torch.float32).to(self._device)
        else:
            assert self.metadata_dim == 0, "Most provide metadata in the metadata-based CTGAN"
            metadata = None

        if tx_start_time is not None:
            if normalize_trigger_data:
                tx_start_time = torch.from_numpy(self._start_time_transformer.transform(tx_start_time))
            tx_start_time = tx_start_time.type(torch.float32).to(self._device)

        steps = (n-1) // self._batch_size + 1
        data = []
        plus_one = False
        for i in range(steps):
            f, t = i * self._batch_size, min((i + 1) * self._batch_size, n)
            if t - f == 1:
                plus_one = True
                f -= 1
            mean = torch.zeros(t - f, self._embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self._device)

            condvec = self._noise.generate_noise_from_metadata(
                batched_graphs=graph[f:t],
                batched_chain=chain[f:t],
                trigger_data=metadata[f:t] if metadata is not None else None,
                tx_start_time=tx_start_time[f:t] if tx_start_time is not None else None
            )

            if condvec is None:
                pass
            else:
                c1 = condvec
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self._generator(fakez)
            fakeact = self._apply_activate(fake)
            if plus_one:
                fakeact = fakeact[1:]
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        return self._transformer.inverse_transform(data, columns=columns)

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        if self._generator is not None:
            self._generator.to(self._device)
