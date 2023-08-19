"""CTGAN module."""

import warnings
from typing import NamedTuple, Union, Optional, List
import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, functional
import torch_geometric
# import wandb

from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.base import BaseSynthesizer, random_state


class Graph(NamedTuple):
    graph_index: Union[int, str]
    x: torch.Tensor
    edge_index: torch.Tensor


class Noise(Module):
    def __init__(
        self, metadata_dim, with_gcn, n_nodes, graph_dim, device, graph_index_to_edges,
        graph_data: Optional[torch.Tensor] = None,
        chain_data: Optional[torch.Tensor] = None,
        metadata: Optional[torch.Tensor] = None,
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

        # Layers
        self.gcn = torch_geometric.nn.GCNConv(1, gcn_node_embedding_size, cached=True)
        self.metadata_layer = Sequential(Linear(1 + metadata_dim, 32), torch.nn.ReLU())
        self.timestamp_layer = Linear(1, 1)
        self.gcn_metadata_embedding = Linear(self.graph_dim + 32, noise_embedding_dim)

        # Data
        self.graph_data = graph_data
        self.chain_data = chain_data
        self.metadata = metadata

    def get_dim(self) -> int:
        return self.noise_embedding_dim

    def forward(self, input_):
        """Apply the Discriminator to the `input_`."""
        assert input_.size()[0] % self.pac == 0
        return self.seq(input_.view(-1, self.pacdim))

    def generate_noise_from_indexes(self, idx: List[int]):
        chain = self.chain_data[idx]
        metadata = None if self.metadata is None else self.metadata[idx]
        graph = None if self.graph_data is None else self.graph_data[idx]

        ones = torch.ones((self.n_nodes, 1)).to(self.device)
        graph = [Graph(graph_index.item(), ones, self.graph_index_to_edges[graph_index.item()])
                 for graph_index in graph]

        # TODO: Do we want to use the same layers for generator and discriminator?
        return self.generate_noise_from_metadata(
            batched_graphs=graph, batched_chain=chain, batched_metadata=metadata
        )

    def generate_noise_from_metadata(self, batched_metadata, batched_chain, batched_graphs):
        return self.generate_noise(
            batched_graphs=batched_graphs, batched_chain=batched_chain,
            batched_metadata=batched_metadata, gcn=self.gcn,
            metadata_layer=self.metadata_layer, timestamp_layer=self.timestamp_layer,
            gcn_metadata_embedding=self.gcn_metadata_embedding, with_gcn=self.with_gcn,
            graph_dim=self.graph_dim
        )

    @staticmethod
    def generate_noise(batched_graphs, batched_chain, batched_metadata, gcn, metadata_layer,
                       timestamp_layer, gcn_metadata_embedding, with_gcn, graph_dim):
        if with_gcn:
            unique_graphs = {graph.graph_index: graph for graph in batched_graphs}
            unique_gcn = {graph.graph_index: gcn(graph.x, graph.edge_index).flatten().unsqueeze(0) for graph in
                          unique_graphs.values()}
            graph_embedding = torch.concatenate([unique_gcn[graph.graph_index] for graph in batched_graphs])
        else:
            graph_embedding = torch.zeros((len(batched_graphs), graph_dim))
            for row_index, graph in enumerate(batched_graphs):
                graph_embedding[row_index][int(graph.graph_index)] = 1

        batched_chain = batched_chain.unsqueeze(1)
        all_metadata = torch.cat([batched_chain, batched_metadata], dim=1) if batched_metadata is not None else batched_chain
        embed_to_noise = torch.cat([graph_embedding, metadata_layer(all_metadata)], dim=1)
        return gcn_metadata_embedding(embed_to_noise)


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
                 functional_loss_freq=None, with_gcn=True, device: torch.device = None):

        assert batch_size % 2 == 0

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
        self._noise = None
        self._generator = None
        self._discriminator = None
        self.optimizerG = None
        self.optimizerD = None

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

    @random_state
    def fit(self, train_data, graph_data, chain_data, discrete_columns=(), metadata_discrete_columns=(), epochs=None, metadata=None):
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
            metadata = torch.from_numpy(metadata).type(torch.float32).to(self._device)

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
            metadata=metadata
        ).to(self._device)

        self._generator = Generator(
            self._embedding_dim + self._noise.get_dim(),
            self._generator_dim,
            data_dim
        ).to(self._device)
        # wandb.watch(self._generator)

        self._discriminator = Discriminator(
            data_dim + self._noise.get_dim(),
            self._discriminator_dim,
            pac=self.pac
        ).to(self._device)
        # wandb.watch(self._discriminator)
        self._fit_for(train_data, epochs)

    def _fit_for(self, train_data, epochs):
        self.optimizerG = self.optimizerG or optim.Adam(
            self._generator.parameters(), lr=self._generator_lr, betas=(0.5, 0.9),
            weight_decay=self._generator_decay
        )

        self.optimizerD = self.optimizerD or optim.Adam(
            self._discriminator.parameters(), lr=self._discriminator_lr,
            betas=(0.5, 0.9), weight_decay=self._discriminator_decay
        )

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        loss_g, loss_d = None, None
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

                    y_fake = self._discriminator(fake_cat)
                    y_real = self._discriminator(real_cat)

                    pen = self._discriminator.calc_gradient_penalty(
                        real_cat, fake_cat, self._device, self.pac)
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                    self.optimizerD.zero_grad(set_to_none=False)
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    self.optimizerD.step()

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
                    y_fake = self._discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = self._discriminator(fakeact)

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1)

                loss_g = -torch.mean(y_fake) + cross_entropy

                self.optimizerG.zero_grad(set_to_none=False)
                loss_g.backward()
                self.optimizerG.step()

            if self._verbose:
                print(f'Epoch {i+1}, Loss G: {loss_g.detach().cpu(): .4f},'  # noqa: T001
                      f'Loss D: {loss_d.detach().cpu(): .4f}',
                      flush=True)

            if self.functional_loss_freq and self.functional_loss and (i+1) % self.functional_loss_freq == 0:
                self.functional_loss(i)

    def continue_fit(self, train_data, epochs=None):
        self._data_sampler = DataSampler(
            train_data,
            self._transformer.output_info_list,
            self._log_frequency,
        )
        self._fit_for(train_data, epochs)

    @random_state
    def sample(self, n, graph_index, chain_index, metadata=None, should_pick_best=True) -> pd.DataFrame:
        """Sample data similar to the training data.

        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        n = n or self._batch_size
        edges = self.graph_index_to_edges[graph_index].to(self._device)
        ones = torch.ones((self.n_nodes, 1)).to(self._device)
        graph = [Graph(x=ones, edge_index=edges, graph_index=graph_index) for _ in range(n)]
        chain = torch.tensor([(chain_index or 0) for _ in range(n)]).type(torch.float32).to(self._device)
        if metadata is not None:
            metadata = self._metadata_transformer.transform(metadata)
            metadata = metadata.repeat(len(graph), axis=0)
            metadata = torch.from_numpy(metadata).type(torch.float32).to(self._device)
        else:
            assert self.metadata_dim == 0, "Most provide metadata in the metadata-based CTGAN"

        mean = torch.zeros(n, self._embedding_dim)
        std = mean + 1
        fakeact = None
        for _ in range(10):
            fakez = torch.normal(mean=mean, std=std).to(self._device)
            c1 = self._noise.generate_noise_from_metadata(
                batched_graphs=graph, batched_chain=chain, batched_metadata=metadata
            )
            fakez = torch.cat([fakez, c1], dim=1)
            fake = self._generator(fakez)
            fakeact = self._apply_activate(fake)
            if not should_pick_best:
                break
            normalized: pd.DataFrame = self._transformer.inverse_transform(fakeact.detach().cpu().numpy(), columns={"graph", "chain"})
            relevant_indexes: pd.Series = normalized["graph"].eq(graph_index)
            if chain_index is not None:
                relevant_indexes &= (normalized["chain"] == chain_index)
            filtered_normalized = normalized[relevant_indexes]
            if len(filtered_normalized) > 0:
                relevant_fakeact = torch.cat([
                    fakeact[relevant_indexes].repeat_interleave(self.pac, dim=0),
                    c1[relevant_indexes].repeat_interleave(self.pac, dim=0)
                ], dim=1)
                y_fake = self._discriminator(relevant_fakeact).detach().cpu().numpy()
                best_idx = np.abs(y_fake).argmin()
                fakeact = relevant_fakeact[best_idx:best_idx+1]
                break
        return self._transformer.inverse_transform(fakeact.detach().cpu().numpy())

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        if self._generator is not None:
            self._generator.to(self._device)
