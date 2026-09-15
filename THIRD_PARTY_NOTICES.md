# Third-party code and resources

The project code retains its existing [MIT license](LICENSE).

`src/publicmodels/mmt_vendor/` contains excerpts from
[salu133445/mmt](https://github.com/salu133445/mmt), commit
`87a8e26168c0407439e5c83e68c16deb5cac8c67`, copyright 2022 Hao-Wen Dong.
Its [upstream MIT license](src/publicmodels/mmt_vendor/LICENSE) is included.
The excerpts remove command-line entry points and unused heavyweight imports;
the model classes and selected encoding function bodies retain upstream behavior.
The adapter tests check encoding and model integration; checkpoint-based tests
require externally downloaded weights.

Datasets and pretrained weights are **not redistributed** in this package.
Obtain them from the upstream repositories and model cards linked in
[Public data and checkpoints](README.md#public-data-and-checkpoints); their terms apply independently of this
repository's code license. Installed Python dependencies retain their own licenses.
