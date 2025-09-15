import os
import numpy as np
import pytest
import extraer_prototipos as ep

DATASET_DIR = "dataset"

@pytest.mark.parametrize("method", ["kmedoids", "topk", "kmeans"])
def test_extract_representative_prototypes(method):
    # Ejecutar extracción con cada método
    protos_bip, labels, protos_bin, shape = ep.extract_representative_prototypes(
        DATASET_DIR, k_per_class=2, method=method, random_state=42
    )

    # 1) Debe devolver al menos un prototipo por clase
    assert len(labels) > 0, f"{method} no devolvió etiquetas"
    assert protos_bip.shape[0] == len(labels)

    # 2) Dimensiones consistentes
    assert protos_bin.shape == protos_bip.shape, "Mismatch bin/bip shape"

    # 3) Valores correctos
    assert np.all(np.isin(protos_bin, [0, 1]))
    assert np.all(np.isin(protos_bip, [-1.0, 1.0]))

    # 4) Shape de muestras
    assert shape is not None and len(shape) == 2

    # 5) Probar ascii_show
    ascii_repr = ep.ascii_show(protos_bip[0], shape)
    assert isinstance(ascii_repr, str)
    assert len(ascii_repr.splitlines()) == shape[0]

def test_empty_class(tmp_path):
    # Crear dataset vacío temporal
    empty_dir = tmp_path / "empty_class"
    empty_dir.mkdir()

    protos_bip, labels, protos_bin, shape = ep.extract_representative_prototypes(
        str(tmp_path), k_per_class=2, method="kmedoids"
    )
    assert protos_bip.shape[0] == 0
    assert labels == []
