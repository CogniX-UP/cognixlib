"""Contains utilities and structures for loading all the nodes as a package hierarchy"""
from cognixcore import Node
from cognixcore.node import get_versioned_nodes

def get_package_nodes(module) -> list[Node]:
    """Returns all the versioned nodes defined in the module given by the parameter"""
    return get_versioned_nodes(module.__name__)

cognix_package: dict[str, list[Node]]
"""
A container for the structure of packages in CogniX

Essentially a dictionary where keys are package name and
values are nodes. The package names can be arbitrarily nested

Example:

    pkg_one: [...]
    pkg_one.sub_pkg: [...]
    . . .
"""

# We will import the packages manually here and create the package structure
# The process is pretty much automated up this point, so we prefer to add them
# manually to have a specific injection point

# Perhaps we'll take a look at automating it even further later

from .classification import _nodes as classification
from .file import _nodes as file
from .stream import _nodes as stream
from .utility import _nodes as utility
from .test import _nodes as test
from .preprocessing import _nodes as preprocessing
from .feature_extraction import _nodes as feature_extraction

cognix_package: dict[str, list[type[Node]]] = {
    'file': get_package_nodes(file),
    'stream': get_package_nodes(stream),
    'util': get_package_nodes(utility),
    'test': get_package_nodes(test),
    'preprocessing': get_package_nodes(preprocessing),
    'feature_extration': get_package_nodes(feature_extraction),
    'classification': get_package_nodes(classification)
}
