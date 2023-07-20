import os
from typing import Optional, List


def subfiles(
    folder: str,
    join: bool = True,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    sort: bool = True,
) -> List[str]:
    """Return all the files that are in a specific directory: Possible to filter for certain prefix, suffix and to sort
    Args:
        folder (str): The folder in which the files are
        join (bool): Whether to return path with folder or only filename
        prefix (str, optional): Prefix to filter the files. Only files with this prefix will be returned.
        suffix (str, optional): Suffix to filter the files. Only files with this suffix will be returned.
        sort (bool): Whether to sort the files.

    Returns:
        res [List[str]]: List with filtered files of the directory
    """
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [
        l(folder, i)
        for i in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, i))
        and (prefix is None or i.startswith(prefix))
        and (suffix is None or i.endswith(suffix))
    ]
    if sort:
        res.sort()
    return res
