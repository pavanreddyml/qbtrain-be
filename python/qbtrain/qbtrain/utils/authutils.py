# qbtrain/utils/authutils.py
from __future__ import annotations

from typing import AbstractSet, Iterable, Literal, Mapping, Optional, Set

from qbtrain.exceptions import PermissionError

Access = Literal["read", "write"]


class Authorizer:
    def __init__(
        self,
        *,
        read_resources_by_perm: Mapping[str, AbstractSet[str]],
        write_resources_by_perm: Mapping[str, AbstractSet[str]],
        bypass_permission: str,
        allow_reads_without_resources: bool = True,
        allow_writes_without_resources: bool = False,
        imply_write_satisfies_read: bool = False,
    ) -> None:
        self._read = {p: {r.lower() for r in (rs or set())} for p, rs in read_resources_by_perm.items()}
        self._write = {p: {r.lower() for r in (rs or set())} for p, rs in write_resources_by_perm.items()}
        self._bypass = bypass_permission
        self._allow_reads_without_resources = bool(allow_reads_without_resources)
        self._allow_writes_without_resources = bool(allow_writes_without_resources)
        self._imply_write_satisfies_read = bool(imply_write_satisfies_read)

    def authorize(self, *, access: Access, resources: Iterable[str], permissions: Iterable[str]) -> Access:
        perms = {str(p) for p in (permissions or []) if p is not None}
        if self._bypass and self._bypass in perms:
            return access

        res = {str(r).strip().lower() for r in (resources or []) if str(r).strip()}
        if not res:
            if access == "read" and self._allow_reads_without_resources:
                return access
            if access == "write" and self._allow_writes_without_resources:
                return access
            raise PermissionError("Request references no resources; blocked by policy.")

        allowed = self._allowed_resources(perms=perms, access=access)
        missing = sorted(r for r in res if r not in allowed)
        if missing:
            raise PermissionError(
                f"Permission denied for {access} on resources: {missing}. "
                f"Provided permissions: {sorted(perms)}"
            )

        return access
    
    def get_permissions_access(self, fmt: Literal["str", "dict"] = "str") -> dict | str:
        all_read = set().union(*self._read.values()) if self._read else set()
        all_write = set().union(*self._write.values()) if self._write else set()

        read_map = dict(self._read)
        write_map = dict(self._write)

        if self._bypass:
            read_map[self._bypass] = all_read
            write_map[self._bypass] = all_write

        if fmt == "str":
            out_str = ""
            out_str += "Read Permissions:\n"
            for perm, resources in read_map.items():
                out_str += f"- {perm}: {sorted(resources)}\n"
            out_str += "Write Permissions:\n"
            for perm, resources in write_map.items():
                out_str += f"- {perm}: {sorted(resources)}\n"
            return out_str
        elif fmt == "dict":
            return {"read": read_map, "write": write_map}
        else:
            raise ValueError(f"Invalid format: {fmt}. Must be 'str' or 'dict'.")

    def _allowed_resources(self, *, perms: Set[str], access: Access) -> Set[str]:
        if access == "read" and self._imply_write_satisfies_read:
            perms = perms | {self._read_equivalent_of_write(p) for p in perms if p.endswith(".Write")}

        src = self._read if access == "read" else self._write
        out: Set[str] = set()
        for p in perms:
            out |= src.get(p, set())
        return out

    @staticmethod
    def _read_equivalent_of_write(write_perm: str) -> str:
        return write_perm[:-6] + ".Read" if write_perm.endswith(".Write") else write_perm
