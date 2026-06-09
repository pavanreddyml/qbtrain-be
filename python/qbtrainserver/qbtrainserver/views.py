# views.py
import gc

from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response


# --- List View (GET /settings/) ---
@api_view(['GET'])
def health(request):
    try:
        return Response({'message': 'Service is healthy'}, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({'error': f'An unexpected error occurred: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


def _gpu_mem_snapshot(torch):
    """Best-effort VRAM stats. Returns a dict that's always JSON-safe."""
    try:
        if not torch.cuda.is_available():
            return {"cuda_available": False}
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        allocated = torch.cuda.memory_allocated(idx)
        reserved = torch.cuda.memory_reserved(idx)
        total = props.total_memory
        return {
            "cuda_available": True,
            "device_index": idx,
            "device_name": props.name,
            "allocated_bytes": int(allocated),
            "reserved_bytes": int(reserved),
            "total_bytes": int(total),
            "free_bytes": int(max(0, total - reserved)),
        }
    except Exception as exc:
        return {"cuda_available": True, "error": f"failed to read VRAM stats: {exc}"}


@api_view(['POST'])
def clear_gpu_memory(request):
    """Release any cached CUDA allocations the framework is holding.

    Sequence:
        1. `gc.collect()`              — drop dangling Python refs to tensors
        2. `torch.cuda.empty_cache()`  — release reserved-but-unused blocks
        3. `torch.cuda.ipc_collect()`  — close IPC handles

    Note: this does NOT free memory that's still referenced by live Python
    objects (e.g. a loaded model in another endpoint's module). It only
    returns to the OS the reserved-but-unallocated pool that PyTorch holds.
    """
    try:
        try:
            import torch
        except Exception as exc:  # torch not installed
            return Response(
                {
                    "ok": False,
                    "error": f"PyTorch is not available on this server: {exc}",
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        before = _gpu_mem_snapshot(torch)

        if not torch.cuda.is_available():
            return Response(
                {
                    "ok": False,
                    "error": "CUDA is not available on this server.",
                    "before": before,
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        gc.collect()
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

        after = _gpu_mem_snapshot(torch)
        freed = max(0, before.get("reserved_bytes", 0) - after.get("reserved_bytes", 0))
        return Response(
            {
                "ok": True,
                "freed_bytes": int(freed),
                "before": before,
                "after": after,
            },
            status=status.HTTP_200_OK,
        )
    except Exception as exc:
        return Response(
            {"ok": False, "error": f"An unexpected error occurred: {exc}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )