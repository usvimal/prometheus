"""System monitoring tools - CPU, memory, disk, network stats."""

from __future__ import annotations

import os
import platform
import time
from typing import Any, Dict, List, Optional

from prometheus.tools.registry import ToolContext, ToolEntry


def _get_cpu_info() -> Dict[str, Any]:
    """Get CPU usage and info."""
    try:
        import psutil
        return {
            "percent": psutil.cpu_percent(interval=0.1),
            "count": psutil.cpu_count(),
            "count_logical": psutil.cpu_count(logical=True),
        }
    except ImportError:
        # Fallback: parse /proc or use platform
        return {
            "platform": platform.system(),
            "processor": platform.processor(),
        }


def _get_memory_info() -> Dict[str, Any]:
    """Get memory usage."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return {
            "total_mb": round(mem.total / (1024 * 1024), 1),
            "available_mb": round(mem.available / (1024 * 1024), 1),
            "used_mb": round(mem.used / (1024 * 1024), 1),
            "percent": mem.percent,
            "swap_total_mb": round(swap.total / (1024 * 1024), 1),
            "swap_used_mb": round(swap.used / (1024 * 1024), 1),
            "swap_percent": swap.percent,
        }
    except ImportError:
        return {"error": "psutil not available"}


def _get_disk_info() -> Dict[str, Any]:
    """Get disk usage."""
    try:
        import psutil
        partitions = psutil.disk_partitions()
        result = []
        for partition in partitions:
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                result.append({
                    "device": partition.device,
                    "mountpoint": partition.mountpoint,
                    "fstype": partition.fstype,
                    "total_gb": round(usage.total / (1024 ** 3), 1),
                    "used_gb": round(usage.used / (1024 ** 3), 1),
                    "free_gb": round(usage.free / (1024 ** 3), 1),
                    "percent": usage.percent,
                })
            except PermissionError:
                continue
        return {"partitions": result}
    except ImportError:
        return {"error": "psutil not available"}


def _get_network_info() -> Dict[str, Any]:
    """Get network I/O stats."""
    try:
        import psutil
        net = psutil.net_io_counters()
        return {
            "bytes_sent_mb": round(net.bytes_sent / (1024 * 1024), 2),
            "bytes_recv_mb": round(net.bytes_recv / (1024 * 1024), 2),
            "packets_sent": net.packets_sent,
            "packets_recv": net.packets_recv,
            "errin": net.errin,
            "errout": net.errout,
            "dropin": net.dropin,
            "dropout": net.dropout,
        }
    except ImportError:
        return {"error": "psutil not available"}


def _get_uptime() -> Dict[str, Any]:
    """Get system uptime."""
    try:
        import psutil
        boot_time = psutil.boot_time()
        uptime_sec = time.time() - boot_time
        hours = int(uptime_sec // 3600)
        minutes = int((uptime_sec % 3600) // 60)
        return {
            "uptime_seconds": round(uptime_sec, 1),
            "uptime_human": f"{hours}h {minutes}m",
            "boot_time": boot_time,
        }
    except ImportError:
        return {"error": "psutil not available"}


def _system_monitor(ctx: ToolContext, category: str = "all") -> str:
    """Get system resource usage metrics.
    
    Categories: all, cpu, memory, disk, network, uptime
    """
    import json
    
    if category == "cpu" or category == "all":
        cpu = _get_cpu_info()
    else:
        cpu = None
        
    if category == "memory" or category == "all":
        memory = _get_memory_info()
    else:
        memory = None
        
    if category == "disk" or category == "all":
        disk = _get_disk_info()
    else:
        disk = None
        
    if category == "network" or category == "all":
        network = _get_network_info()
    else:
        network = None
        
    if category == "uptime" or category == "all":
        uptime = _get_uptime()
    else:
        uptime = None
    
    result = {
        "platform": platform.system(),
        "python_version": platform.python_version(),
        "timestamp": time.time(),
    }
    
    if cpu is not None:
        result["cpu"] = cpu
    if memory is not None:
        result["memory"] = memory
    if disk is not None:
        result["disk"] = disk
    if network is not None:
        result["network"] = network
    if uptime is not None:
        result["uptime"] = uptime
    
    return json.dumps(result, indent=2)


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("system_monitor", {
            "name": "system_monitor",
            "description": "Get system resource usage: CPU, memory, disk, network, uptime. Categories: all, cpu, memory, disk, network, uptime. Useful for self-awareness and monitoring.",
            "parameters": {"type": "object", "properties": {
                "category": {"type": "string", "description": "Resource category to query: all, cpu, memory, disk, network, uptime", "default": "all"}
            }, "required": []},
        }, _system_monitor),
    ]
